"""
3D Ewald summation on latent charges + latent atomic dipoles (smeared kernel,
same conventions as ewald.py). S(k) = sum_i (q_i + i k.mu_i) e^{ik.r_i}.
BatchedEwaldQU: pure torch reference. TritonEwaldQU: Triton kernels with
analytic first and second derivatives (forces / stress / BEC training).
"""
from typing import Optional

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def _kgrid(cell, dl, k_sq_max, device, dtype):
    # flat per-graph k list (hemisphere), as TritonEwald
    n_graphs = cell.shape[0]
    Grec = 2.0 * torch.pi * torch.linalg.inv(cell).transpose(1, 2)
    Nk = torch.clamp((torch.norm(cell, dim=2) / dl).to(torch.int64), min=1)
    dims = 2 * Nk + 1
    Pbox = dims.prod(1)
    kgraph = torch.repeat_interleave(torch.arange(n_graphs, device=device), Pbox)
    box_off = torch.cumsum(Pbox, 0) - Pbox
    bl = torch.arange(kgraph.shape[0], device=device) - box_off[kgraph]
    dg = dims[kgraph]
    i0 = bl // (dg[:, 1] * dg[:, 2])
    rem = bl % (dg[:, 1] * dg[:, 2])
    i1 = rem // dg[:, 2]
    i2 = rem % dg[:, 2]
    nvec = (torch.stack([i0, i1, i2], -1) - Nk[kgraph]).to(dtype)
    kvec = torch.einsum('md,mde->me', nvec, Grec[kgraph])
    k_sq = (kvec ** 2).sum(-1)
    sph = (k_sq > 0) & (k_sq <= k_sq_max)
    nz = (nvec != 0).to(torch.int64)
    fnz = torch.argmax(nz, dim=1)
    sign = torch.gather(nvec, 1, fnz.unsqueeze(1)).squeeze(1)
    allz = (nvec == 0).all(dim=1)
    hemi = (sign > 0) | allz
    factors = torch.where(allz, torch.ones_like(k_sq), torch.full_like(k_sq, 2.0))
    keep = sph & hemi
    return kgraph[keep], kvec[keep], k_sq[keep], factors[keep]


class BatchedEwaldQU(nn.Module):
    """forward(q, mu, r, cell, batch) -> [n_graphs]  (torch reference)"""

    def __init__(self, dl=2.0, sigma=1.0, remove_self_interaction=True, norm_factor=90.4756):
        super().__init__()
        self.dl = dl
        self.sigma = sigma
        self.sigma_sq_half = sigma ** 2 / 2.0
        self.remove_self_interaction = remove_self_interaction
        self.norm_factor = norm_factor
        self.k_sq_max = (2.0 * torch.pi / dl) ** 2

    def _finish(self, pot, q, mu, batch, n_graphs, cell):
        pot = pot / torch.linalg.det(cell).abs()
        if self.remove_self_interaction:
            c = (2 * torch.pi) ** 1.5
            self_q = torch.zeros(n_graphs, device=q.device, dtype=q.dtype)
            self_q.index_add_(0, batch, q ** 2)
            self_mu = torch.zeros_like(self_q)
            self_mu.index_add_(0, batch, (mu ** 2).sum(dim=1))
            pot = pot - self_q / (self.sigma * c) - self_mu / (3.0 * self.sigma ** 3 * c)
        return pot * self.norm_factor

    def forward(self, q, mu, r, cell, batch: Optional[torch.Tensor] = None):
        q = q.reshape(r.shape[0])
        N = r.shape[0]
        device, dtype = r.device, r.dtype
        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=device)
        n_graphs = cell.shape[0]
        kgraph, kvec, k_sq, factors = _kgrid(cell, self.dl, self.k_sq_max, device, dtype)
        # per-graph scatter over k: evaluate each atom only against its own graph's k
        order = torch.argsort(kgraph)
        kgraph, kvec, k_sq, factors = kgraph[order], kvec[order], k_sq[order], factors[order]
        nk = torch.bincount(kgraph, minlength=n_graphs)
        koff = torch.cumsum(nk, 0) - nk
        M = kvec.shape[0]
        Sr = torch.zeros(M, device=device, dtype=dtype)
        Si = torch.zeros(M, device=device, dtype=dtype)
        for g in range(n_graphs):
            am = batch == g
            kk = kvec[koff[g]:koff[g] + nk[g]]
            ph = r[am] @ kk.T                                   # [n, m]
            w = mu[am] @ kk.T
            c, s = torch.cos(ph), torch.sin(ph)
            qa = q[am].unsqueeze(1)
            Sr[koff[g]:koff[g] + nk[g]] = (qa * c - w * s).sum(0)
            Si[koff[g]:koff[g] + nk[g]] = (qa * s + w * c).sum(0)
        kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq
        pot = torch.zeros(n_graphs, device=device, dtype=dtype)
        pot.index_add_(0, kgraph, factors * kfac * (Sr ** 2 + Si ** 2))
        return self._finish(pot, q, mu, batch, n_graphs, cell)


if _HAS_TRITON:
    BLK = 128

    @triton.jit
    def _fwd_qu(q, mu, r, k, kg, aoff, na, Sr, Si, M, B: tl.constexpr):
        m = tl.program_id(0)
        if m >= M:
            return
        g = tl.load(kg + m); a0 = tl.load(aoff + g); n = tl.load(na + g)
        kx = tl.load(k + m*3+0); ky = tl.load(k + m*3+1); kz = tl.load(k + m*3+2)
        z = tl.zeros((), dtype=Sr.dtype.element_ty); ar = z; ai = z
        for a in range(0, n, B):
            o = a + tl.arange(0, B); mk = o < n; idx = a0 + o
            rx = tl.load(r+idx*3+0, mask=mk, other=0.); ry = tl.load(r+idx*3+1, mask=mk, other=0.)
            rz = tl.load(r+idx*3+2, mask=mk, other=0.)
            ux = tl.load(mu+idx*3+0, mask=mk, other=0.); uy = tl.load(mu+idx*3+1, mask=mk, other=0.)
            uz = tl.load(mu+idx*3+2, mask=mk, other=0.)
            qa = tl.load(q+idx, mask=mk, other=0.)
            ph = kx*rx+ky*ry+kz*rz; c = tl.cos(ph); s = tl.sin(ph)
            w = kx*ux+ky*uy+kz*uz
            ar += tl.sum(tl.where(mk, qa*c - w*s, z)); ai += tl.sum(tl.where(mk, qa*s + w*c, z))
        tl.store(Sr+m, ar); tl.store(Si+m, ai)

    @triton.jit
    def _bwd_atom_qu(q, mu, r, k, ba, koff, nk, dSr, dSi, dq, dmu, dr, N, B: tl.constexpr):
        a = tl.program_id(0)
        if a >= N:
            return
        g = tl.load(ba + a); k0 = tl.load(koff + g); nn = tl.load(nk + g)
        rx = tl.load(r+a*3+0); ry = tl.load(r+a*3+1); rz = tl.load(r+a*3+2)
        ux = tl.load(mu+a*3+0); uy = tl.load(mu+a*3+1); uz = tl.load(mu+a*3+2)
        qa = tl.load(q+a)
        z = tl.zeros((), dtype=dr.dtype.element_ty)
        gq = z; mx = z; my = z; mz = z; dx = z; dy = z; dz = z
        for kk in range(0, nn, B):
            o = kk + tl.arange(0, B); mk = o < nn; ki = k0 + o
            kx = tl.load(k+ki*3+0, mask=mk, other=0.); ky = tl.load(k+ki*3+1, mask=mk, other=0.)
            kz = tl.load(k+ki*3+2, mask=mk, other=0.)
            ph = kx*rx+ky*ry+kz*rz; c = tl.cos(ph); s = tl.sin(ph)
            w = kx*ux+ky*uy+kz*uz
            dr_ = tl.load(dSr+ki, mask=mk, other=0.); di_ = tl.load(dSi+ki, mask=mk, other=0.)
            P = -s*dr_ + c*di_; Q = c*dr_ + s*di_
            coef = qa*P - w*Q
            gq += tl.sum(tl.where(mk, Q, z))
            mx += tl.sum(tl.where(mk, P*kx, z)); my += tl.sum(tl.where(mk, P*ky, z)); mz += tl.sum(tl.where(mk, P*kz, z))
            dx += tl.sum(tl.where(mk, coef*kx, z)); dy += tl.sum(tl.where(mk, coef*ky, z)); dz += tl.sum(tl.where(mk, coef*kz, z))
        tl.store(dq+a, gq)
        tl.store(dmu+a*3+0, mx); tl.store(dmu+a*3+1, my); tl.store(dmu+a*3+2, mz)
        tl.store(dr+a*3+0, dx); tl.store(dr+a*3+1, dy); tl.store(dr+a*3+2, dz)

    @triton.jit
    def _bwd_k_qu(q, mu, r, k, kg, aoff, na, dSr, dSi, dk, M, B: tl.constexpr):
        m = tl.program_id(0)
        if m >= M:
            return
        g = tl.load(kg + m); a0 = tl.load(aoff + g); n = tl.load(na + g)
        kx = tl.load(k+m*3+0); ky = tl.load(k+m*3+1); kz = tl.load(k+m*3+2)
        dr_ = tl.load(dSr+m); di_ = tl.load(dSi+m)
        z = tl.zeros((), dtype=dk.dtype.element_ty); dx = z; dy = z; dz = z
        for a in range(0, n, B):
            o = a + tl.arange(0, B); mk = o < n; idx = a0 + o
            rx = tl.load(r+idx*3+0, mask=mk, other=0.); ry = tl.load(r+idx*3+1, mask=mk, other=0.)
            rz = tl.load(r+idx*3+2, mask=mk, other=0.)
            ux = tl.load(mu+idx*3+0, mask=mk, other=0.); uy = tl.load(mu+idx*3+1, mask=mk, other=0.)
            uz = tl.load(mu+idx*3+2, mask=mk, other=0.)
            qa = tl.load(q+idx, mask=mk, other=0.)
            ph = kx*rx+ky*ry+kz*rz; c = tl.cos(ph); s = tl.sin(ph)
            w = kx*ux+ky*uy+kz*uz
            P = -s*dr_ + c*di_; Q = c*dr_ + s*di_
            coef = qa*P - w*Q
            dx += tl.sum(tl.where(mk, coef*rx + P*ux, z)); dy += tl.sum(tl.where(mk, coef*ry + P*uy, z))
            dz += tl.sum(tl.where(mk, coef*rz + P*uz, z))
        tl.store(dk+m*3+0, dx); tl.store(dk+m*3+1, dy); tl.store(dk+m*3+2, dz)

    @triton.jit
    def _ddw_dS_qu(q, mu, r, k, kg, aoff, na, gdq, gdmu, gdr, gdk, gSr, gSi, M, B: tl.constexpr):
        m = tl.program_id(0)
        if m >= M:
            return
        g = tl.load(kg + m); a0 = tl.load(aoff + g); n = tl.load(na + g)
        kx = tl.load(k+m*3+0); ky = tl.load(k+m*3+1); kz = tl.load(k+m*3+2)
        gkx = tl.load(gdk+m*3+0); gky = tl.load(gdk+m*3+1); gkz = tl.load(gdk+m*3+2)
        z = tl.zeros((), dtype=gSr.dtype.element_ty); ar = z; ai = z
        for a in range(0, n, B):
            o = a + tl.arange(0, B); mk = o < n; idx = a0 + o
            rx = tl.load(r+idx*3+0, mask=mk, other=0.); ry = tl.load(r+idx*3+1, mask=mk, other=0.)
            rz = tl.load(r+idx*3+2, mask=mk, other=0.)
            ux = tl.load(mu+idx*3+0, mask=mk, other=0.); uy = tl.load(mu+idx*3+1, mask=mk, other=0.)
            uz = tl.load(mu+idx*3+2, mask=mk, other=0.)
            grx = tl.load(gdr+idx*3+0, mask=mk, other=0.); gry = tl.load(gdr+idx*3+1, mask=mk, other=0.)
            grz = tl.load(gdr+idx*3+2, mask=mk, other=0.)
            gux = tl.load(gdmu+idx*3+0, mask=mk, other=0.); guy = tl.load(gdmu+idx*3+1, mask=mk, other=0.)
            guz = tl.load(gdmu+idx*3+2, mask=mk, other=0.)
            qa = tl.load(q+idx, mask=mk, other=0.); gq = tl.load(gdq+idx, mask=mk, other=0.)
            ph = kx*rx+ky*ry+kz*rz; c = tl.cos(ph); s = tl.sin(ph)
            w = kx*ux+ky*uy+kz*uz
            D = grx*kx+gry*ky+grz*kz + gkx*rx+gky*ry+gkz*rz
            E = gux*kx+guy*ky+guz*kz + gkx*ux+gky*uy+gkz*uz
            ar += tl.sum(tl.where(mk, gq*c - E*s - D*(qa*s + w*c), z))
            ai += tl.sum(tl.where(mk, gq*s + E*c + D*(qa*c - w*s), z))
        tl.store(gSr+m, ar); tl.store(gSi+m, ai)

    @triton.jit
    def _ddw_atom_qu(q, mu, r, k, ba, koff, nk, dSr, dSi, gdq, gdmu, gdr, gdk,
                     gq_out, gmu_out, gr_out, N, B: tl.constexpr):
        a = tl.program_id(0)
        if a >= N:
            return
        g = tl.load(ba + a); k0 = tl.load(koff + g); nn = tl.load(nk + g)
        rx = tl.load(r+a*3+0); ry = tl.load(r+a*3+1); rz = tl.load(r+a*3+2)
        ux = tl.load(mu+a*3+0); uy = tl.load(mu+a*3+1); uz = tl.load(mu+a*3+2)
        grx = tl.load(gdr+a*3+0); gry = tl.load(gdr+a*3+1); grz = tl.load(gdr+a*3+2)
        gux = tl.load(gdmu+a*3+0); guy = tl.load(gdmu+a*3+1); guz = tl.load(gdmu+a*3+2)
        qa = tl.load(q+a); gq = tl.load(gdq+a)
        z = tl.zeros((), dtype=gr_out.dtype.element_ty)
        oq = z; mx = z; my = z; mz = z; dx = z; dy = z; dz = z
        for kk in range(0, nn, B):
            o = kk + tl.arange(0, B); mk = o < nn; ki = k0 + o
            kx = tl.load(k+ki*3+0, mask=mk, other=0.); ky = tl.load(k+ki*3+1, mask=mk, other=0.)
            kz = tl.load(k+ki*3+2, mask=mk, other=0.)
            gkx = tl.load(gdk+ki*3+0, mask=mk, other=0.); gky = tl.load(gdk+ki*3+1, mask=mk, other=0.)
            gkz = tl.load(gdk+ki*3+2, mask=mk, other=0.)
            dr_ = tl.load(dSr+ki, mask=mk, other=0.); di_ = tl.load(dSi+ki, mask=mk, other=0.)
            ph = kx*rx+ky*ry+kz*rz; c = tl.cos(ph); s = tl.sin(ph)
            w = kx*ux+ky*uy+kz*uz
            P = -s*dr_ + c*di_; Q = c*dr_ + s*di_
            D = grx*kx+gry*ky+grz*kz + gkx*rx+gky*ry+gkz*rz
            E = gux*kx+guy*ky+guz*kz + gkx*ux+gky*uy+gkz*uz
            oq += tl.sum(tl.where(mk, P*D, z))
            mx += tl.sum(tl.where(mk, -Q*D*kx + P*gkx, z)); my += tl.sum(tl.where(mk, -Q*D*ky + P*gky, z))
            mz += tl.sum(tl.where(mk, -Q*D*kz + P*gkz, z))
            A = gq*P - E*Q - (qa*Q + w*P)*D; C = qa*P - w*Q
            dx += tl.sum(tl.where(mk, A*kx + C*gkx, z)); dy += tl.sum(tl.where(mk, A*ky + C*gky, z))
            dz += tl.sum(tl.where(mk, A*kz + C*gkz, z))
        tl.store(gq_out+a, oq)
        tl.store(gmu_out+a*3+0, mx); tl.store(gmu_out+a*3+1, my); tl.store(gmu_out+a*3+2, mz)
        tl.store(gr_out+a*3+0, dx); tl.store(gr_out+a*3+1, dy); tl.store(gr_out+a*3+2, dz)

    @triton.jit
    def _ddw_k_qu(q, mu, r, k, kg, aoff, na, dSr, dSi, gdq, gdmu, gdr, gdk, gk_out, M, B: tl.constexpr):
        m = tl.program_id(0)
        if m >= M:
            return
        g = tl.load(kg + m); a0 = tl.load(aoff + g); n = tl.load(na + g)
        kx = tl.load(k+m*3+0); ky = tl.load(k+m*3+1); kz = tl.load(k+m*3+2)
        gkx = tl.load(gdk+m*3+0); gky = tl.load(gdk+m*3+1); gkz = tl.load(gdk+m*3+2)
        dr_ = tl.load(dSr+m); di_ = tl.load(dSi+m)
        z = tl.zeros((), dtype=gk_out.dtype.element_ty); dx = z; dy = z; dz = z
        for a in range(0, n, B):
            o = a + tl.arange(0, B); mk = o < n; idx = a0 + o
            rx = tl.load(r+idx*3+0, mask=mk, other=0.); ry = tl.load(r+idx*3+1, mask=mk, other=0.)
            rz = tl.load(r+idx*3+2, mask=mk, other=0.)
            ux = tl.load(mu+idx*3+0, mask=mk, other=0.); uy = tl.load(mu+idx*3+1, mask=mk, other=0.)
            uz = tl.load(mu+idx*3+2, mask=mk, other=0.)
            grx = tl.load(gdr+idx*3+0, mask=mk, other=0.); gry = tl.load(gdr+idx*3+1, mask=mk, other=0.)
            grz = tl.load(gdr+idx*3+2, mask=mk, other=0.)
            gux = tl.load(gdmu+idx*3+0, mask=mk, other=0.); guy = tl.load(gdmu+idx*3+1, mask=mk, other=0.)
            guz = tl.load(gdmu+idx*3+2, mask=mk, other=0.)
            qa = tl.load(q+idx, mask=mk, other=0.); gq = tl.load(gdq+idx, mask=mk, other=0.)
            ph = kx*rx+ky*ry+kz*rz; c = tl.cos(ph); s = tl.sin(ph)
            w = kx*ux+ky*uy+kz*uz
            P = -s*dr_ + c*di_; Q = c*dr_ + s*di_
            D = grx*kx+gry*ky+grz*kz + gkx*rx+gky*ry+gkz*rz
            E = gux*kx+guy*ky+guz*kz + gkx*ux+gky*uy+gkz*uz
            A = gq*P - E*Q - (qa*Q + w*P)*D; C = qa*P - w*Q
            dx += tl.sum(tl.where(mk, A*rx + P*gux - Q*D*ux + C*grx, z))
            dy += tl.sum(tl.where(mk, A*ry + P*guy - Q*D*uy + C*gry, z))
            dz += tl.sum(tl.where(mk, A*rz + P*guz - Q*D*uz + C*grz, z))
        tl.store(gk_out+m*3+0, dx); tl.store(gk_out+m*3+1, dy); tl.store(gk_out+m*3+2, dz)

    class _SBwdQU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, dSr, dSi, q, mu, r, k, kg, aoff, na, ba, koff, nk):
            N = q.shape[0]; M = k.shape[0]
            dSr = dSr.contiguous(); dSi = dSi.contiguous()
            dq = torch.zeros_like(q); dmu = torch.zeros_like(mu); dr = torch.zeros_like(r); dk = torch.zeros_like(k)
            _bwd_atom_qu[(N,)](q, mu, r, k, ba, koff, nk, dSr, dSi, dq, dmu, dr, N, B=BLK)
            _bwd_k_qu[(M,)](q, mu, r, k, kg, aoff, na, dSr, dSi, dk, M, B=BLK)
            ctx.save_for_backward(dSr, dSi, q, mu, r, k, kg, aoff, na, ba, koff, nk)
            return dq, dmu, dr, dk

        @staticmethod
        def backward(ctx, gdq, gdmu, gdr, gdk):
            dSr, dSi, q, mu, r, k, kg, aoff, na, ba, koff, nk = ctx.saved_tensors
            N = q.shape[0]; M = k.shape[0]
            gdq = gdq.contiguous(); gdmu = gdmu.contiguous(); gdr = gdr.contiguous(); gdk = gdk.contiguous()
            gSr = torch.zeros_like(dSr); gSi = torch.zeros_like(dSi)
            gq = torch.zeros_like(q); gmu = torch.zeros_like(mu); gr = torch.zeros_like(r); gk = torch.zeros_like(k)
            _ddw_dS_qu[(M,)](q, mu, r, k, kg, aoff, na, gdq, gdmu, gdr, gdk, gSr, gSi, M, B=BLK)
            _ddw_atom_qu[(N,)](q, mu, r, k, ba, koff, nk, dSr, dSi, gdq, gdmu, gdr, gdk, gq, gmu, gr, N, B=BLK)
            _ddw_k_qu[(M,)](q, mu, r, k, kg, aoff, na, dSr, dSi, gdq, gdmu, gdr, gdk, gk, M, B=BLK)
            return gSr, gSi, gq, gmu, gr, gk, None, None, None, None, None, None

    class _SFwdQU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, mu, r, k, kg, aoff, na, ba, koff, nk):
            M = k.shape[0]
            Sr = torch.empty(M, device=q.device, dtype=q.dtype)
            Si = torch.empty(M, device=q.device, dtype=q.dtype)
            _fwd_qu[(M,)](q, mu, r, k, kg, aoff, na, Sr, Si, M, B=BLK)
            ctx.save_for_backward(q, mu, r, k, kg, aoff, na, ba, koff, nk)
            return Sr, Si

        @staticmethod
        def backward(ctx, dSr, dSi):
            q, mu, r, k, kg, aoff, na, ba, koff, nk = ctx.saved_tensors
            dq, dmu, dr, dk = _SBwdQU.apply(dSr, dSi, q, mu, r, k, kg, aoff, na, ba, koff, nk)
            return dq, dmu, dr, dk, None, None, None, None, None, None

    class TritonEwaldQU(BatchedEwaldQU):
        def forward(self, q, mu, r, cell, batch: Optional[torch.Tensor] = None):
            q = q.reshape(r.shape[0])
            N = r.shape[0]
            device, dtype = r.device, r.dtype
            if batch is None:
                batch = torch.zeros(N, dtype=torch.long, device=device)
            n_graphs = cell.shape[0]
            kgraph, kvec, k_sq, factors = _kgrid(cell, self.dl, self.k_sq_max, device, dtype)
            kg32 = kgraph.to(torch.int32).contiguous()
            kvec = kvec.contiguous(); k_sq = k_sq.contiguous(); factors = factors.contiguous()
            na = torch.bincount(batch, minlength=n_graphs)
            aoff = (torch.cumsum(na, 0) - na).to(torch.int32).contiguous()
            na32 = na.to(torch.int32).contiguous()
            nk = torch.bincount(kgraph, minlength=n_graphs)
            koff = (torch.cumsum(nk, 0) - nk).to(torch.int32).contiguous()
            nk32 = nk.to(torch.int32).contiguous()
            ba32 = batch.to(torch.int32).contiguous()
            Sr, Si = _SFwdQU.apply(q.contiguous(), mu.contiguous(), r.contiguous(), kvec, kg32, aoff, na32, ba32, koff, nk32)
            kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq
            pot = torch.zeros(n_graphs, device=device, dtype=dtype)
            pot.index_add_(0, kgraph, factors * kfac * (Sr ** 2 + Si ** 2))
            return self._finish(pot, q, mu, batch, n_graphs, cell)
