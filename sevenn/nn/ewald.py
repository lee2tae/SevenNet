"""
Reciprocal-space Ewald kernels for SevenNet LES.
"""
from typing import Optional

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # triton is optional
    _HAS_TRITON = False


if _HAS_TRITON:
    BLK = 128


    # ----------------------------- forward -----------------------------
    @triton.jit
    def _fwd(q, r, k, kg, aoff, na, Sr, Si, M, NQ: tl.constexpr, B: tl.constexpr):
        m = tl.program_id(0); ch = tl.program_id(1)
        if m >= M:
            return
        g = tl.load(kg + m); a0 = tl.load(aoff + g); n = tl.load(na + g)
        kx = tl.load(k + m*3+0); ky = tl.load(k + m*3+1); kz = tl.load(k + m*3+2)
        z = tl.zeros((), dtype=Sr.dtype.element_ty); ar = z; ai = z
        for a in range(0, n, B):
            o = a + tl.arange(0, B); mk = o < n; idx = a0 + o
            rx = tl.load(r+idx*3+0, mask=mk, other=0.); ry = tl.load(r+idx*3+1, mask=mk, other=0.)
            rz = tl.load(r+idx*3+2, mask=mk, other=0.)
            qa = tl.load(q+idx*NQ+ch, mask=mk, other=0.)
            ph = kx*rx+ky*ry+kz*rz
            ar += tl.sum(tl.where(mk, qa*tl.cos(ph), z)); ai += tl.sum(tl.where(mk, qa*tl.sin(ph), z))
        tl.store(Sr+m*NQ+ch, ar); tl.store(Si+m*NQ+ch, ai)


    # ----------------------- first backward (VJP) -----------------------
    @triton.jit
    def _bwd_q(q, r, k, ba, koff, nk, dSr, dSi, dq, N, NQ: tl.constexpr, B: tl.constexpr):
        a = tl.program_id(0); ch = tl.program_id(1)
        if a >= N:
            return
        g = tl.load(ba + a); k0 = tl.load(koff + g); nn = tl.load(nk + g)
        rx = tl.load(r+a*3+0); ry = tl.load(r+a*3+1); rz = tl.load(r+a*3+2)
        acc = tl.zeros((), dtype=dq.dtype.element_ty)
        for kk in range(0, nn, B):
            o = kk + tl.arange(0, B); mk = o < nn; ki = k0 + o
            kx = tl.load(k+ki*3+0, mask=mk, other=0.); ky = tl.load(k+ki*3+1, mask=mk, other=0.)
            kz = tl.load(k+ki*3+2, mask=mk, other=0.)
            ph = kx*rx+ky*ry+kz*rz
            dr_ = tl.load(dSr+ki*NQ+ch, mask=mk, other=0.); di_ = tl.load(dSi+ki*NQ+ch, mask=mk, other=0.)
            acc += tl.sum(tl.where(mk, dr_*tl.cos(ph)+di_*tl.sin(ph), acc*0))
        tl.store(dq+a*NQ+ch, acc)


    @triton.jit
    def _bwd_r(q, r, k, ba, koff, nk, dSr, dSi, dr, N, NQ: tl.constexpr, B: tl.constexpr):
        a = tl.program_id(0)
        if a >= N:
            return
        g = tl.load(ba + a); k0 = tl.load(koff + g); nn = tl.load(nk + g)
        rx = tl.load(r+a*3+0); ry = tl.load(r+a*3+1); rz = tl.load(r+a*3+2)
        z = tl.zeros((), dtype=dr.dtype.element_ty); dx = z; dy = z; dz = z
        for kk in range(0, nn, B):
            o = kk + tl.arange(0, B); mk = o < nn; ki = k0 + o
            kx = tl.load(k+ki*3+0, mask=mk, other=0.); ky = tl.load(k+ki*3+1, mask=mk, other=0.)
            kz = tl.load(k+ki*3+2, mask=mk, other=0.)
            ph = kx*rx+ky*ry+kz*rz; c = tl.cos(ph); s = tl.sin(ph)
            coef = tl.zeros_like(kx)
            for ch in range(NQ):
                qa = tl.load(q+a*NQ+ch)
                dr_ = tl.load(dSr+ki*NQ+ch, mask=mk, other=0.); di_ = tl.load(dSi+ki*NQ+ch, mask=mk, other=0.)
                coef += qa*(-s*dr_ + c*di_)
            dx += tl.sum(tl.where(mk, coef*kx, z)); dy += tl.sum(tl.where(mk, coef*ky, z))
            dz += tl.sum(tl.where(mk, coef*kz, z))
        tl.store(dr+a*3+0, dx); tl.store(dr+a*3+1, dy); tl.store(dr+a*3+2, dz)


    @triton.jit
    def _bwd_k(q, r, k, kg, aoff, na, dSr, dSi, dk, M, NQ: tl.constexpr, B: tl.constexpr):
        m = tl.program_id(0)
        if m >= M:
            return
        g = tl.load(kg + m); a0 = tl.load(aoff + g); n = tl.load(na + g)
        kx = tl.load(k+m*3+0); ky = tl.load(k+m*3+1); kz = tl.load(k+m*3+2)
        z = tl.zeros((), dtype=dk.dtype.element_ty); dx = z; dy = z; dz = z
        for a in range(0, n, B):
            o = a + tl.arange(0, B); mk = o < n; idx = a0 + o
            rx = tl.load(r+idx*3+0, mask=mk, other=0.); ry = tl.load(r+idx*3+1, mask=mk, other=0.)
            rz = tl.load(r+idx*3+2, mask=mk, other=0.)
            ph = kx*rx+ky*ry+kz*rz; c = tl.cos(ph); s = tl.sin(ph)
            coef = tl.zeros_like(rx)
            for ch in range(NQ):
                qa = tl.load(q+idx*NQ+ch, mask=mk, other=0.)
                dr_ = tl.load(dSr+m*NQ+ch); di_ = tl.load(dSi+m*NQ+ch)
                coef += qa*(-s*dr_ + c*di_)
            dx += tl.sum(tl.where(mk, coef*rx, z)); dy += tl.sum(tl.where(mk, coef*ry, z))
            dz += tl.sum(tl.where(mk, coef*rz, z))
        tl.store(dk+m*3+0, dx); tl.store(dk+m*3+1, dy); tl.store(dk+m*3+2, dz)


    # ----------------------- double backward -----------------------
    @triton.jit
    def _ddw_dS(q, r, k, kg, aoff, na, dSr, dSi, gdq, gdr, gdk, gSr, gSi,
                M, NQ: tl.constexpr, B: tl.constexpr):
        m = tl.program_id(0); ch = tl.program_id(1)
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
            grx = tl.load(gdr+idx*3+0, mask=mk, other=0.); gry = tl.load(gdr+idx*3+1, mask=mk, other=0.)
            grz = tl.load(gdr+idx*3+2, mask=mk, other=0.)
            ph = kx*rx+ky*ry+kz*rz; c = tl.cos(ph); s = tl.sin(ph)
            D = grx*kx+gry*ky+grz*kz + gkx*rx+gky*ry+gkz*rz
            qa = tl.load(q+idx*NQ+ch, mask=mk, other=0.); gq = tl.load(gdq+idx*NQ+ch, mask=mk, other=0.)
            ar += tl.sum(tl.where(mk, gq*c - qa*s*D, z))
            ai += tl.sum(tl.where(mk, gq*s + qa*c*D, z))
        tl.store(gSr+m*NQ+ch, ar); tl.store(gSi+m*NQ+ch, ai)


    @triton.jit
    def _ddw_q(q, r, k, ba, koff, nk, dSr, dSi, gdr, gdk, gq_out,
              N, NQ: tl.constexpr, B: tl.constexpr):
        a = tl.program_id(0); ch = tl.program_id(1)
        if a >= N:
            return
        g = tl.load(ba + a); k0 = tl.load(koff + g); nn = tl.load(nk + g)
        rx = tl.load(r+a*3+0); ry = tl.load(r+a*3+1); rz = tl.load(r+a*3+2)
        grx = tl.load(gdr+a*3+0); gry = tl.load(gdr+a*3+1); grz = tl.load(gdr+a*3+2)
        acc = tl.zeros((), dtype=gq_out.dtype.element_ty)
        for kk in range(0, nn, B):
            o = kk + tl.arange(0, B); mk = o < nn; ki = k0 + o
            kx = tl.load(k+ki*3+0, mask=mk, other=0.); ky = tl.load(k+ki*3+1, mask=mk, other=0.)
            kz = tl.load(k+ki*3+2, mask=mk, other=0.)
            gkx = tl.load(gdk+ki*3+0, mask=mk, other=0.); gky = tl.load(gdk+ki*3+1, mask=mk, other=0.)
            gkz = tl.load(gdk+ki*3+2, mask=mk, other=0.)
            ph = kx*rx+ky*ry+kz*rz; c = tl.cos(ph); s = tl.sin(ph)
            D = grx*kx+gry*ky+grz*kz + gkx*rx+gky*ry+gkz*rz
            dr_ = tl.load(dSr+ki*NQ+ch, mask=mk, other=0.); di_ = tl.load(dSi+ki*NQ+ch, mask=mk, other=0.)
            P = -s*dr_ + c*di_
            acc += tl.sum(tl.where(mk, P*D, acc*0))
        tl.store(gq_out+a*NQ+ch, acc)


    @triton.jit
    def _ddw_r(q, r, k, ba, koff, nk, dSr, dSi, gdq, gdr, gdk, gr_out,
              N, NQ: tl.constexpr, B: tl.constexpr):
        a = tl.program_id(0)
        if a >= N:
            return
        g = tl.load(ba + a); k0 = tl.load(koff + g); nn = tl.load(nk + g)
        rx = tl.load(r+a*3+0); ry = tl.load(r+a*3+1); rz = tl.load(r+a*3+2)
        grx = tl.load(gdr+a*3+0); gry = tl.load(gdr+a*3+1); grz = tl.load(gdr+a*3+2)
        z = tl.zeros((), dtype=gr_out.dtype.element_ty); dx = z; dy = z; dz = z
        for kk in range(0, nn, B):
            o = kk + tl.arange(0, B); mk = o < nn; ki = k0 + o
            kx = tl.load(k+ki*3+0, mask=mk, other=0.); ky = tl.load(k+ki*3+1, mask=mk, other=0.)
            kz = tl.load(k+ki*3+2, mask=mk, other=0.)
            gkx = tl.load(gdk+ki*3+0, mask=mk, other=0.); gky = tl.load(gdk+ki*3+1, mask=mk, other=0.)
            gkz = tl.load(gdk+ki*3+2, mask=mk, other=0.)
            ph = kx*rx+ky*ry+kz*rz; c = tl.cos(ph); s = tl.sin(ph)
            D = grx*kx+gry*ky+grz*kz + gkx*rx+gky*ry+gkz*rz
            A1 = tl.zeros_like(kx); A2 = tl.zeros_like(kx); A3 = tl.zeros_like(kx)
            for ch in range(NQ):
                dr_ = tl.load(dSr+ki*NQ+ch, mask=mk, other=0.); di_ = tl.load(dSi+ki*NQ+ch, mask=mk, other=0.)
                P = -s*dr_ + c*di_; Q = c*dr_ + s*di_
                qa = tl.load(q+a*NQ+ch); gq = tl.load(gdq+a*NQ+ch)
                A1 += P*gq; A2 += P*qa; A3 += qa*Q
            dx += tl.sum(tl.where(mk, A1*kx + A2*gkx - A3*D*kx, z))
            dy += tl.sum(tl.where(mk, A1*ky + A2*gky - A3*D*ky, z))
            dz += tl.sum(tl.where(mk, A1*kz + A2*gkz - A3*D*kz, z))
        tl.store(gr_out+a*3+0, dx); tl.store(gr_out+a*3+1, dy); tl.store(gr_out+a*3+2, dz)


    @triton.jit
    def _ddw_k(q, r, k, kg, aoff, na, dSr, dSi, gdq, gdr, gdk, gk_out,
              M, NQ: tl.constexpr, B: tl.constexpr):
        m = tl.program_id(0)
        if m >= M:
            return
        g = tl.load(kg + m); a0 = tl.load(aoff + g); n = tl.load(na + g)
        kx = tl.load(k+m*3+0); ky = tl.load(k+m*3+1); kz = tl.load(k+m*3+2)
        gkx = tl.load(gdk+m*3+0); gky = tl.load(gdk+m*3+1); gkz = tl.load(gdk+m*3+2)
        z = tl.zeros((), dtype=gk_out.dtype.element_ty); dx = z; dy = z; dz = z
        for a in range(0, n, B):
            o = a + tl.arange(0, B); mk = o < n; idx = a0 + o
            rx = tl.load(r+idx*3+0, mask=mk, other=0.); ry = tl.load(r+idx*3+1, mask=mk, other=0.)
            rz = tl.load(r+idx*3+2, mask=mk, other=0.)
            grx = tl.load(gdr+idx*3+0, mask=mk, other=0.); gry = tl.load(gdr+idx*3+1, mask=mk, other=0.)
            grz = tl.load(gdr+idx*3+2, mask=mk, other=0.)
            ph = kx*rx+ky*ry+kz*rz; c = tl.cos(ph); s = tl.sin(ph)
            D = grx*kx+gry*ky+grz*kz + gkx*rx+gky*ry+gkz*rz
            A1 = tl.zeros_like(rx); A2 = tl.zeros_like(rx); A3 = tl.zeros_like(rx)
            for ch in range(NQ):
                dr_ = tl.load(dSr+m*NQ+ch); di_ = tl.load(dSi+m*NQ+ch)
                P = -s*dr_ + c*di_; Q = c*dr_ + s*di_
                qa = tl.load(q+idx*NQ+ch, mask=mk, other=0.); gq = tl.load(gdq+idx*NQ+ch, mask=mk, other=0.)
                A1 += P*gq; A2 += P*qa; A3 += qa*Q
            dx += tl.sum(tl.where(mk, A1*rx + A2*grx - A3*D*rx, z))
            dy += tl.sum(tl.where(mk, A1*ry + A2*gry - A3*D*ry, z))
            dz += tl.sum(tl.where(mk, A1*rz + A2*grz - A3*D*rz, z))
        tl.store(gk_out+m*3+0, dx); tl.store(gk_out+m*3+1, dy); tl.store(gk_out+m*3+2, dz)


    # ----------------------- autograd wiring -----------------------
    class _SBwd(torch.autograd.Function):
        @staticmethod
        def forward(ctx, dSr, dSi, q, r, k, kg, aoff, na, ba, koff, nk):
            N, NQ = q.shape; M = k.shape[0]
            dSr = dSr.contiguous(); dSi = dSi.contiguous()
            dq = torch.zeros_like(q); dr = torch.zeros_like(r); dk = torch.zeros_like(k)
            _bwd_q[(N, NQ)](q, r, k, ba, koff, nk, dSr, dSi, dq, N, NQ=NQ, B=BLK)
            _bwd_r[(N,)](q, r, k, ba, koff, nk, dSr, dSi, dr, N, NQ=NQ, B=BLK)
            _bwd_k[(M,)](q, r, k, kg, aoff, na, dSr, dSi, dk, M, NQ=NQ, B=BLK)
            ctx.save_for_backward(dSr, dSi, q, r, k, kg, aoff, na, ba, koff, nk)
            ctx.NQ = NQ
            return dq, dr, dk

        @staticmethod
        def backward(ctx, gdq, gdr, gdk):
            dSr, dSi, q, r, k, kg, aoff, na, ba, koff, nk = ctx.saved_tensors
            NQ = ctx.NQ; N = q.shape[0]; M = k.shape[0]
            gdq = gdq.contiguous(); gdr = gdr.contiguous(); gdk = gdk.contiguous()
            gSr = torch.zeros_like(dSr); gSi = torch.zeros_like(dSi)
            gq = torch.zeros_like(q); gr = torch.zeros_like(r); gk = torch.zeros_like(k)
            _ddw_dS[(M, NQ)](q, r, k, kg, aoff, na, dSr, dSi, gdq, gdr, gdk, gSr, gSi, M, NQ=NQ, B=BLK)
            _ddw_q[(N, NQ)](q, r, k, ba, koff, nk, dSr, dSi, gdr, gdk, gq, N, NQ=NQ, B=BLK)
            _ddw_r[(N,)](q, r, k, ba, koff, nk, dSr, dSi, gdq, gdr, gdk, gr, N, NQ=NQ, B=BLK)
            _ddw_k[(M,)](q, r, k, kg, aoff, na, dSr, dSi, gdq, gdr, gdk, gk, M, NQ=NQ, B=BLK)
            return gSr, gSi, gq, gr, gk, None, None, None, None, None, None


    class _SFwd(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, r, k, kg, aoff, na, ba, koff, nk):
            N, NQ = q.shape; M = k.shape[0]
            Sr = torch.empty(M, NQ, device=q.device, dtype=q.dtype)
            Si = torch.empty(M, NQ, device=q.device, dtype=q.dtype)
            _fwd[(M, NQ)](q, r, k, kg, aoff, na, Sr, Si, M, NQ=NQ, B=BLK)
            ctx.save_for_backward(q, r, k, kg, aoff, na, ba, koff, nk)
            return Sr, Si

        @staticmethod
        def backward(ctx, dSr, dSi):
            q, r, k, kg, aoff, na, ba, koff, nk = ctx.saved_tensors
            dq, dr, dk = _SBwd.apply(dSr, dSi, q, r, k, kg, aoff, na, ba, koff, nk)
            return dq, dr, dk, None, None, None, None, None, None


    class TritonEwald(torch.nn.Module):
        def __init__(self, dl=2.0, sigma=1.0, remove_self_interaction=True, norm_factor=90.4756):
            super().__init__()
            self.dl = dl; self.sigma = sigma; self.sigma_sq_half = sigma**2/2.0
            self.remove_self_interaction = remove_self_interaction
            self.norm_factor = norm_factor; self.twopi = 2.0*torch.pi
            self.k_sq_max = (self.twopi/self.dl)**2

        def forward(self, q, r, cell, batch=None):
            if q.dim() == 1:
                q = q.unsqueeze(1)
            N = r.shape[0]; device, dtype = r.device, r.dtype
            if batch is None:
                batch = torch.zeros(N, dtype=torch.long, device=device)
            n_graphs = cell.shape[0]; n_q = q.shape[1]
            cell_inv = torch.linalg.inv(cell)
            Grec = self.twopi*cell_inv.transpose(1, 2)
            Nk = torch.clamp((torch.norm(cell, dim=2)/self.dl).to(torch.int64), min=1)
            dims = 2*Nk+1; Pbox = dims.prod(1)
            kgraph = torch.repeat_interleave(torch.arange(n_graphs, device=device), Pbox)
            box_off = torch.cumsum(Pbox, 0)-Pbox
            bl = torch.arange(kgraph.shape[0], device=device)-box_off[kgraph]
            dg = dims[kgraph]
            i0 = bl//(dg[:, 1]*dg[:, 2]); rem = bl % (dg[:, 1]*dg[:, 2])
            i1 = rem//dg[:, 2]; i2 = rem % dg[:, 2]
            nvec = (torch.stack([i0, i1, i2], -1)-Nk[kgraph]).to(dtype)
            kvec = torch.einsum('md,mde->me', nvec, Grec[kgraph])
            k_sq = (kvec**2).sum(-1)
            sph = (k_sq > 0) & (k_sq <= self.k_sq_max)
            nz = (nvec != 0).to(torch.int64); fnz = torch.argmax(nz, dim=1)
            sign = torch.gather(nvec, 1, fnz.unsqueeze(1)).squeeze(1)
            allz = (nvec == 0).all(dim=1)
            hemi = (sign > 0) | allz
            factors = torch.where(allz, torch.ones_like(k_sq), torch.full_like(k_sq, 2.0))
            keep = sph & hemi
            kgraph = kgraph[keep].to(torch.int32).contiguous()
            kvec = kvec[keep].contiguous(); k_sq = k_sq[keep].contiguous()
            factors = factors[keep].contiguous(); M = kvec.shape[0]
            na = torch.bincount(batch, minlength=n_graphs)
            atom_off = (torch.cumsum(na, 0)-na).to(torch.int32).contiguous()
            na32 = na.to(torch.int32).contiguous()
            nk = torch.bincount(kgraph.to(torch.long), minlength=n_graphs)
            koff = (torch.cumsum(nk, 0)-nk).to(torch.int32).contiguous()
            nk32 = nk.to(torch.int32).contiguous(); batch32 = batch.to(torch.int32).contiguous()
            q = q.contiguous(); r = r.contiguous()
            Sr, Si = _SFwd.apply(q, r, kvec, kgraph, atom_off, na32, batch32, koff, nk32)
            S_sq = Sr**2 + Si**2
            kfac = torch.exp(-self.sigma_sq_half*k_sq)/k_sq
            w = (factors*kfac).unsqueeze(-1)
            pot = torch.zeros(n_graphs, n_q, device=device, dtype=dtype)
            pot.index_add_(0, kgraph.to(torch.long), w*S_sq)
            volume = torch.linalg.det(cell).abs()
            pot = pot/volume.unsqueeze(1)
            if self.remove_self_interaction:
                q_sq = torch.zeros(n_graphs, n_q, device=device, dtype=dtype)
                q_sq.index_add_(0, batch, q**2)
                pot = pot - q_sq/(self.sigma*(2*torch.pi)**1.5)
            return (pot*self.norm_factor).sum(dim=1)


    # ------------------- direct sum (open boundary) -------------------
    # primitive: phi_i^c = sum_{j != i, graph(i)} q_j^c f(r_ij)
    # f(r) = erf(a r)/r,  g = f'/r,  h = g'/r,  a = 1/(sqrt(2) sigma)
    @triton.jit
    def _ds_fwd(q, r, ba, aoff, na, phi, N, A: tl.constexpr, C0: tl.constexpr, NQ: tl.constexpr, B: tl.constexpr):
        i = tl.program_id(0); ch = tl.program_id(1)
        if i >= N:
            return
        g = tl.load(ba + i); a0 = tl.load(aoff + g); n = tl.load(na + g)
        rx = tl.load(r+i*3+0); ry = tl.load(r+i*3+1); rz = tl.load(r+i*3+2)
        acc = tl.zeros((), dtype=phi.dtype.element_ty)
        for jj in range(0, n, B):
            o = jj + tl.arange(0, B); mk = o < n; idx = a0 + o
            mk = mk & (idx != i)
            jx = tl.load(r+idx*3+0, mask=mk, other=0.); jy = tl.load(r+idx*3+1, mask=mk, other=0.)
            jz = tl.load(r+idx*3+2, mask=mk, other=0.)
            dx = rx-jx; dy = ry-jy; dz = rz-jz
            rr = tl.maximum(tl.sqrt(dx*dx+dy*dy+dz*dz), 1e-10)
            f = tl.erf(A*rr)/rr
            qj = tl.load(q+idx*NQ+ch, mask=mk, other=0.)
            acc += tl.sum(tl.where(mk, qj*f, acc*0))
        tl.store(phi+i*NQ+ch, acc)


    @triton.jit
    def _ds_bwd_r(q, r, ba, aoff, na, dphi, dr_out, N, A: tl.constexpr, C0: tl.constexpr,
                  NQ: tl.constexpr, B: tl.constexpr):
        # dr_i = sum_{j != i} sum_c (dphi_i^c q_j^c + dphi_j^c q_i^c) g(r_ij) d_ij
        i = tl.program_id(0)
        if i >= N:
            return
        g = tl.load(ba + i); a0 = tl.load(aoff + g); n = tl.load(na + g)
        rx = tl.load(r+i*3+0); ry = tl.load(r+i*3+1); rz = tl.load(r+i*3+2)
        z = tl.zeros((), dtype=dr_out.dtype.element_ty); ox = z; oy = z; oz = z
        for jj in range(0, n, B):
            o = jj + tl.arange(0, B); mk = o < n; idx = a0 + o
            mk = mk & (idx != i)
            jx = tl.load(r+idx*3+0, mask=mk, other=0.); jy = tl.load(r+idx*3+1, mask=mk, other=0.)
            jz = tl.load(r+idx*3+2, mask=mk, other=0.)
            dx = rx-jx; dy = ry-jy; dz = rz-jz
            rr = tl.maximum(tl.sqrt(dx*dx+dy*dy+dz*dz), 1e-10)
            gg = C0*tl.exp(-A*A*rr*rr)/(rr*rr) - tl.erf(A*rr)/(rr*rr*rr)
            w = tl.zeros_like(dx)
            for ch in range(NQ):
                di = tl.load(dphi+i*NQ+ch); qi = tl.load(q+i*NQ+ch)
                dj = tl.load(dphi+idx*NQ+ch, mask=mk, other=0.)
                qj = tl.load(q+idx*NQ+ch, mask=mk, other=0.)
                w += di*qj + dj*qi
            ox += tl.sum(tl.where(mk, w*gg*dx, z)); oy += tl.sum(tl.where(mk, w*gg*dy, z))
            oz += tl.sum(tl.where(mk, w*gg*dz, z))
        tl.store(dr_out+i*3+0, ox); tl.store(dr_out+i*3+1, oy); tl.store(dr_out+i*3+2, oz)


    @triton.jit
    def _ds_ddw_phi(q, r, ba, aoff, na, gdq, gdr, out, N, A: tl.constexpr, C0: tl.constexpr,
                    NQ: tl.constexpr, B: tl.constexpr):
        # gdphi_i^c = sum_{j != i} [gdq_j^c f_ij + q_j^c g_ij (gdr_i - gdr_j).d_ij]
        i = tl.program_id(0); ch = tl.program_id(1)
        if i >= N:
            return
        g = tl.load(ba + i); a0 = tl.load(aoff + g); n = tl.load(na + g)
        rx = tl.load(r+i*3+0); ry = tl.load(r+i*3+1); rz = tl.load(r+i*3+2)
        gix = tl.load(gdr+i*3+0); giy = tl.load(gdr+i*3+1); giz = tl.load(gdr+i*3+2)
        acc = tl.zeros((), dtype=out.dtype.element_ty)
        for jj in range(0, n, B):
            o = jj + tl.arange(0, B); mk = o < n; idx = a0 + o
            mk = mk & (idx != i)
            jx = tl.load(r+idx*3+0, mask=mk, other=0.); jy = tl.load(r+idx*3+1, mask=mk, other=0.)
            jz = tl.load(r+idx*3+2, mask=mk, other=0.)
            gjx = tl.load(gdr+idx*3+0, mask=mk, other=0.); gjy = tl.load(gdr+idx*3+1, mask=mk, other=0.)
            gjz = tl.load(gdr+idx*3+2, mask=mk, other=0.)
            dx = rx-jx; dy = ry-jy; dz = rz-jz
            rr = tl.maximum(tl.sqrt(dx*dx+dy*dy+dz*dz), 1e-10)
            f = tl.erf(A*rr)/rr
            gg = C0*tl.exp(-A*A*rr*rr)/(rr*rr) - tl.erf(A*rr)/(rr*rr*rr)
            D = (gix-gjx)*dx + (giy-gjy)*dy + (giz-gjz)*dz
            gq_j = tl.load(gdq+idx*NQ+ch, mask=mk, other=0.)
            qj = tl.load(q+idx*NQ+ch, mask=mk, other=0.)
            acc += tl.sum(tl.where(mk, gq_j*f + qj*gg*D, acc*0))
        tl.store(out+i*NQ+ch, acc)


    @triton.jit
    def _ds_ddw_q(r, dphi, ba, aoff, na, gdr, out, N, A: tl.constexpr, C0: tl.constexpr,
                  NQ: tl.constexpr, B: tl.constexpr):
        # gq_i^c = sum_{j != i} dphi_j^c g_ij (gdr_i - gdr_j).d_ij
        i = tl.program_id(0); ch = tl.program_id(1)
        if i >= N:
            return
        g = tl.load(ba + i); a0 = tl.load(aoff + g); n = tl.load(na + g)
        rx = tl.load(r+i*3+0); ry = tl.load(r+i*3+1); rz = tl.load(r+i*3+2)
        gix = tl.load(gdr+i*3+0); giy = tl.load(gdr+i*3+1); giz = tl.load(gdr+i*3+2)
        acc = tl.zeros((), dtype=out.dtype.element_ty)
        for jj in range(0, n, B):
            o = jj + tl.arange(0, B); mk = o < n; idx = a0 + o
            mk = mk & (idx != i)
            jx = tl.load(r+idx*3+0, mask=mk, other=0.); jy = tl.load(r+idx*3+1, mask=mk, other=0.)
            jz = tl.load(r+idx*3+2, mask=mk, other=0.)
            gjx = tl.load(gdr+idx*3+0, mask=mk, other=0.); gjy = tl.load(gdr+idx*3+1, mask=mk, other=0.)
            gjz = tl.load(gdr+idx*3+2, mask=mk, other=0.)
            dx = rx-jx; dy = ry-jy; dz = rz-jz
            rr = tl.maximum(tl.sqrt(dx*dx+dy*dy+dz*dz), 1e-10)
            gg = C0*tl.exp(-A*A*rr*rr)/(rr*rr) - tl.erf(A*rr)/(rr*rr*rr)
            D = (gix-gjx)*dx + (giy-gjy)*dy + (giz-gjz)*dz
            dj = tl.load(dphi+idx*NQ+ch, mask=mk, other=0.)
            acc += tl.sum(tl.where(mk, dj*gg*D, acc*0))
        tl.store(out+i*NQ+ch, acc)


    @triton.jit
    def _ds_ddw_r(q, r, dphi, ba, aoff, na, gdq, gdr, out, N, A: tl.constexpr, C0: tl.constexpr,
                  NQ: tl.constexpr, B: tl.constexpr):
        # gr_i = sum_{j != i} { a_ij g_ij d_ij
        #        + w_ij [(gdr_i - gdr_j) g_ij + ((gdr_i - gdr_j).d_ij) h_ij d_ij] }
        # a_ij = sum_c (dphi_i^c gdq_j^c + dphi_j^c gdq_i^c)
        # w_ij = sum_c (dphi_i^c q_j^c + dphi_j^c q_i^c)
        i = tl.program_id(0)
        if i >= N:
            return
        g = tl.load(ba + i); a0 = tl.load(aoff + g); n = tl.load(na + g)
        rx = tl.load(r+i*3+0); ry = tl.load(r+i*3+1); rz = tl.load(r+i*3+2)
        gix = tl.load(gdr+i*3+0); giy = tl.load(gdr+i*3+1); giz = tl.load(gdr+i*3+2)
        z = tl.zeros((), dtype=out.dtype.element_ty); ox = z; oy = z; oz = z
        for jj in range(0, n, B):
            o = jj + tl.arange(0, B); mk = o < n; idx = a0 + o
            mk = mk & (idx != i)
            jx = tl.load(r+idx*3+0, mask=mk, other=0.); jy = tl.load(r+idx*3+1, mask=mk, other=0.)
            jz = tl.load(r+idx*3+2, mask=mk, other=0.)
            gjx = tl.load(gdr+idx*3+0, mask=mk, other=0.); gjy = tl.load(gdr+idx*3+1, mask=mk, other=0.)
            gjz = tl.load(gdr+idx*3+2, mask=mk, other=0.)
            dx = rx-jx; dy = ry-jy; dz = rz-jz
            rr = tl.maximum(tl.sqrt(dx*dx+dy*dy+dz*dz), 1e-10)
            r2 = rr*rr
            e = C0*tl.exp(-A*A*r2)
            erf_ = tl.erf(A*rr)
            gg = e/r2 - erf_/(r2*rr)
            hh = -e*(2.0*A*A*r2 + 3.0)/(r2*r2) + 3.0*erf_/(r2*r2*rr)
            aa = tl.zeros_like(dx); w = tl.zeros_like(dx)
            for ch in range(NQ):
                di = tl.load(dphi+i*NQ+ch); qi = tl.load(q+i*NQ+ch)
                gqi = tl.load(gdq+i*NQ+ch)
                dj = tl.load(dphi+idx*NQ+ch, mask=mk, other=0.)
                qj = tl.load(q+idx*NQ+ch, mask=mk, other=0.)
                gqj = tl.load(gdq+idx*NQ+ch, mask=mk, other=0.)
                aa += di*gqj + dj*gqi
                w += di*qj + dj*qi
            ux = gix-gjx; uy = giy-gjy; uz = giz-gjz
            D = ux*dx + uy*dy + uz*dz
            cd = aa*gg + w*hh*D
            ox += tl.sum(tl.where(mk, cd*dx + w*gg*ux, z))
            oy += tl.sum(tl.where(mk, cd*dy + w*gg*uy, z))
            oz += tl.sum(tl.where(mk, cd*dz + w*gg*uz, z))
        tl.store(out+i*3+0, ox); tl.store(out+i*3+1, oy); tl.store(out+i*3+2, oz)


    class _DSumBwd(torch.autograd.Function):
        @staticmethod
        def forward(ctx, dphi, q, r, ba, aoff, na, A, C0):
            N, NQ = q.shape
            dphi = dphi.contiguous()
            dq = torch.zeros_like(q); dr = torch.zeros_like(r)
            # f is symmetric: dq_j = sum_i dphi_i f_ij == fwd kernel on dphi
            _ds_fwd[(N, NQ)](dphi, r, ba, aoff, na, dq, N, A, C0, NQ=NQ, B=BLK)
            _ds_bwd_r[(N,)](q, r, ba, aoff, na, dphi, dr, N, A, C0, NQ=NQ, B=BLK)
            ctx.save_for_backward(dphi, q, r, ba, aoff, na)
            ctx.A = A; ctx.C0 = C0; ctx.NQ = NQ
            return dq, dr

        @staticmethod
        def backward(ctx, gdq, gdr):
            dphi, q, r, ba, aoff, na = ctx.saved_tensors
            A = ctx.A; C0 = ctx.C0; NQ = ctx.NQ; N = q.shape[0]
            gdq = gdq.contiguous(); gdr = gdr.contiguous()
            gphi = torch.zeros_like(dphi)
            gq = torch.zeros_like(q); gr = torch.zeros_like(r)
            _ds_ddw_phi[(N, NQ)](q, r, ba, aoff, na, gdq, gdr, gphi, N, A, C0, NQ=NQ, B=BLK)
            _ds_ddw_q[(N, NQ)](r, dphi, ba, aoff, na, gdr, gq, N, A, C0, NQ=NQ, B=BLK)
            _ds_ddw_r[(N,)](q, r, dphi, ba, aoff, na, gdq, gdr, gr, N, A, C0, NQ=NQ, B=BLK)
            return gphi, gq, gr, None, None, None, None, None


    class _DSumFwd(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, r, ba, aoff, na, A, C0):
            N, NQ = q.shape
            phi = torch.zeros_like(q)
            _ds_fwd[(N, NQ)](q, r, ba, aoff, na, phi, N, A, C0, NQ=NQ, B=BLK)
            ctx.save_for_backward(q, r, ba, aoff, na)
            ctx.A = A; ctx.C0 = C0
            return phi

        @staticmethod
        def backward(ctx, dphi):
            q, r, ba, aoff, na = ctx.saved_tensors
            dq, dr = _DSumBwd.apply(dphi, q, r, ba, aoff, na, ctx.A, ctx.C0)
            return dq, dr, None, None, None, None, None


    class TritonDirectSum(torch.nn.Module):
        """Triton counterpart of DirectSum; same convention and signature."""

        def __init__(self, sigma=1.0, remove_self_interaction=True, norm_factor=90.4756):
            super().__init__()
            self.sigma = sigma
            self.remove_self_interaction = remove_self_interaction
            self.norm_factor = norm_factor
            self.twopi = 2.0*torch.pi
            self.alpha = 1.0/(sigma*2.0**0.5)
            self.c0 = 2.0*self.alpha/torch.pi**0.5

        def forward(self, q, r, batch=None, n_graphs=None):
            if q.dim() == 1:
                q = q.unsqueeze(1)
            N = r.shape[0]
            device, dtype = r.device, r.dtype
            if batch is None:
                batch = torch.zeros(N, dtype=torch.long, device=device)
            if n_graphs is None:
                n_graphs = int(batch.max().item()) + 1 if N > 0 else 0
            n_q = q.shape[1]
            if N == 0:
                return torch.zeros(n_graphs, device=device, dtype=dtype)
            na = torch.bincount(batch, minlength=n_graphs)
            aoff = (torch.cumsum(na, 0)-na).to(torch.int32).contiguous()
            na32 = na.to(torch.int32).contiguous()
            ba = batch.to(torch.int32).contiguous()
            phi = _DSumFwd.apply(
                q.contiguous(), r.contiguous(), ba, aoff, na32, self.alpha, self.c0
            )
            pot = torch.zeros(n_graphs, n_q, device=device, dtype=dtype)
            pot.index_add_(0, batch, q*phi)
            pot = pot/self.twopi/2.0
            if not self.remove_self_interaction:
                q_sq = torch.zeros(n_graphs, n_q, device=device, dtype=dtype)
                q_sq.index_add_(0, batch, q**2)
                pot = pot + q_sq/(self.sigma*self.twopi**1.5)
            return (pot*self.norm_factor).sum(dim=1)


class BatchedEwald(nn.Module):
    """forward(q, r, cell, batch) -> [n_graphs]"""

    def __init__(
        self,
        dl: float = 2.0,
        sigma: float = 1.0,
        remove_self_interaction: bool = True,
        norm_factor: float = 90.4756,
    ):
        super().__init__()
        self.dl = dl
        self.sigma = sigma
        self.sigma_sq_half = sigma ** 2 / 2.0
        self.remove_self_interaction = remove_self_interaction
        self.norm_factor = norm_factor
        self.twopi = 2.0 * torch.pi
        self.k_sq_max = (self.twopi / self.dl) ** 2

    def forward(
        self,
        q: torch.Tensor,                       # [N, n_q] or [N]
        r: torch.Tensor,                       # [N, 3]
        cell: torch.Tensor,                    # [n_graphs, 3, 3]
        batch: Optional[torch.Tensor] = None,  # [N]
    ) -> torch.Tensor:                         # [n_graphs]
        if q.dim() == 1:
            q = q.unsqueeze(1)
        N = r.shape[0]
        device, dtype = r.device, r.dtype
        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=device)
        n_graphs = cell.shape[0]
        n_q = q.shape[1]

        # --- reciprocal lattice vectors (kept differentiable in `cell`) ---
        cell_inv = torch.linalg.inv(cell)                 # [G,3,3]
        G = self.twopi * cell_inv.transpose(1, 2)         # [G,3,3]  G = 2pi (M^-1)^T
        norms = torch.norm(cell, dim=2)                   # [G,3]  |a_i| per structure
        Nk = torch.clamp((norms / self.dl).to(torch.int64), min=1)   # [G,3]
        Nk_max = Nk.max(dim=0).values                     # [3]  shared grid envelope

        a0 = torch.arange(-int(Nk_max[0]), int(Nk_max[0]) + 1, device=device)
        a1 = torch.arange(-int(Nk_max[1]), int(Nk_max[1]) + 1, device=device)
        a2 = torch.arange(-int(Nk_max[2]), int(Nk_max[2]) + 1, device=device)
        nvec = torch.stack(
            torch.meshgrid([a0, a1, a2], indexing='ij'), dim=-1
        ).reshape(-1, 3).to(dtype)                         # [P,3]
        P = nvec.shape[0]

        # --- per-structure k vectors: kvec[g,p] = nvec[p] @ G[g] ---
        kvec = torch.einsum('pd,gde->gpe', nvec, G)       # [G,P,3]
        k_sq = (kvec ** 2).sum(dim=-1)                    # [G,P]

        # --- validity: own integer box AND spherical cutoff AND hemisphere ---
        # box_mask restricts the shared grid back to each structure's own
        # [-Nk,Nk] box, so the surviving set matches the reference exactly.
        box_mask = (nvec.abs().unsqueeze(0) <= Nk.unsqueeze(1)).all(dim=2)  # [G,P]
        spherical = (k_sq > 0) & (k_sq <= self.k_sq_max)                    # [G,P]

        # hemisphere selection is a function of nvec only -> shared across graphs
        non_zero = (nvec != 0).to(torch.int64)
        first_nz = torch.argmax(non_zero, dim=1)                           # [P]
        sign = torch.gather(nvec, 1, first_nz.unsqueeze(1)).squeeze(1)     # [P]
        all_zero = (nvec == 0).all(dim=1)                                  # [P]
        hemisphere = (sign > 0) | all_zero                                 # [P]
        factors = torch.where(
            all_zero,
            torch.ones(P, device=device, dtype=dtype),
            torch.full((P,), 2.0, device=device, dtype=dtype),
        )                                                                  # [P]

        valid = box_mask & spherical & hemisphere.unsqueeze(0)             # [G,P]

        # drop k-points unused by every graph
        keep = valid.any(dim=0)                            # [P]
        kvec = kvec[:, keep]                               # [G,P',3]
        k_sq = k_sq[:, keep]                               # [G,P']
        valid = valid[:, keep]                             # [G,P']
        factors = factors[keep]                            # [P']
        P = kvec.shape[1]

        validf = valid.to(dtype)

        # --- structure factor S(k) = sum_atoms q * e^{i k.r}, scattered by graph ---
        kvec_atom = kvec[batch]                            # [N,P',3]
        k_dot_r = torch.einsum('nd,npd->np', r, kvec_atom)  # [N,P']
        cos = torch.cos(k_dot_r)                           # [N,P']
        sin = torch.sin(k_dot_r)                           # [N,P']

        qc = q.unsqueeze(2) * cos.unsqueeze(1)             # [N,n_q,P']
        qs = q.unsqueeze(2) * sin.unsqueeze(1)             # [N,n_q,P']
        S_real = torch.zeros(n_graphs, n_q, P, device=device, dtype=dtype)
        S_imag = torch.zeros(n_graphs, n_q, P, device=device, dtype=dtype)
        S_real.index_add_(0, batch, qc)                    # scatter over atoms
        S_imag.index_add_(0, batch, qs)
        S_sq = S_real ** 2 + S_imag ** 2                   # [G,n_q,P']

        # --- assemble potential: (1/V) sum_k factors * exp(-s^2 k^2/2)/k^2 * |S|^2 ---
        k_sq_safe = torch.where(valid, k_sq, torch.ones_like(k_sq))   # avoid /0
        kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq_safe      # [G,P']
        weight = factors.unsqueeze(0) * kfac * validf                # [G,P']
        volume = torch.linalg.det(cell).abs()                        # [G]
        pot = (weight.unsqueeze(1) * S_sq).sum(dim=2) / volume.unsqueeze(1)  # [G,n_q]

        # --- self-interaction removal (per charge channel) ---
        if self.remove_self_interaction:
            q_sq_tot = torch.zeros(n_graphs, n_q, device=device, dtype=dtype)
            q_sq_tot.index_add_(0, batch, q ** 2)                     # [G,n_q]
            pot = pot - q_sq_tot / (self.sigma * (2 * torch.pi) ** 1.5)

        pot = pot * self.norm_factor                                 # [G,n_q]
        return pot.sum(dim=1)                                        # [G]


class FlatBatchedEwald(nn.Module):
    """
    k-points are flattened for batch computation.
    """

    def __init__(
        self,
        dl: float = 2.0,
        sigma: float = 1.0,
        remove_self_interaction: bool = True,
        norm_factor: float = 90.4756,
    ):
        super().__init__()
        self.dl = dl
        self.sigma = sigma
        self.sigma_sq_half = sigma ** 2 / 2.0
        self.remove_self_interaction = remove_self_interaction
        self.norm_factor = norm_factor
        self.twopi = 2.0 * torch.pi
        self.k_sq_max = (self.twopi / self.dl) ** 2

    def forward(
        self,
        q: torch.Tensor,                       # [N, n_q] or [N]
        r: torch.Tensor,                       # [N, 3]
        cell: torch.Tensor,                    # [n_graphs, 3, 3]
        batch: Optional[torch.Tensor] = None,  # [N]
    ) -> torch.Tensor:                         # [n_graphs]
        if q.dim() == 1:
            q = q.unsqueeze(1)
        N = r.shape[0]
        device, dtype = r.device, r.dtype
        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=device)
        n_graphs = cell.shape[0]
        n_q = q.shape[1]

        # --- reciprocal lattice vectors (kept differentiable in `cell`) ---
        cell_inv = torch.linalg.inv(cell)                 # [G,3,3]
        G = self.twopi * cell_inv.transpose(1, 2)         # [G,3,3]  G = 2pi (M^-1)^T
        norms = torch.norm(cell, dim=2)                   # [G,3]  |a_i| per structure
        Nk = torch.clamp((norms / self.dl).to(torch.int64), min=1)   # [G,3]

        # --- enumerate each graph's own [-Nk,Nk] box ---
        dims = 2 * Nk + 1                                  # [G,3]  box size per axis
        Pbox = dims.prod(1)                                # [G]    candidates per graph
        kgraph = torch.repeat_interleave(                  # [M0]  graph of each candidate
            torch.arange(n_graphs, device=device), Pbox
        )
        box_off = torch.cumsum(Pbox, 0) - Pbox             # [G]
        box_local = torch.arange(kgraph.shape[0], device=device) - box_off[kgraph]
        dg = dims[kgraph]                                  # [M0,3]
        i0 = box_local // (dg[:, 1] * dg[:, 2])            # decode flat idx -> (i0,i1,i2)
        rem = box_local % (dg[:, 1] * dg[:, 2])
        i1 = rem // dg[:, 2]
        i2 = rem % dg[:, 2]
        nvec = (torch.stack([i0, i1, i2], -1) - Nk[kgraph]).to(dtype)  # [M0,3] in [-Nk,Nk]

        kvec = torch.einsum('md,mde->me', nvec, G[kgraph])  # [M0,3]
        k_sq = (kvec ** 2).sum(dim=-1)                      # [M0]

        # --- validity: spherical cutoff AND hemisphere (box holds by construction) ---
        spherical = (k_sq > 0) & (k_sq <= self.k_sq_max)
        non_zero = (nvec != 0).to(torch.int64)
        first_nz = torch.argmax(non_zero, dim=1)
        sign = torch.gather(nvec, 1, first_nz.unsqueeze(1)).squeeze(1)
        all_zero = (nvec == 0).all(dim=1)
        hemisphere = (sign > 0) | all_zero
        factors = torch.where(all_zero, torch.ones_like(k_sq),
                              torch.full_like(k_sq, 2.0))
        keep = spherical & hemisphere                      # [M0]

        # --- ragged k-list ---
        kgraph = kgraph[keep]
        kvec_flat = kvec[keep]                             # [M,3]
        k_sq_flat = k_sq[keep]                             # [M]
        factors_flat = factors[keep]                       # [M]

        # --- within-graph (atom, k-point) pairs (atoms contiguous by graph) ---
        na = torch.bincount(batch, minlength=n_graphs)             # [G]
        atom_off = torch.cumsum(na, 0) - na                        # [G]
        counts = na[kgraph]                                        # [M]  atoms per k-point
        base = torch.repeat_interleave(atom_off[kgraph], counts)   # [P_tot]
        block_off = torch.cumsum(counts, 0) - counts
        local = torch.arange(base.shape[0], device=device) \
            - torch.repeat_interleave(block_off, counts)
        a_idx = (base + local).to(torch.int32)                     # [P_tot]

        # --- structure factor via scatter-add ---
        M = kvec_flat.shape[0]
        k_idx = torch.repeat_interleave(torch.arange(M, device=device), counts)  # [P_tot]
        kvec_pair = torch.repeat_interleave(kvec_flat, counts, dim=0)  # [P_tot,3]
        phase = (r[a_idx] * kvec_pair).sum(-1)                     # [P_tot]
        qa = q[a_idx]                                              # [P_tot,n_q]
        S_real = torch.zeros(M, n_q, device=device, dtype=dtype).index_add(
            0, k_idx, qa * torch.cos(phase).unsqueeze(-1))         # [M,n_q]
        S_imag = torch.zeros(M, n_q, device=device, dtype=dtype).index_add(
            0, k_idx, qa * torch.sin(phase).unsqueeze(-1))
        S_sq = S_real ** 2 + S_imag ** 2                          # [M,n_q]

        # --- assemble per-graph potential (k_sq_flat > 0 guaranteed) ---
        kfac = torch.exp(-self.sigma_sq_half * k_sq_flat) / k_sq_flat
        weight = (factors_flat * kfac).unsqueeze(-1)              # [M,1]
        pot = torch.zeros(n_graphs, n_q, device=device, dtype=dtype)
        pot.index_add_(0, kgraph, weight * S_sq)                  # [G,n_q]
        volume = torch.linalg.det(cell).abs()
        pot = pot / volume.unsqueeze(1)

        if self.remove_self_interaction:
            q_sq_tot = torch.zeros(n_graphs, n_q, device=device, dtype=dtype)
            q_sq_tot.index_add_(0, batch, q ** 2)
            pot = pot - q_sq_tot / (self.sigma * (2 * torch.pi) ** 1.5)
        return (pot * self.norm_factor).sum(dim=1)


class DirectSum(nn.Module):
    """
    Open-boundary (non-pbc) screened Coulomb energy, batched over graphs.

    E_g = (norm_factor / 4pi) * sum_{i != j in g} q_i q_j erf(r_ij / sqrt(2)s) / r_ij

    The i != j sum has no self term: remove_self_interaction=True does
    nothing, False adds it (reciprocal kernels subtract it when True).
    """

    def __init__(
        self,
        sigma: float = 1.0,
        remove_self_interaction: bool = True,
        norm_factor: float = 90.4756,
    ):
        super().__init__()
        self.sigma = sigma
        self.remove_self_interaction = remove_self_interaction
        self.norm_factor = norm_factor
        self.twopi = 2.0 * torch.pi

    def forward(
        self,
        q: torch.Tensor,                       # [N, n_q] or [N]
        r: torch.Tensor,                       # [N, 3]
        batch: Optional[torch.Tensor] = None,  # [N]
        n_graphs: Optional[int] = None,
    ) -> torch.Tensor:                         # [n_graphs]
        if q.dim() == 1:
            q = q.unsqueeze(1)
        N = r.shape[0]
        device, dtype = r.device, r.dtype
        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=device)
        if n_graphs is None:
            n_graphs = int(batch.max().item()) + 1 if N > 0 else 0
        n_q = q.shape[1]

        # within-graph (i, j) pairs: atom i pairs with every atom of its graph
        na = torch.bincount(batch, minlength=n_graphs)             # [G]
        atom_off = torch.cumsum(na, 0) - na                        # [G]
        counts = na[batch]                                         # [N]
        base = torch.repeat_interleave(atom_off[batch], counts)    # [P]
        block_off = torch.cumsum(counts, 0) - counts
        local = torch.arange(base.shape[0], device=device) \
            - torch.repeat_interleave(block_off, counts)
        j_idx = base + local                                       # [P]
        i_idx = torch.repeat_interleave(
            torch.arange(N, device=device), counts
        )                                                          # [P]
        offdiag = i_idx != j_idx
        i_idx = i_idx[offdiag]
        j_idx = j_idx[offdiag]

        # clamp matches the triton kernel; erf(ar)/r is finite as r -> 0
        r_ij = torch.norm(r[i_idx] - r[j_idx], dim=-1).clamp(min=1e-10)
        f = torch.special.erf(r_ij / (self.sigma * 2.0 ** 0.5)) / r_ij

        e_pair = q[i_idx] * q[j_idx] * f.unsqueeze(-1)             # [P', n_q]
        pot = torch.zeros(n_graphs, n_q, device=device, dtype=dtype)
        pot.index_add_(0, batch[i_idx], e_pair)
        pot = pot / self.twopi / 2.0

        if not self.remove_self_interaction:
            q_sq_tot = torch.zeros(n_graphs, n_q, device=device, dtype=dtype)
            q_sq_tot.index_add_(0, batch, q ** 2)
            pot = pot + q_sq_tot / (self.sigma * self.twopi ** 1.5)
        return (pot * self.norm_factor).sum(dim=1)


class AutoBatchedEwald(nn.Module):
    """
    Per-batch dispatch between BatchedEwald and FlatBatchedEwald.
    Batched for homogeneous, Flat for heterogeneous.
    """

    def __init__(
        self,
        dl: float = 2.0,
        sigma: float = 1.0,
        remove_self_interaction: bool = True,
        norm_factor: float = 90.4756,
        flat_overhead: float = 3.0,
    ):
        super().__init__()
        self.dl = dl
        self.flat_overhead = flat_overhead
        self.batched = BatchedEwald(dl, sigma, remove_self_interaction, norm_factor)
        self.flat = FlatBatchedEwald(dl, sigma, remove_self_interaction, norm_factor)

    def _nkbox(self, cell: torch.Tensor) -> torch.Tensor:
        Nk = torch.clamp((torch.norm(cell, dim=2) / self.dl).to(torch.int64), min=1)
        return Nk

    def _costs(self, cell: torch.Tensor, batch: Optional[torch.Tensor]):
        n_graphs = cell.shape[0]
        Nk = self._nkbox(cell)
        Vg = (2 * Nk + 1).prod(1).to(torch.float64)
        Vshared = Vg.max()
        if batch is None:
            na = torch.full((n_graphs,), 1.0, dtype=torch.float64, device=cell.device)
        else:
            na = torch.bincount(batch, minlength=n_graphs).to(torch.float64)
        cost_batched = na.sum() * Vshared
        cost_flat = self.flat_overhead * (na * Vg).sum()
        return Vg, Vshared, cost_batched, cost_flat

    def homogeneity(
        self,
        cell: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> dict:
        """Diagnostic: heterogeneity ratio and the kernel it routes to."""
        Vg, Vshared, cost_batched, cost_flat = self._costs(cell, batch)
        return {
            'het_ratio': (Vshared / Vg.median()).item(),
            'cost_batched': cost_batched.item(),
            'cost_flat': cost_flat.item(),
            'kernel': 'flat' if cost_flat.item() < cost_batched.item() else 'batched',
        }

    def forward(
        self,
        q: torch.Tensor,
        r: torch.Tensor,
        cell: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch is None or cell.shape[0] == 1:   # single structure -> homogeneous
            return self.batched(q=q, r=r, cell=cell, batch=batch)
        with torch.no_grad():
            _, _, cost_batched, cost_flat = self._costs(cell, batch)
            use_flat = bool((cost_flat < cost_batched).item())   # single sync
        kernel = self.flat if use_flat else self.batched
        return kernel(q=q, r=r, cell=cell, batch=batch)
