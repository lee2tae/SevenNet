"""
1D-periodic Ewald kernels for SevenNet LES.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # triton is optional
    _HAS_TRITON = False


def _leggauss(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    import numpy as np

    x, w = np.polynomial.legendre.leggauss(n)
    # map from [-1, 1] to [0, 1]
    return (
        torch.tensor(0.5 * (x + 1.0), dtype=torch.float64),
        torch.tensor(0.5 * w, dtype=torch.float64),
    )


class _Ewald1DGrids(nn.Module):
    """Shared quadrature/mode machinery for the torch and triton modules."""

    def __init__(self, sigma: float, tol: float, n_nodes: int):
        super().__init__()
        self.sigma = sigma
        self.tol = tol
        self.n_nodes = n_nodes
        self.alpha = 1.0 / (sigma * 2.0 ** 0.5)
        self.k_max = (2.0 * math.log(1.0 / tol)) ** 0.5 / sigma
        x, w = _leggauss(n_nodes)
        self.register_buffer('gl_x', x)     # nodes on [0, 1]
        self.register_buffer('gl_w', w)
        x0, w0 = _leggauss(2 * n_nodes)
        self.register_buffer('gl_x0', x0)
        self.register_buffer('gl_w0', w0)

    def geometry(
        self, r: torch.Tensor, axis: torch.Tensor, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split positions into transverse xperp and scaled axial zeta."""
        L = torch.linalg.norm(axis, dim=1)              # (G,)
        a_hat = axis / L.unsqueeze(1)                   # (G, 3)
        z = (r * a_hat[batch]).sum(dim=1)               # (N,)
        xperp = r - z.unsqueeze(1) * a_hat[batch]       # (N, 3)
        k1 = 2.0 * math.pi / L                          # (G,)
        zeta = z * k1[batch]                            # (N,)
        beta1 = 0.5 * (self.sigma * k1) ** 2            # (G,)
        inv_l = 1.0 / L                                 # (G,)
        return xperp, zeta, inv_l, beta1

    def grids(
        self,
        xperp: torch.Tensor,
        inv_l: torch.Tensor,
        batch: torch.Tensor,
        n_graphs: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """Detached PER-GRAPH node grids + mode counts.

        Each graph's grid extent depends only on its own L and transverse
        span, so a graph's energy is exactly batch-invariant (identical to
        its single-graph value up to reduction order)."""
        with torch.no_grad():
            beta1 = (
                0.5 * (self.sigma * 2.0 * math.pi) ** 2
            ) * inv_l.to(dtype) ** 2                                 # (G,)
            log_tol = math.log(1.0 / self.tol)
            s_hi = (0.5 * torch.log(log_tol / beta1)).clamp(min=0.05)
            fmax = float('inf')  # torch.finfo is not TorchScript-able
            idx3 = batch.unsqueeze(1).expand(-1, 3)
            mx = torch.full(
                (n_graphs, 3), -fmax, device=device, dtype=dtype
            ).scatter_reduce_(0, idx3, xperp.to(dtype), 'amax')
            mn = torch.full(
                (n_graphs, 3), fmax, device=device, dtype=dtype
            ).scatter_reduce_(0, idx3, xperp.to(dtype), 'amin')
            u_max = ((mx - mn).clamp(min=0.0) ** 2).sum(dim=1)      # (G,)
            s0_hi = (0.5 * torch.log(
                (self.alpha ** 2 * u_max).clamp(min=1.0) / self.tol
            )).clamp(min=1.0)
            m_modes = torch.clamp(
                torch.floor(self.k_max / (2.0 * math.pi * inv_l)), min=1
            ).to(torch.int64)                                       # (G,)
        gx = self.gl_x.to(device=device, dtype=dtype)
        gw = self.gl_w.to(device=device, dtype=dtype)
        s = s_hi.unsqueeze(1) * gx                                  # (G, Gn)
        p_nodes = self.alpha ** 2 * torch.exp(-2.0 * s)
        e2 = torch.exp(2.0 * s)
        w_nodes = 4.0 * s_hi.unsqueeze(1) * gw
        gx0 = self.gl_x0.to(device=device, dtype=dtype)
        gw0 = self.gl_w0.to(device=device, dtype=dtype)
        s0 = s0_hi.unsqueeze(1) * gx0                               # (G, G0)
        p0_nodes = self.alpha ** 2 * torch.exp(-2.0 * s0)
        w0_nodes = 2.0 * s0_hi.unsqueeze(1) * gw0
        return p_nodes, e2, w_nodes, p0_nodes, w0_nodes, m_modes


class Ewald1DSum(nn.Module):
    """
    1D-periodic (wire) screened Coulomb energy, batched over graphs.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        remove_self_interaction: bool = True,
        norm_factor: float = 90.4756,
        tol: float = 1e-7,
        n_nodes: int = 32,
    ):
        super().__init__()
        self.sigma = sigma
        self.remove_self_interaction = remove_self_interaction
        self.norm_factor = norm_factor
        self.twopi = 2.0 * torch.pi
        self.grids = _Ewald1DGrids(sigma, tol, n_nodes)

    def forward(
        self,
        q: torch.Tensor,                       # [N, n_q] or [N]
        r: torch.Tensor,                       # [N, 3]
        axis: torch.Tensor,                    # [n_graphs, 3] nonzero row
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
        if N == 0:
            return torch.zeros(n_graphs, device=device, dtype=dtype)

        xperp, zeta, inv_l, beta1 = self.grids.geometry(r, axis, batch)
        p_n, e2, w_n, p0_n, w0_n, m_modes = self.grids.grids(
            xperp, inv_l, batch, n_graphs, dtype, device
        )

        # within-graph (i, j) pairs INCLUDING i == j (own-image interaction)
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

        delta = xperp[i_idx] - xperp[j_idx]                        # [P, 3]
        u = (delta ** 2).sum(dim=-1)                               # [P]
        th = zeta[i_idx] - zeta[j_idx]                             # [P]

        # per graph (pairs of a graph are contiguous): k_z = 0 mode
        # (-Ein(alpha^2 u)/2, background-regularized) + k_z != 0 modes
        pair_off = 0
        f_pair = torch.zeros_like(u)
        for g in range(n_graphs):
            na_g = int(na[g].item())
            npair = na_g * na_g  # int ** 2 is float in TorchScript
            if npair == 0:
                continue
            # explicit bounds: slice() objects are not TorchScript-able
            end = pair_off + npair
            u_g = u[pair_off:end].unsqueeze(-1)                    # [Pg, 1]
            v0 = ((torch.exp(-u_g * p0_n[g]) - 1.0) * w0_n[g]).sum(dim=-1)
            mg = int(m_modes[g].item())
            m = torch.arange(1, mg + 1, device=device, dtype=dtype)
            t_g = w_n[g] * torch.exp(
                -beta1[g] * (m ** 2).unsqueeze(-1) * e2[g]
            )                                                      # [M, Gn]
            v_m = torch.exp(-u_g * p_n[g]) @ t_g.transpose(0, 1)   # [Pg, M]
            cos_m = torch.cos(th[pair_off:end].unsqueeze(-1) * m)  # [Pg, M]
            f_pair[pair_off:end] = v0 + (cos_m * v_m).sum(dim=-1)
            pair_off = end
        f_pair = f_pair * inv_l[batch[i_idx]]                      # [P]

        e_pair = q[i_idx] * q[j_idx] * f_pair.unsqueeze(-1)        # [P, n_q]
        pot = torch.zeros(n_graphs, n_q, device=device, dtype=dtype)
        pot.index_add_(0, batch[i_idx], e_pair)
        pot = pot / self.twopi / 2.0

        # Fourier sum contains the (i,0)=(j,0) self term 2 alpha/sqrt(pi);
        # subtract it exactly like the reciprocal kernels.
        if self.remove_self_interaction:
            q_sq_tot = torch.zeros(
                n_graphs, n_q, device=device, dtype=dtype
            )
            q_sq_tot.index_add_(0, batch, q ** 2)
            pot = pot - q_sq_tot / (self.sigma * self.twopi ** 1.5)
        return (pot * self.norm_factor).sum(dim=1)


# ===================== triton implementation =====================
if _HAS_TRITON:
    BLK1D = 64

    @triton.jit
    def _w1_fwd(q, xp, zt, ba, aoff, na, invl, be1, mg,
                P, E2, W, P0, W0, phi, N,
                NQ: tl.constexpr, NQP: tl.constexpr,
                GN: tl.constexpr, G0: tl.constexpr, B: tl.constexpr):
        i = tl.program_id(0)
        if i >= N:
            return
        g = tl.load(ba + i); a0i = tl.load(aoff + g); n = tl.load(na + g)
        il = tl.load(invl + g); b1 = tl.load(be1 + g); M = tl.load(mg + g)
        xi0 = tl.load(xp+i*3+0); xi1 = tl.load(xp+i*3+1); xi2 = tl.load(xp+i*3+2)
        zi = tl.load(zt + i)
        ch = tl.arange(0, NQP); chm = ch < NQ
        acc = tl.zeros((NQP,), dtype=phi.dtype.element_ty)
        for jj in range(0, n, B):
            o = jj + tl.arange(0, B); mk = o < n; idx = a0i + o
            x0 = tl.load(xp+idx*3+0, mask=mk, other=0.)
            x1 = tl.load(xp+idx*3+1, mask=mk, other=0.)
            x2 = tl.load(xp+idx*3+2, mask=mk, other=0.)
            d0 = xi0-x0; d1 = xi1-x1; d2 = xi2-x2
            u = d0*d0 + d1*d1 + d2*d2
            th = zi - tl.load(zt+idx, mask=mk, other=0.)
            cth = tl.cos(th)
            F = tl.zeros((B,), dtype=phi.dtype.element_ty)
            for n0 in range(G0):
                p0 = tl.load(P0+g*G0+n0); w0 = tl.load(W0+g*G0+n0)
                F += w0*(tl.exp(-u*p0) - 1.0)
            for nn in range(GN):
                p = tl.load(P+g*GN+nn); e2n = tl.load(E2+g*GN+nn)
                wn = tl.load(W+g*GN+nn)
                A = tl.exp(-u*p)
                be = b1*e2n
                cp = tl.full((B,), 1.0, dtype=phi.dtype.element_ty)
                cc = cth
                for m in range(1, M+1):
                    t = wn*tl.exp(-be*(m*m))
                    F += t*A*cc
                    cn = 2.0*cth*cc - cp; cp = cc; cc = cn
            F = tl.where(mk, F, 0.0)
            qj = tl.load(q + idx[:, None]*NQ + ch[None, :],
                         mask=mk[:, None] & chm[None, :], other=0.)
            acc += tl.sum(qj * F[:, None], axis=0)
        tl.store(phi + i*NQ + ch, il*acc, mask=chm)

    @triton.jit
    def _w1_bwd_geo(q, xp, zt, dphi, ba, aoff, na, invl, be1, mg,
                    P, E2, W, P0, W0, dxp, dzt, dil, dbe, N,
                    NQ: tl.constexpr, NQP: tl.constexpr,
                    GN: tl.constexpr, G0: tl.constexpr, B: tl.constexpr):
        # dxp_i = sum_j 2 il W F_u d ; dzt_i = sum_j il W F_th
        # dil_i = sum_j w F (partial) ; dbe_i = sum_j il w F_be (partial)
        # W = sum_c (dphi_i q_j + dphi_j q_i) ; w = sum_c dphi_i q_j
        i = tl.program_id(0)
        if i >= N:
            return
        g = tl.load(ba + i); a0i = tl.load(aoff + g); n = tl.load(na + g)
        il = tl.load(invl + g); b1 = tl.load(be1 + g); M = tl.load(mg + g)
        xi0 = tl.load(xp+i*3+0); xi1 = tl.load(xp+i*3+1); xi2 = tl.load(xp+i*3+2)
        zi = tl.load(zt + i)
        ch = tl.arange(0, NQP); chm = ch < NQ
        qi = tl.load(q + i*NQ + ch, mask=chm, other=0.)
        dpi = tl.load(dphi + i*NQ + ch, mask=chm, other=0.)
        z = tl.zeros((), dtype=dxp.dtype.element_ty)
        ox0 = z; ox1 = z; ox2 = z; oz = z; oil = z; obe = z
        for jj in range(0, n, B):
            o = jj + tl.arange(0, B); mk = o < n; idx = a0i + o
            x0 = tl.load(xp+idx*3+0, mask=mk, other=0.)
            x1 = tl.load(xp+idx*3+1, mask=mk, other=0.)
            x2 = tl.load(xp+idx*3+2, mask=mk, other=0.)
            d0 = xi0-x0; d1 = xi1-x1; d2 = xi2-x2
            u = d0*d0 + d1*d1 + d2*d2
            th = zi - tl.load(zt+idx, mask=mk, other=0.)
            cth = tl.cos(th); sth = tl.sin(th)
            F = tl.zeros((B,), dtype=dxp.dtype.element_ty)
            FU = tl.zeros((B,), dtype=dxp.dtype.element_ty)
            FTH = tl.zeros((B,), dtype=dxp.dtype.element_ty)
            FBE = tl.zeros((B,), dtype=dxp.dtype.element_ty)
            for n0 in range(G0):
                p0 = tl.load(P0+g*G0+n0); w0 = tl.load(W0+g*G0+n0)
                a0e = tl.exp(-u*p0)
                F += w0*(a0e - 1.0)
                FU += -w0*p0*a0e
            for nn in range(GN):
                p = tl.load(P+g*GN+nn); e2n = tl.load(E2+g*GN+nn)
                wn = tl.load(W+g*GN+nn)
                A = tl.exp(-u*p)
                be = b1*e2n
                cp = tl.full((B,), 1.0, dtype=dxp.dtype.element_ty)
                cc = cth
                sp = tl.zeros((B,), dtype=dxp.dtype.element_ty)
                sc = sth
                for m in range(1, M+1):
                    t = wn*tl.exp(-be*(m*m))
                    tA = t*A
                    F += tA*cc
                    FU += -p*tA*cc
                    FTH += -m*tA*sc
                    FBE += -(m*m)*e2n*tA*cc
                    cn = 2.0*cth*cc - cp; cp = cc; cc = cn
                    sn = 2.0*cth*sc - sp; sp = sc; sc = sn
            qj = tl.load(q + idx[:, None]*NQ + ch[None, :],
                         mask=mk[:, None] & chm[None, :], other=0.)
            dpj = tl.load(dphi + idx[:, None]*NQ + ch[None, :],
                          mask=mk[:, None] & chm[None, :], other=0.)
            Wv = tl.sum(dpi[None, :]*qj + dpj*qi[None, :], axis=1)
            wv = tl.sum(dpi[None, :]*qj, axis=1)
            Wv = tl.where(mk, Wv, 0.0); wv = tl.where(mk, wv, 0.0)
            cU = 2.0*il*Wv*FU
            ox0 += tl.sum(cU*d0); ox1 += tl.sum(cU*d1); ox2 += tl.sum(cU*d2)
            oz += tl.sum(il*Wv*FTH)
            oil += tl.sum(wv*F)
            obe += tl.sum(il*wv*FBE)
        tl.store(dxp+i*3+0, ox0); tl.store(dxp+i*3+1, ox1)
        tl.store(dxp+i*3+2, ox2)
        tl.store(dzt+i, oz); tl.store(dil+i, oil); tl.store(dbe+i, obe)

    @triton.jit
    def _w1_ddw_pq(q, xp, zt, dphi, gdq, gdxp, gdzt, gdil, gdbe,
                   ba, aoff, na, invl, be1, mg, P, E2, W, P0, W0,
                   gdphi_o, gq_o, N,
                   NQ: tl.constexpr, NQP: tl.constexpr,
                   GN: tl.constexpr, G0: tl.constexpr, B: tl.constexpr):
        # S = 2 il F_u Dx + il F_th Dz + gdil_g F + il gdbe_g F_be
        # gdphi_i = sum_j [il gdq_j F + q_j S]; gq_i = sum_j dphi_j S
        i = tl.program_id(0)
        if i >= N:
            return
        g = tl.load(ba + i); a0i = tl.load(aoff + g); n = tl.load(na + g)
        il = tl.load(invl + g); b1 = tl.load(be1 + g); M = tl.load(mg + g)
        gil = tl.load(gdil + g); gbe = tl.load(gdbe + g)
        xi0 = tl.load(xp+i*3+0); xi1 = tl.load(xp+i*3+1); xi2 = tl.load(xp+i*3+2)
        zi = tl.load(zt + i)
        gxi0 = tl.load(gdxp+i*3+0); gxi1 = tl.load(gdxp+i*3+1)
        gxi2 = tl.load(gdxp+i*3+2)
        gzi = tl.load(gdzt + i)
        ch = tl.arange(0, NQP); chm = ch < NQ
        acc_p = tl.zeros((NQP,), dtype=gdphi_o.dtype.element_ty)
        acc_q = tl.zeros((NQP,), dtype=gdphi_o.dtype.element_ty)
        for jj in range(0, n, B):
            o = jj + tl.arange(0, B); mk = o < n; idx = a0i + o
            x0 = tl.load(xp+idx*3+0, mask=mk, other=0.)
            x1 = tl.load(xp+idx*3+1, mask=mk, other=0.)
            x2 = tl.load(xp+idx*3+2, mask=mk, other=0.)
            d0 = xi0-x0; d1 = xi1-x1; d2 = xi2-x2
            u = d0*d0 + d1*d1 + d2*d2
            th = zi - tl.load(zt+idx, mask=mk, other=0.)
            cth = tl.cos(th); sth = tl.sin(th)
            F = tl.zeros((B,), dtype=gdphi_o.dtype.element_ty)
            FU = tl.zeros((B,), dtype=gdphi_o.dtype.element_ty)
            FTH = tl.zeros((B,), dtype=gdphi_o.dtype.element_ty)
            FBE = tl.zeros((B,), dtype=gdphi_o.dtype.element_ty)
            for n0 in range(G0):
                p0 = tl.load(P0+g*G0+n0); w0 = tl.load(W0+g*G0+n0)
                a0e = tl.exp(-u*p0)
                F += w0*(a0e - 1.0)
                FU += -w0*p0*a0e
            for nn in range(GN):
                p = tl.load(P+g*GN+nn); e2n = tl.load(E2+g*GN+nn)
                wn = tl.load(W+g*GN+nn)
                A = tl.exp(-u*p)
                be = b1*e2n
                cp = tl.full((B,), 1.0, dtype=gdphi_o.dtype.element_ty)
                cc = cth
                sp = tl.zeros((B,), dtype=gdphi_o.dtype.element_ty)
                sc = sth
                for m in range(1, M+1):
                    t = wn*tl.exp(-be*(m*m))
                    tA = t*A
                    F += tA*cc
                    FU += -p*tA*cc
                    FTH += -m*tA*sc
                    FBE += -(m*m)*e2n*tA*cc
                    cn = 2.0*cth*cc - cp; cp = cc; cc = cn
                    sn = 2.0*cth*sc - sp; sp = sc; sc = sn
            gx0 = gxi0 - tl.load(gdxp+idx*3+0, mask=mk, other=0.)
            gx1 = gxi1 - tl.load(gdxp+idx*3+1, mask=mk, other=0.)
            gx2 = gxi2 - tl.load(gdxp+idx*3+2, mask=mk, other=0.)
            Dx = gx0*d0 + gx1*d1 + gx2*d2
            Dz = gzi - tl.load(gdzt+idx, mask=mk, other=0.)
            S = 2.0*il*FU*Dx + il*FTH*Dz + gil*F + il*gbe*FBE
            S = tl.where(mk, S, 0.0)
            F = tl.where(mk, F, 0.0)
            qj = tl.load(q + idx[:, None]*NQ + ch[None, :],
                         mask=mk[:, None] & chm[None, :], other=0.)
            dpj = tl.load(dphi + idx[:, None]*NQ + ch[None, :],
                          mask=mk[:, None] & chm[None, :], other=0.)
            gqj = tl.load(gdq + idx[:, None]*NQ + ch[None, :],
                          mask=mk[:, None] & chm[None, :], other=0.)
            acc_p += tl.sum(il*gqj*F[:, None] + qj*S[:, None], axis=0)
            acc_q += tl.sum(dpj*S[:, None], axis=0)
        tl.store(gdphi_o + i*NQ + ch, acc_p, mask=chm)
        tl.store(gq_o + i*NQ + ch, acc_q, mask=chm)

    @triton.jit
    def _w1_ddw_geo(q, xp, zt, dphi, gdq, gdxp, gdzt, gdil, gdbe,
                    ba, aoff, na, invl, be1, mg, P, E2, W, P0, W0,
                    gxp_o, gzt_o, gil_o, gbe_o, N,
                    NQ: tl.constexpr, NQP: tl.constexpr,
                    GN: tl.constexpr, G0: tl.constexpr, B: tl.constexpr):
        # gxp_i = sum_j [coef*d + 2 il W F_u Gx],
        #   coef = 2 il Aw F_u + 4 il W F_uu Dx + 2 il W F_uth Dz
        #          + 2 gil W F_u + 2 il gbe W F_ube
        # gzt_i = sum_j [il Aw F_th + 2 il W F_uth Dx + il W F_thth Dz
        #                + gil W F_th + il gbe W F_thbe]
        # gil_i = sum_j [a F + 2 W F_u (gdxp_i . d) + W F_th gdzt_i
        #                + gbe w F_be]                       (ordered partial)
        # gbe_i = sum_j [il a F_be + 2 il W F_ube (gdxp_i . d)
        #                + il W F_thbe gdzt_i + gil w F_be
        #                + il gbe w F_bebe]                  (ordered partial)
        # Aw = sum_c (dphi_i gdq_j + gdq_i dphi_j); a = sum_c dphi_i gdq_j
        i = tl.program_id(0)
        if i >= N:
            return
        g = tl.load(ba + i); a0i = tl.load(aoff + g); n = tl.load(na + g)
        il = tl.load(invl + g); b1 = tl.load(be1 + g); M = tl.load(mg + g)
        gil = tl.load(gdil + g); gbe = tl.load(gdbe + g)
        xi0 = tl.load(xp+i*3+0); xi1 = tl.load(xp+i*3+1); xi2 = tl.load(xp+i*3+2)
        zi = tl.load(zt + i)
        gxi0 = tl.load(gdxp+i*3+0); gxi1 = tl.load(gdxp+i*3+1)
        gxi2 = tl.load(gdxp+i*3+2)
        gzi = tl.load(gdzt + i)
        ch = tl.arange(0, NQP); chm = ch < NQ
        qi = tl.load(q + i*NQ + ch, mask=chm, other=0.)
        dpi = tl.load(dphi + i*NQ + ch, mask=chm, other=0.)
        gqi = tl.load(gdq + i*NQ + ch, mask=chm, other=0.)
        z = tl.zeros((), dtype=gxp_o.dtype.element_ty)
        ox0 = z; ox1 = z; ox2 = z; oz = z; oil = z; obe = z
        for jj in range(0, n, B):
            o = jj + tl.arange(0, B); mk = o < n; idx = a0i + o
            x0 = tl.load(xp+idx*3+0, mask=mk, other=0.)
            x1 = tl.load(xp+idx*3+1, mask=mk, other=0.)
            x2 = tl.load(xp+idx*3+2, mask=mk, other=0.)
            d0 = xi0-x0; d1 = xi1-x1; d2 = xi2-x2
            u = d0*d0 + d1*d1 + d2*d2
            th = zi - tl.load(zt+idx, mask=mk, other=0.)
            cth = tl.cos(th); sth = tl.sin(th)
            dt_ = gxp_o.dtype.element_ty
            F = tl.zeros((B,), dtype=dt_); FU = tl.zeros((B,), dtype=dt_)
            FTH = tl.zeros((B,), dtype=dt_); FUU = tl.zeros((B,), dtype=dt_)
            FUTH = tl.zeros((B,), dtype=dt_); FTHTH = tl.zeros((B,), dtype=dt_)
            FBE = tl.zeros((B,), dtype=dt_); FUBE = tl.zeros((B,), dtype=dt_)
            FTHBE = tl.zeros((B,), dtype=dt_); FBEBE = tl.zeros((B,), dtype=dt_)
            for n0 in range(G0):
                p0 = tl.load(P0+g*G0+n0); w0 = tl.load(W0+g*G0+n0)
                a0e = tl.exp(-u*p0)
                F += w0*(a0e - 1.0)
                FU += -w0*p0*a0e
                FUU += w0*p0*p0*a0e
            for nn in range(GN):
                p = tl.load(P+g*GN+nn); e2n = tl.load(E2+g*GN+nn)
                wn = tl.load(W+g*GN+nn)
                A = tl.exp(-u*p)
                be = b1*e2n
                cp = tl.full((B,), 1.0, dtype=dt_)
                cc = cth
                sp = tl.zeros((B,), dtype=dt_)
                sc = sth
                for m in range(1, M+1):
                    m2e = (m*m)*e2n
                    t = wn*tl.exp(-be*(m*m))
                    tA = t*A
                    tAc = tA*cc; tAs = tA*sc
                    F += tAc
                    FU += -p*tAc
                    FTH += -m*tAs
                    FUU += p*p*tAc
                    FUTH += p*m*tAs
                    FTHTH += -(m*m)*tAc
                    FBE += -m2e*tAc
                    FUBE += m2e*p*tAc
                    FTHBE += m2e*m*tAs
                    FBEBE += m2e*m2e*tAc
                    cn = 2.0*cth*cc - cp; cp = cc; cc = cn
                    sn = 2.0*cth*sc - sp; sp = sc; sc = sn
            gx0 = gxi0 - tl.load(gdxp+idx*3+0, mask=mk, other=0.)
            gx1 = gxi1 - tl.load(gdxp+idx*3+1, mask=mk, other=0.)
            gx2 = gxi2 - tl.load(gdxp+idx*3+2, mask=mk, other=0.)
            Dx = gx0*d0 + gx1*d1 + gx2*d2
            Dz = gzi - tl.load(gdzt+idx, mask=mk, other=0.)
            pdx = gxi0*d0 + gxi1*d1 + gxi2*d2
            qj = tl.load(q + idx[:, None]*NQ + ch[None, :],
                         mask=mk[:, None] & chm[None, :], other=0.)
            dpj = tl.load(dphi + idx[:, None]*NQ + ch[None, :],
                          mask=mk[:, None] & chm[None, :], other=0.)
            gqj = tl.load(gdq + idx[:, None]*NQ + ch[None, :],
                          mask=mk[:, None] & chm[None, :], other=0.)
            Wv = tl.sum(dpi[None, :]*qj + dpj*qi[None, :], axis=1)
            wv = tl.sum(dpi[None, :]*qj, axis=1)
            Aw = tl.sum(dpi[None, :]*gqj + gqi[None, :]*dpj, axis=1)
            av = tl.sum(dpi[None, :]*gqj, axis=1)
            Wv = tl.where(mk, Wv, 0.0); wv = tl.where(mk, wv, 0.0)
            Aw = tl.where(mk, Aw, 0.0); av = tl.where(mk, av, 0.0)
            coef = (2.0*il*Aw*FU + 4.0*il*Wv*FUU*Dx + 2.0*il*Wv*FUTH*Dz
                    + 2.0*gil*Wv*FU + 2.0*il*gbe*Wv*FUBE)
            cG = 2.0*il*Wv*FU
            ox0 += tl.sum(coef*d0 + cG*gx0)
            ox1 += tl.sum(coef*d1 + cG*gx1)
            ox2 += tl.sum(coef*d2 + cG*gx2)
            oz += tl.sum(il*Aw*FTH + 2.0*il*Wv*FUTH*Dx + il*Wv*FTHTH*Dz
                         + gil*Wv*FTH + il*gbe*Wv*FTHBE)
            oil += tl.sum(av*F + 2.0*Wv*FU*pdx + Wv*FTH*gzi + gbe*wv*FBE)
            obe += tl.sum(il*av*FBE + 2.0*il*Wv*FUBE*pdx + il*Wv*FTHBE*gzi
                          + gil*wv*FBE + il*gbe*wv*FBEBE)
        tl.store(gxp_o+i*3+0, ox0); tl.store(gxp_o+i*3+1, ox1)
        tl.store(gxp_o+i*3+2, ox2)
        tl.store(gzt_o+i, oz); tl.store(gil_o+i, oil); tl.store(gbe_o+i, obe)

    # ----------------------- autograd wiring -----------------------
    class _W1Bwd(torch.autograd.Function):
        @staticmethod
        def forward(ctx, dphi, q, xp, zt, invl, be1,
                    ba, aoff, na, mg, P, E2, W, P0, W0):
            N, NQ = q.shape
            NQP = triton.next_power_of_2(NQ)
            GN = P.shape[1]; G0 = P0.shape[1]
            dphi = dphi.contiguous()
            dq = torch.zeros_like(q)
            dxp = torch.zeros_like(xp)
            dzt = torch.zeros_like(zt)
            dil_p = torch.zeros_like(zt)
            dbe_p = torch.zeros_like(zt)
            # F symmetric: dq_i = sum_j dphi_j il F_ij == fwd kernel on dphi
            _w1_fwd[(N,)](dphi, xp, zt, ba, aoff, na, invl, be1, mg,
                          P, E2, W, P0, W0, dq, N,
                          NQ=NQ, NQP=NQP, GN=GN, G0=G0, B=BLK1D)
            _w1_bwd_geo[(N,)](q, xp, zt, dphi, ba, aoff, na, invl, be1, mg,
                              P, E2, W, P0, W0, dxp, dzt, dil_p, dbe_p, N,
                              NQ=NQ, NQP=NQP, GN=GN, G0=G0, B=BLK1D)
            dinvl = torch.zeros_like(invl).index_add_(0, ba.long(), dil_p)
            dbeta = torch.zeros_like(be1).index_add_(0, ba.long(), dbe_p)
            ctx.save_for_backward(dphi, q, xp, zt, invl, be1,
                                  ba, aoff, na, mg, P, E2, W, P0, W0)
            return dq, dxp, dzt, dinvl, dbeta

        @staticmethod
        def backward(ctx, gdq, gdxp, gdzt, gdinvl, gdbeta):
            (dphi, q, xp, zt, invl, be1,
             ba, aoff, na, mg, P, E2, W, P0, W0) = ctx.saved_tensors
            N, NQ = q.shape
            NQP = triton.next_power_of_2(NQ)
            GN = P.shape[1]; G0 = P0.shape[1]
            gdq = gdq.contiguous(); gdxp = gdxp.contiguous()
            gdzt = gdzt.contiguous()
            gdinvl = gdinvl.contiguous(); gdbeta = gdbeta.contiguous()
            gdphi = torch.zeros_like(dphi)
            gq = torch.zeros_like(q)
            gxp = torch.zeros_like(xp)
            gzt = torch.zeros_like(zt)
            gil_p = torch.zeros_like(zt)
            gbe_p = torch.zeros_like(zt)
            _w1_ddw_pq[(N,)](q, xp, zt, dphi, gdq, gdxp, gdzt, gdinvl,
                             gdbeta, ba, aoff, na, invl, be1, mg,
                             P, E2, W, P0, W0, gdphi, gq, N,
                             NQ=NQ, NQP=NQP, GN=GN, G0=G0, B=BLK1D)
            _w1_ddw_geo[(N,)](q, xp, zt, dphi, gdq, gdxp, gdzt, gdinvl,
                              gdbeta, ba, aoff, na, invl, be1, mg,
                              P, E2, W, P0, W0, gxp, gzt, gil_p, gbe_p, N,
                              NQ=NQ, NQP=NQP, GN=GN, G0=G0, B=BLK1D)
            ginvl = torch.zeros_like(invl).index_add_(0, ba.long(), gil_p)
            gbeta = torch.zeros_like(be1).index_add_(0, ba.long(), gbe_p)
            return (gdphi, gq, gxp, gzt, ginvl, gbeta,
                    None, None, None, None, None, None, None, None, None)

    class _W1Fwd(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, xp, zt, invl, be1,
                    ba, aoff, na, mg, P, E2, W, P0, W0):
            N, NQ = q.shape
            NQP = triton.next_power_of_2(NQ)
            GN = P.shape[1]; G0 = P0.shape[1]
            phi = torch.zeros_like(q)
            _w1_fwd[(N,)](q, xp, zt, ba, aoff, na, invl, be1, mg,
                          P, E2, W, P0, W0, phi, N,
                          NQ=NQ, NQP=NQP, GN=GN, G0=G0, B=BLK1D)
            ctx.save_for_backward(q, xp, zt, invl, be1,
                                  ba, aoff, na, mg, P, E2, W, P0, W0)
            return phi

        @staticmethod
        def backward(ctx, dphi):
            (q, xp, zt, invl, be1,
             ba, aoff, na, mg, P, E2, W, P0, W0) = ctx.saved_tensors
            dq, dxp, dzt, dinvl, dbeta = _W1Bwd.apply(
                dphi, q, xp, zt, invl, be1,
                ba, aoff, na, mg, P, E2, W, P0, W0,
            )
            return (dq, dxp, dzt, dinvl, dbeta,
                    None, None, None, None, None, None, None, None, None)

    class TritonEwald1DSum(nn.Module):
        """Triton counterpart of Ewald1DSum; same convention and signature."""

        def __init__(
            self,
            sigma: float = 1.0,
            remove_self_interaction: bool = True,
            norm_factor: float = 90.4756,
            tol: float = 1e-7,
            n_nodes: int = 32,
        ):
            super().__init__()
            self.sigma = sigma
            self.remove_self_interaction = remove_self_interaction
            self.norm_factor = norm_factor
            self.twopi = 2.0 * torch.pi
            self.grids = _Ewald1DGrids(sigma, tol, n_nodes)

        def forward(
            self,
            q: torch.Tensor,
            r: torch.Tensor,
            axis: torch.Tensor,
            batch: Optional[torch.Tensor] = None,
            n_graphs: Optional[int] = None,
        ) -> torch.Tensor:
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

            xperp, zeta, inv_l, beta1 = self.grids.geometry(r, axis, batch)
            p_n, e2, w_n, p0_n, w0_n, m_modes = self.grids.grids(
                xperp, inv_l, batch, n_graphs, dtype, device
            )
            na = torch.bincount(batch, minlength=n_graphs)
            aoff = (torch.cumsum(na, 0) - na).to(torch.int32).contiguous()
            na32 = na.to(torch.int32).contiguous()
            ba = batch.to(torch.int32).contiguous()
            mg32 = m_modes.to(torch.int32).contiguous()

            phi = _W1Fwd.apply(
                q.contiguous(), xperp.contiguous(), zeta.contiguous(),
                inv_l.contiguous(), beta1.contiguous(),
                ba, aoff, na32, mg32,
                p_n.contiguous(), e2.contiguous(), w_n.contiguous(),
                p0_n.contiguous(), w0_n.contiguous(),
            )
            pot = torch.zeros(n_graphs, n_q, device=device, dtype=dtype)
            pot.index_add_(0, batch, q * phi)
            pot = pot / self.twopi / 2.0
            if self.remove_self_interaction:
                q_sq = torch.zeros(
                    n_graphs, n_q, device=device, dtype=dtype
                )
                q_sq.index_add_(0, batch, q ** 2)
                pot = pot - q_sq / (self.sigma * self.twopi ** 1.5)
            return (pot * self.norm_factor).sum(dim=1)
