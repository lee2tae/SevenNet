"""
Triton Ewald (q + mu) with third-order autograd support (forward / backward / double / triple backward).

Everything is a derivative of the structure-factor map  S_m = sum_i A_im e^{i phi_im},  A = q + i k.mu,  phi = k.r.
For directions u, v in x = (q, mu, r, k):  a_u = u_q + i(k.u_mu + u_k.mu),  phi_u = k.u_r + u_k.r,
a_uv = i(u_k.v_mu + v_k.u_mu),  phi_uv = u_k.v_r + v_k.u_r  (both constant), and
  D_u S           = sum_i (a_u + i phi_u A) e
  D_v D_u S       = sum_i B2 e,   B2 = a_uv + i phi_v a_u + i phi_u a_v + (i phi_uv - phi_u phi_v) A
  D_w D_v D_u S   = sum_i (D_w B2 + i phi_w B2) e
The triple-backward needs (i) D_v D_u S itself and (ii) the gradient of Re sum conj(dS) D_v D_u S w.r.t. x, i.e. the
third directional derivative with w = unit vectors. Everything else reuses the first/second-order kernels of ewald_qu.
The reference for all of this is the pure-torch BatchedEwaldQU (autograd to any order).
"""
from typing import Optional

import torch

from .ewald_qu import _HAS_TRITON, BatchedEwaldQU, _kgrid

if _HAS_TRITON:
    import triton
    import triton.language as tl

    from .ewald_qu import BLK, _bwd_atom_qu, _bwd_k_qu, _ddw_atom_qu, _ddw_dS_qu, _ddw_k_qu, _fwd_qu

    @triton.jit
    def _d2S_qu(q, mu, r, k, kg, aoff, na, uq, umu, ur, uk, vq, vmu, vr, vk, oSr, oSi, M, B: tl.constexpr):
        # oS_m = D_v D_u S_m  (k-major)
        m = tl.program_id(0)
        if m >= M:
            return
        g = tl.load(kg + m); a0 = tl.load(aoff + g); n = tl.load(na + g)
        kx = tl.load(k+m*3+0); ky = tl.load(k+m*3+1); kz = tl.load(k+m*3+2)
        ukx = tl.load(uk+m*3+0); uky = tl.load(uk+m*3+1); ukz = tl.load(uk+m*3+2)
        vkx = tl.load(vk+m*3+0); vky = tl.load(vk+m*3+1); vkz = tl.load(vk+m*3+2)
        z = tl.zeros((), dtype=oSr.dtype.element_ty); ar = z; ai = z
        for a in range(0, n, B):
            o = a + tl.arange(0, B); mk = o < n; idx = a0 + o
            rx = tl.load(r+idx*3+0, mask=mk, other=0.); ry = tl.load(r+idx*3+1, mask=mk, other=0.); rz = tl.load(r+idx*3+2, mask=mk, other=0.)
            ux = tl.load(mu+idx*3+0, mask=mk, other=0.); uy = tl.load(mu+idx*3+1, mask=mk, other=0.); uz = tl.load(mu+idx*3+2, mask=mk, other=0.)
            qa = tl.load(q+idx, mask=mk, other=0.)
            uqa = tl.load(uq+idx, mask=mk, other=0.); vqa = tl.load(vq+idx, mask=mk, other=0.)
            umx = tl.load(umu+idx*3+0, mask=mk, other=0.); umy = tl.load(umu+idx*3+1, mask=mk, other=0.); umz = tl.load(umu+idx*3+2, mask=mk, other=0.)
            vmx = tl.load(vmu+idx*3+0, mask=mk, other=0.); vmy = tl.load(vmu+idx*3+1, mask=mk, other=0.); vmz = tl.load(vmu+idx*3+2, mask=mk, other=0.)
            urx = tl.load(ur+idx*3+0, mask=mk, other=0.); ury = tl.load(ur+idx*3+1, mask=mk, other=0.); urz = tl.load(ur+idx*3+2, mask=mk, other=0.)
            vrx = tl.load(vr+idx*3+0, mask=mk, other=0.); vry = tl.load(vr+idx*3+1, mask=mk, other=0.); vrz = tl.load(vr+idx*3+2, mask=mk, other=0.)
            ph = kx*rx+ky*ry+kz*rz; c = tl.cos(ph); s = tl.sin(ph)
            Ar = qa; Ai = kx*ux+ky*uy+kz*uz
            aur = uqa; aui = kx*umx+ky*umy+kz*umz + ukx*ux+uky*uy+ukz*uz
            avr = vqa; avi = kx*vmx+ky*vmy+kz*vmz + vkx*ux+vky*uy+vkz*uz
            phu = kx*urx+ky*ury+kz*urz + ukx*rx+uky*ry+ukz*rz
            phv = kx*vrx+ky*vry+kz*vrz + vkx*rx+vky*ry+vkz*rz
            auvi = ukx*vmx+uky*vmy+ukz*vmz + vkx*umx+vky*umy+vkz*umz
            phuv = ukx*vrx+uky*vry+ukz*vrz + vkx*urx+vky*ury+vkz*urz
            gr_ = -phu*phv; gi_ = phuv
            B2r = -phv*aui - phu*avi + gr_*Ar - gi_*Ai
            B2i = auvi + phv*aur + phu*avr + gr_*Ai + gi_*Ar
            ar += tl.sum(tl.where(mk, B2r*c - B2i*s, z)); ai += tl.sum(tl.where(mk, B2r*s + B2i*c, z))
        tl.store(oSr+m, ar); tl.store(oSi+m, ai)

    @triton.jit
    def _d3_atom_qu(q, mu, r, k, ba, koff, nk, dSr, dSi, uq, umu, ur, uk, vq, vmu, vr, vk, oq, omu, orr, N, B: tl.constexpr):
        # gradient of Re sum_m conj(dS_m) D_v D_u S_m w.r.t. q_a, mu_a, r_a  (atom-major)
        a = tl.program_id(0)
        if a >= N:
            return
        g = tl.load(ba + a); k0 = tl.load(koff + g); nn = tl.load(nk + g)
        rx = tl.load(r+a*3+0); ry = tl.load(r+a*3+1); rz = tl.load(r+a*3+2)
        ux = tl.load(mu+a*3+0); uy = tl.load(mu+a*3+1); uz = tl.load(mu+a*3+2)
        qa = tl.load(q+a); uqa = tl.load(uq+a); vqa = tl.load(vq+a)
        umx = tl.load(umu+a*3+0); umy = tl.load(umu+a*3+1); umz = tl.load(umu+a*3+2)
        vmx = tl.load(vmu+a*3+0); vmy = tl.load(vmu+a*3+1); vmz = tl.load(vmu+a*3+2)
        urx = tl.load(ur+a*3+0); ury = tl.load(ur+a*3+1); urz = tl.load(ur+a*3+2)
        vrx = tl.load(vr+a*3+0); vry = tl.load(vr+a*3+1); vrz = tl.load(vr+a*3+2)
        z = tl.zeros((), dtype=orr.dtype.element_ty)
        gq = z; mx = z; my = z; mz = z; dx = z; dy = z; dz = z
        for kk in range(0, nn, B):
            o = kk + tl.arange(0, B); mk = o < nn; ki = k0 + o
            kx = tl.load(k+ki*3+0, mask=mk, other=0.); ky = tl.load(k+ki*3+1, mask=mk, other=0.); kz = tl.load(k+ki*3+2, mask=mk, other=0.)
            ukx = tl.load(uk+ki*3+0, mask=mk, other=0.); uky = tl.load(uk+ki*3+1, mask=mk, other=0.); ukz = tl.load(uk+ki*3+2, mask=mk, other=0.)
            vkx = tl.load(vk+ki*3+0, mask=mk, other=0.); vky = tl.load(vk+ki*3+1, mask=mk, other=0.); vkz = tl.load(vk+ki*3+2, mask=mk, other=0.)
            dr_ = tl.load(dSr+ki, mask=mk, other=0.); di_ = tl.load(dSi+ki, mask=mk, other=0.)
            ph = kx*rx+ky*ry+kz*rz; c = tl.cos(ph); s = tl.sin(ph)
            # conj(dS) e = (dr c + di s) + i (dr s - di c)  -> contribution Re[(conj(dS) e) * total]
            Er = dr_*c + di_*s; Ei = dr_*s - di_*c
            Ar = qa; Ai = kx*ux+ky*uy+kz*uz
            aur = uqa; aui = kx*umx+ky*umy+kz*umz + ukx*ux+uky*uy+ukz*uz
            avr = vqa; avi = kx*vmx+ky*vmy+kz*vmz + vkx*ux+vky*uy+vkz*uz
            phu = kx*urx+ky*ury+kz*urz + ukx*rx+uky*ry+ukz*rz
            phv = kx*vrx+ky*vry+kz*vrz + vkx*rx+vky*ry+vkz*rz
            auvi = ukx*vmx+uky*vmy+ukz*vmz + vkx*umx+vky*umy+vkz*umz
            phuv = ukx*vrx+uky*vry+ukz*vrz + vkx*urx+vky*ury+vkz*urz
            gr_ = -phu*phv; gi_ = phuv
            B2r = -phv*aui - phu*avi + gr_*Ar - gi_*Ai
            B2i = auvi + phv*aur + phu*avr + gr_*Ai + gi_*Ar
            # q: total = g
            gq += tl.sum(tl.where(mk, Er*gr_ - Ei*gi_, z))
            # mu^b: total = -phv uk^b - phu vk^b + i k_b g
            tr = -phv*ukx - phu*vkx - kx*gi_; ti = kx*gr_
            mx += tl.sum(tl.where(mk, Er*tr - Ei*ti, z))
            tr = -phv*uky - phu*vky - ky*gi_; ti = ky*gr_
            my += tl.sum(tl.where(mk, Er*tr - Ei*ti, z))
            tr = -phv*ukz - phu*vkz - kz*gi_; ti = kz*gr_
            mz += tl.sum(tl.where(mk, Er*tr - Ei*ti, z))
            # r^a: total = i vk^a a_u + i uk^a a_v - (uk^a phv + phu vk^a) A + i k_a B2
            cf = ukx*phv + phu*vkx
            tr = -vkx*aui - ukx*avi - cf*Ar - kx*B2i; ti = vkx*aur + ukx*avr - cf*Ai + kx*B2r
            dx += tl.sum(tl.where(mk, Er*tr - Ei*ti, z))
            cf = uky*phv + phu*vky
            tr = -vky*aui - uky*avi - cf*Ar - ky*B2i; ti = vky*aur + uky*avr - cf*Ai + ky*B2r
            dy += tl.sum(tl.where(mk, Er*tr - Ei*ti, z))
            cf = ukz*phv + phu*vkz
            tr = -vkz*aui - ukz*avi - cf*Ar - kz*B2i; ti = vkz*aur + ukz*avr - cf*Ai + kz*B2r
            dz += tl.sum(tl.where(mk, Er*tr - Ei*ti, z))
        tl.store(oq+a, gq)
        tl.store(omu+a*3+0, mx); tl.store(omu+a*3+1, my); tl.store(omu+a*3+2, mz)
        tl.store(orr+a*3+0, dx); tl.store(orr+a*3+1, dy); tl.store(orr+a*3+2, dz)

    @triton.jit
    def _d3_k_qu(q, mu, r, k, kg, aoff, na, dSr, dSi, uq, umu, ur, uk, vq, vmu, vr, vk, ok, M, B: tl.constexpr):
        # gradient of Re sum_m conj(dS_m) D_v D_u S_m w.r.t. k_m  (k-major)
        m = tl.program_id(0)
        if m >= M:
            return
        g = tl.load(kg + m); a0 = tl.load(aoff + g); n = tl.load(na + g)
        kx = tl.load(k+m*3+0); ky = tl.load(k+m*3+1); kz = tl.load(k+m*3+2)
        ukx = tl.load(uk+m*3+0); uky = tl.load(uk+m*3+1); ukz = tl.load(uk+m*3+2)
        vkx = tl.load(vk+m*3+0); vky = tl.load(vk+m*3+1); vkz = tl.load(vk+m*3+2)
        dr_ = tl.load(dSr+m); di_ = tl.load(dSi+m)
        z = tl.zeros((), dtype=ok.dtype.element_ty); dx = z; dy = z; dz = z
        for a in range(0, n, B):
            o = a + tl.arange(0, B); mk = o < n; idx = a0 + o
            rx = tl.load(r+idx*3+0, mask=mk, other=0.); ry = tl.load(r+idx*3+1, mask=mk, other=0.); rz = tl.load(r+idx*3+2, mask=mk, other=0.)
            ux = tl.load(mu+idx*3+0, mask=mk, other=0.); uy = tl.load(mu+idx*3+1, mask=mk, other=0.); uz = tl.load(mu+idx*3+2, mask=mk, other=0.)
            qa = tl.load(q+idx, mask=mk, other=0.)
            uqa = tl.load(uq+idx, mask=mk, other=0.); vqa = tl.load(vq+idx, mask=mk, other=0.)
            umx = tl.load(umu+idx*3+0, mask=mk, other=0.); umy = tl.load(umu+idx*3+1, mask=mk, other=0.); umz = tl.load(umu+idx*3+2, mask=mk, other=0.)
            vmx = tl.load(vmu+idx*3+0, mask=mk, other=0.); vmy = tl.load(vmu+idx*3+1, mask=mk, other=0.); vmz = tl.load(vmu+idx*3+2, mask=mk, other=0.)
            urx = tl.load(ur+idx*3+0, mask=mk, other=0.); ury = tl.load(ur+idx*3+1, mask=mk, other=0.); urz = tl.load(ur+idx*3+2, mask=mk, other=0.)
            vrx = tl.load(vr+idx*3+0, mask=mk, other=0.); vry = tl.load(vr+idx*3+1, mask=mk, other=0.); vrz = tl.load(vr+idx*3+2, mask=mk, other=0.)
            ph = kx*rx+ky*ry+kz*rz; c = tl.cos(ph); s = tl.sin(ph)
            Er = dr_*c + di_*s; Ei = dr_*s - di_*c
            Ar = qa; Ai = kx*ux+ky*uy+kz*uz
            aur = uqa; aui = kx*umx+ky*umy+kz*umz + ukx*ux+uky*uy+ukz*uz
            avr = vqa; avi = kx*vmx+ky*vmy+kz*vmz + vkx*ux+vky*uy+vkz*uz
            phu = kx*urx+ky*ury+kz*urz + ukx*rx+uky*ry+ukz*rz
            phv = kx*vrx+ky*vry+kz*vrz + vkx*rx+vky*ry+vkz*rz
            auvi = ukx*vmx+uky*vmy+ukz*vmz + vkx*umx+vky*umy+vkz*umz
            phuv = ukx*vrx+uky*vry+ukz*vrz + vkx*urx+vky*ury+vkz*urz
            gr_ = -phu*phv; gi_ = phuv
            B2r = -phv*aui - phu*avi + gr_*Ar - gi_*Ai
            B2i = auvi + phv*aur + phu*avr + gr_*Ai + gi_*Ar
            # k^a: total = i vr^a a_u + i ur^a a_v - phv um^a - phu vm^a + i mu^a g - (ur^a phv + phu vr^a) A + i r^a B2
            cf = urx*phv + phu*vrx
            tr = -vrx*aui - urx*avi - phv*umx - phu*vmx - ux*gi_ - cf*Ar - rx*B2i
            ti = vrx*aur + urx*avr + ux*gr_ - cf*Ai + rx*B2r
            dx += tl.sum(tl.where(mk, Er*tr - Ei*ti, z))
            cf = ury*phv + phu*vry
            tr = -vry*aui - ury*avi - phv*umy - phu*vmy - uy*gi_ - cf*Ar - ry*B2i
            ti = vry*aur + ury*avr + uy*gr_ - cf*Ai + ry*B2r
            dy += tl.sum(tl.where(mk, Er*tr - Ei*ti, z))
            cf = urz*phv + phu*vrz
            tr = -vrz*aui - urz*avi - phv*umz - phu*vmz - uz*gi_ - cf*Ar - rz*B2i
            ti = vrz*aur + urz*avr + uz*gr_ - cf*Ai + rz*B2r
            dz += tl.sum(tl.where(mk, Er*tr - Ei*ti, z))
        tl.store(ok+m*3+0, dx); tl.store(ok+m*3+1, dy); tl.store(ok+m*3+2, dz)

    def _bwd(dSr, dSi, q, mu, r, k, kg, aoff, na, ba, koff, nk):
        N = q.shape[0]; M = k.shape[0]
        dq = torch.zeros_like(q); dmu = torch.zeros_like(mu); dr = torch.zeros_like(r); dk = torch.zeros_like(k)
        _bwd_atom_qu[(N,)](q, mu, r, k, ba, koff, nk, dSr, dSi, dq, dmu, dr, N, B=BLK)
        _bwd_k_qu[(M,)](q, mu, r, k, kg, aoff, na, dSr, dSi, dk, M, B=BLK)
        return dq, dmu, dr, dk

    def _ddw(dSr, dSi, gdq, gdmu, gdr, gdk, q, mu, r, k, kg, aoff, na, ba, koff, nk, with_S=True):
        N = q.shape[0]; M = k.shape[0]
        gq = torch.zeros_like(q); gmu = torch.zeros_like(mu); gr = torch.zeros_like(r); gk = torch.zeros_like(k)
        _ddw_atom_qu[(N,)](q, mu, r, k, ba, koff, nk, dSr, dSi, gdq, gdmu, gdr, gdk, gq, gmu, gr, N, B=BLK)
        _ddw_k_qu[(M,)](q, mu, r, k, kg, aoff, na, dSr, dSi, gdq, gdmu, gdr, gdk, gk, M, B=BLK)
        if not with_S:
            return gq, gmu, gr, gk
        gSr = torch.zeros_like(dSr); gSi = torch.zeros_like(dSi)
        _ddw_dS_qu[(M,)](q, mu, r, k, kg, aoff, na, gdq, gdmu, gdr, gdk, gSr, gSi, M, B=BLK)
        return gSr, gSi, gq, gmu, gr, gk

    class _SDdwQU3(torch.autograd.Function):
        """double-backward stage as an autograd Function, so that it can itself be differentiated once more."""

        @staticmethod
        def forward(ctx, dSr, dSi, gdq, gdmu, gdr, gdk, q, mu, r, k, kg, aoff, na, ba, koff, nk):
            gdq = gdq.contiguous(); gdmu = gdmu.contiguous(); gdr = gdr.contiguous(); gdk = gdk.contiguous()
            out = _ddw(dSr, dSi, gdq, gdmu, gdr, gdk, q, mu, r, k, kg, aoff, na, ba, koff, nk)
            ctx.save_for_backward(dSr, dSi, gdq, gdmu, gdr, gdk, q, mu, r, k, kg, aoff, na, ba, koff, nk)
            return out

        @staticmethod
        def backward(ctx, hSr, hSi, hq, hmu, hr, hk):
            dSr, dSi, uq, umu, ur, uk, q, mu, r, k, kg, aoff, na, ba, koff, nk = ctx.saved_tensors
            N = q.shape[0]; M = k.shape[0]
            hSr = hSr.contiguous(); hSi = hSi.contiguous()
            hq = hq.contiguous(); hmu = hmu.contiguous(); hr = hr.contiguous(); hk = hk.contiguous()
            # d/d(gd) : J^T hS  +  H[dS, hx]
            o_u = list(_bwd(hSr, hSi, q, mu, r, k, kg, aoff, na, ba, koff, nk))
            h2 = _ddw(dSr, dSi, hq, hmu, hr, hk, q, mu, r, k, kg, aoff, na, ba, koff, nk, with_S=False)
            for i in range(4):
                o_u[i] = o_u[i] + h2[i]
            # d/d(dS) : D_v D_u S  with u = gd, v = hx
            oSr = torch.zeros_like(dSr); oSi = torch.zeros_like(dSi)
            _d2S_qu[(M,)](q, mu, r, k, kg, aoff, na, uq, umu, ur, uk, hq, hmu, hr, hk, oSr, oSi, M, B=BLK)
            # d/dx : H[hS, u]  +  third derivative contracted with (dS, u, v=hx)
            ox = list(_ddw(hSr, hSi, uq, umu, ur, uk, q, mu, r, k, kg, aoff, na, ba, koff, nk, with_S=False))
            oq3 = torch.zeros_like(q); omu3 = torch.zeros_like(mu); or3 = torch.zeros_like(r); ok3 = torch.zeros_like(k)
            _d3_atom_qu[(N,)](q, mu, r, k, ba, koff, nk, dSr, dSi, uq, umu, ur, uk, hq, hmu, hr, hk, oq3, omu3, or3, N, B=BLK)
            _d3_k_qu[(M,)](q, mu, r, k, kg, aoff, na, dSr, dSi, uq, umu, ur, uk, hq, hmu, hr, hk, ok3, M, B=BLK)
            ox = [ox[0] + oq3, ox[1] + omu3, ox[2] + or3, ox[3] + ok3]
            return (oSr, oSi, o_u[0], o_u[1], o_u[2], o_u[3], ox[0], ox[1], ox[2], ox[3],
                    None, None, None, None, None, None)

    class _SBwdQU3(torch.autograd.Function):
        @staticmethod
        def forward(ctx, dSr, dSi, q, mu, r, k, kg, aoff, na, ba, koff, nk):
            dSr = dSr.contiguous(); dSi = dSi.contiguous()
            out = _bwd(dSr, dSi, q, mu, r, k, kg, aoff, na, ba, koff, nk)
            ctx.save_for_backward(dSr, dSi, q, mu, r, k, kg, aoff, na, ba, koff, nk)
            return out

        @staticmethod
        def backward(ctx, gdq, gdmu, gdr, gdk):
            dSr, dSi, q, mu, r, k, kg, aoff, na, ba, koff, nk = ctx.saved_tensors
            gSr, gSi, gq, gmu, gr, gk = _SDdwQU3.apply(dSr, dSi, gdq, gdmu, gdr, gdk, q, mu, r, k, kg, aoff, na, ba, koff, nk)
            return gSr, gSi, gq, gmu, gr, gk, None, None, None, None, None, None

    class _SFwdQU3(torch.autograd.Function):
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
            dq, dmu, dr, dk = _SBwdQU3.apply(dSr, dSi, q, mu, r, k, kg, aoff, na, ba, koff, nk)
            return dq, dmu, dr, dk, None, None, None, None, None, None

    class TritonEwaldQU3(BatchedEwaldQU):
        """Same energy as TritonEwaldQU; autograd valid through third order."""

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
            Sr, Si = _SFwdQU3.apply(q.contiguous(), mu.contiguous(), r.contiguous(), kvec, kg32, aoff, na32, ba32, koff, nk32)
            kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq
            pot = torch.zeros(n_graphs, device=device, dtype=dtype)
            pot.index_add_(0, kgraph, factors * kfac * (Sr ** 2 + Si ** 2))
            return self._finish(pot, q, mu, batch, n_graphs, cell)
