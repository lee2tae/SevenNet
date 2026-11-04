"""
Closed-form local-field (depolarization) screening of the LES BEC; les_config: local_field: True | {sigma, h, c, gamma,
learn_gamma}. Requires epsilon_mode: learned_atomic (the per-atom eps defines the dielectric map).

Z*_i <- L_i Z*_i,   L_i = I + P_long[u~ - u_bar](r_i) / u_bar
u_j = eps_j^(-gamma);  u~(r) = (sum_j u_j G(r - r_j) + c) / (sum_j G(r - r_j) + c) on an FFT grid (Gaussian width sigma),
so uniform eps gives L = I for any density and empty space gives u = 1 (vacuum);  P_long[f](r) = sum_{k!=0} k^k^ f_k e^{ikr}.
Exact for laminates of any orientation (series/parallel capacitor: L = I + n n (u / u_bar - 1)), the cell-averaged field
equals the applied field, and there is no iteration: one structure factor + one FFT per frame. Only 3D-periodic cells get
L != I. The base BEC's acoustic sum rule is kept by removing the per-frame mean of (L_i - I) Z*_i.
"""
import itertools
import math

import torch
import torch.nn as nn

import sevenn._keys as KEY

LES_LOCAL_FIELD = 'les_local_field'   # (N, 3, 3) L_i


def local_field_tensor(pos, cell, u, sigma=1.25, h=None, c=1e-4, chunk=2 ** 24):
    """L (N, 3, 3) for one 3D-periodic frame in pos.dtype; pos/cell constant, u (N,) carries gradients."""
    h = h or sigma / 2
    dt, dev = pos.dtype, pos.device
    ct = torch.complex128 if dt == torch.float64 else torch.complex64
    pos, u = pos.detach(), u.to(dt)
    cc = cell.detach().double().cpu()                     # 3x3 algebra on CPU: no solver launch / sync per frame
    n = [max(4, math.ceil(float(x) / h)) for x in torch.linalg.norm(cc, dim=1)]
    ng = n[0] * n[1] * n[2]
    vol = float(torch.linalg.det(cc).abs())
    m = torch.stack(torch.meshgrid(*[torch.fft.fftfreq(ni, 1.0 / ni, device=dev, dtype=dt) for ni in n],
                                   indexing='ij'), -1).reshape(-1, 3)
    k = m @ (2 * math.pi * torch.linalg.inv(cc).T).to(dev, dt)
    k2 = (k * k).sum(1)
    g = torch.exp(-0.5 * sigma * sigma * k2)
    step = max(1, chunk // ng)
    Su = torch.zeros(ng, dtype=ct, device=dev)
    S1 = torch.zeros_like(Su)
    for i in range(0, pos.shape[0], step):
        E = torch.exp(-1j * (pos[i:i + step] @ k.T))
        Su = Su + u[i:i + step].to(ct) @ E
        S1 = S1 + E.sum(0)
    A = torch.fft.ifftn((Su * g).reshape(n)).real * (ng / vol)
    B = torch.fft.ifftn((S1 * g).reshape(n)).real * (ng / vol)
    ut = (A + c) / (B + c)
    ub = ut.mean()
    F = torch.fft.fftn(ut - ub).reshape(-1) / ng
    kk = k[:, :, None] * k[:, None, :] / torch.where(k2 > 0, k2, torch.ones_like(k2))[:, None, None]
    P = []
    for i in range(0, pos.shape[0], step):
        E = torch.exp(-1j * (pos[i:i + step] @ k.T))
        P.append(torch.einsum('nk,kab->nab', (E.conj() * F).real, kk))
    return torch.eye(3, dtype=dt, device=dev) + torch.cat(P) / ub


def local_field_batched(pos, cells, u, counts, sigma=1.25, h=None, c=1e-4, max_elems=2 ** 25):
    """L for contiguous 3D-periodic frames (atom counts `counts`, cells (G, 3, 3)); same per-frame grids as
    local_field_tensor, but one padded set of batched kernels for the whole batch (launch-bound regime)."""
    h = h or sigma / 2
    dt, dev = pos.dtype, pos.device
    ct = torch.complex128 if dt == torch.float64 else torch.complex64
    cc = cells.detach().double().cpu()
    G, nmax = len(counts), max(counts)
    shapes = [[max(4, math.ceil(float(x) / h)) for x in torch.linalg.norm(cc[g], dim=1)] for g in range(G)]
    ngs = [a * b * c_ for a, b, c_ in shapes]
    kmax = max(ngs)
    if G * nmax * kmax > max_elems:
        starts = [0] + list(itertools.accumulate(counts))[:-1]
        return torch.cat([local_field_tensor(pos[s:s + n], cells[g], u[s:s + n], sigma, h, c)
                          for g, (s, n) in enumerate(zip(starts, counts))])
    dims = torch.tensor(shapes, device=dev)                                                    # (G, 3)
    idx = torch.arange(kmax, device=dev)[None, :, None]
    n2, n12 = dims[:, None, 2:3], dims[:, None, 1:2] * dims[:, None, 2:3]
    i = torch.cat([idx // n12, idx % n12 // n2, idx % n2], -1)                                 # (G, K, 3) C-order index
    m = torch.where(i < (dims[:, None] + 1) // 2, i, i - dims[:, None])                        # fftfreq order
    m = m * (idx[..., 0] < torch.tensor(ngs, device=dev)[:, None])[..., None]                  # pads -> k = 0
    k = torch.bmm(m.to(dt), (2 * math.pi * torch.linalg.inv(cc).transpose(1, 2)).to(dev, dt))  # (G, K, 3)
    vol = torch.linalg.det(cc).abs().tolist()
    cnt = torch.tensor(counts, device=dev)
    gid = torch.repeat_interleave(torch.arange(G, device=dev), cnt)
    loc = torch.arange(pos.shape[0], device=dev) - torch.repeat_interleave(cnt.cumsum(0) - cnt, cnt)
    pp = pos.detach().new_zeros(G, nmax, 3).index_put((gid, loc), pos.detach())
    up = u.to(dt).new_zeros(G, nmax).index_put((gid, loc), u.to(dt))
    am = pos.new_zeros(G, nmax).index_put((gid, loc), torch.ones_like(u, dtype=dt))
    E = torch.exp(-1j * torch.bmm(pp, k.transpose(1, 2))) * am[:, :, None]                      # (G, nmax, K)
    Su = torch.bmm(up[:, None, :].to(ct), E).squeeze(1)
    S1 = E.sum(1)
    g_k = torch.exp(-0.5 * sigma * sigma * (k * k).sum(-1))
    Fs, ubs = [], []
    for g, n in enumerate(shapes):
        ng = ngs[g]
        A = torch.fft.ifftn((Su[g, :ng] * g_k[g, :ng]).reshape(n)).real * (ng / vol[g])
        B = torch.fft.ifftn((S1[g, :ng] * g_k[g, :ng]).reshape(n)).real * (ng / vol[g])
        ut = (A + c) / (B + c)
        ub = ut.mean()
        Fs.append(nn.functional.pad(torch.fft.fftn(ut - ub).reshape(-1) / ng, (0, kmax - ng)))
        ubs.append(ub)
    F = torch.stack(Fs)
    k2 = (k * k).sum(-1)
    kk = (k[..., :, None] * k[..., None, :] / torch.where(k2 > 0, k2, torch.ones_like(k2))[..., None, None]).reshape(G, kmax, 9)
    P = torch.bmm((E.conj() * F[:, None, :]).real, kk) / torch.stack(ubs)[:, None, None]       # (G, nmax, 9)
    return torch.eye(3, dtype=dt, device=dev) + P[gid, loc].reshape(-1, 3, 3)


def localfield_bec_cls(base):
    class LocalFieldBEC(base):
        def __init__(self, lf_sigma=1.25, lf_h=None, lf_c=1e-4, lf_gamma=1.0, lf_learn_gamma=True, **kwargs):
            super().__init__(**kwargs)
            if self.epsilon_mode != 'learned_atomic':
                raise ValueError("local_field requires epsilon_mode: 'learned_atomic'")
            if self.output_index is not None:
                raise ValueError('local_field does not support bec_output_index')
            self.lf_sigma, self.lf_h, self.lf_c = lf_sigma, lf_h, lf_c
            raw = torch.tensor(math.log(math.expm1(lf_gamma)))       # gamma = softplus(raw) > 0
            if lf_learn_gamma:
                self.lf_gamma_raw = nn.Parameter(raw)
            else:
                self.register_buffer('lf_gamma_raw', raw)

        def forward(self, data):
            data = super().forward(data)
            if not self.compute_bec or self.data_key_out not in data:
                return data
            z = data[self.data_key_out]
            gamma = nn.functional.softplus(self.lf_gamma_raw)
            if KEY.LES_BEC_REF in data and bool(torch.isnan(data[KEY.LES_BEC_REF]).all()):
                data[self.data_key_out] = z + 0.0 * gamma               # EFS-only batch: keep gamma in the graph (DDP)
                return data
            pos = data[KEY.POS]
            n_atoms = pos.shape[0]
            batch = data[KEY.BATCH].long() if self._is_batch_data else torch.zeros(n_atoms, dtype=torch.long,
                                                                                 device=pos.device)
            cell = data[KEY.CELL].view(-1, 3, 3)
            u = data[self.data_key_epsilon].reshape(-1).pow(-gamma)
            pbc3 = (torch.linalg.det(cell.detach().double().cpu()).abs() >= 1e-6).tolist()
            counts = torch.bincount(batch, minlength=cell.shape[0]).tolist()             # graphs are contiguous
            L = torch.eye(3, dtype=z.dtype, device=z.device).repeat(n_atoms, 1, 1)
            if any(pbc3):
                sel = torch.tensor(pbc3, device=pos.device)[batch]
                Lp = local_field_batched(pos[sel], cell[torch.tensor(pbc3, device=pos.device)], u[sel],
                                         [n for n, p in zip(counts, pbc3) if p], self.lf_sigma, self.lf_h, self.lf_c)
                L = L.index_put((sel.nonzero().flatten(),), Lp.to(z.dtype)) if not all(pbc3) else Lp.to(z.dtype)
            d = torch.bmm(L, z) - z
            tot = torch.zeros((cell.shape[0], 3, 3), device=z.device, dtype=z.dtype).index_add_(0, batch, d)
            cnt = torch.bincount(batch, minlength=cell.shape[0]).to(z.dtype)
            data[self.data_key_out] = z + d - (tot / cnt[:, None, None])[batch]
            data[LES_LOCAL_FIELD] = L.detach()
            return data

    return LocalFieldBEC
