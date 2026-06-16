"""
Native batched Ewald (reciprocal-space) kernel for SevenNet LES.
"""
from typing import Optional

import torch
import torch.nn as nn


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
            torch.meshgrid(a0, a1, a2, indexing='ij'), dim=-1
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
        validf = valid.to(dtype)

        # --- structure factor S(k) = sum_atoms q * e^{i k.r}, scattered by graph ---
        kvec_atom = kvec[batch]                            # [N,P,3]
        k_dot_r = torch.einsum('nd,npd->np', r, kvec_atom)  # [N,P]
        cos = torch.cos(k_dot_r)                           # [N,P]
        sin = torch.sin(k_dot_r)                           # [N,P]

        qc = q.unsqueeze(2) * cos.unsqueeze(1)             # [N,n_q,P]
        qs = q.unsqueeze(2) * sin.unsqueeze(1)             # [N,n_q,P]
        S_real = torch.zeros(n_graphs, n_q, P, device=device, dtype=dtype)
        S_imag = torch.zeros(n_graphs, n_q, P, device=device, dtype=dtype)
        S_real.index_add_(0, batch, qc)                    # scatter over atoms
        S_imag.index_add_(0, batch, qs)
        S_sq = S_real ** 2 + S_imag ** 2                   # [G,n_q,P]

        # --- assemble potential: (1/V) sum_k factors * exp(-s^2 k^2/2)/k^2 * |S|^2 ---
        k_sq_safe = torch.where(valid, k_sq, torch.ones_like(k_sq))   # avoid /0
        kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq_safe      # [G,P]
        weight = factors.unsqueeze(0) * kfac * validf                # [G,P]
        volume = torch.linalg.det(cell)                              # [G]  signed
        pot = (weight.unsqueeze(1) * S_sq).sum(dim=2) / volume.unsqueeze(1)  # [G,n_q]

        # --- self-interaction removal (matches reference: total q^2 per graph) ---
        if self.remove_self_interaction:
            q_sq_tot = torch.zeros(n_graphs, device=device, dtype=dtype)
            q_sq_tot.index_add_(0, batch, (q ** 2).sum(dim=1))        # [G]
            pot = pot - (q_sq_tot / (self.sigma * (2 * torch.pi) ** 1.5)).unsqueeze(1)

        pot = pot * self.norm_factor                                 # [G,n_q]
        return pot.sum(dim=1)                                        # [G]
