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
        volume = torch.linalg.det(cell)                              # [G]  signed
        pot = (weight.unsqueeze(1) * S_sq).sum(dim=2) / volume.unsqueeze(1)  # [G,n_q]

        # --- self-interaction removal (matches reference: total q^2 per graph) ---
        if self.remove_self_interaction:
            q_sq_tot = torch.zeros(n_graphs, device=device, dtype=dtype)
            q_sq_tot.index_add_(0, batch, (q ** 2).sum(dim=1))        # [G]
            pot = pot - (q_sq_tot / (self.sigma * (2 * torch.pi) ** 1.5)).unsqueeze(1)

        pot = pot * self.norm_factor                                 # [G,n_q]
        return pot.sum(dim=1)                                        # [G]


class HybridBatchedEwald(nn.Module):
    """
    Adaptive wrapper: keeps small/normal cells together as the bulk and 
    recursively peels the largest into further groups
    """

    def __init__(
        self,
        dl: float = 2.0,
        sigma: float = 1.0,
        remove_self_interaction: bool = True,
        norm_factor: float = 90.4756,
        mem_budget: Optional[int] = None,
        safety: float = 0.8,
    ):
        super().__init__()
        self.dl = dl
        self.mem_budget = mem_budget
        self.safety = safety
        self.core = BatchedEwald(dl, sigma, remove_self_interaction, norm_factor)

    def _grid_Pprime(self, cell: torch.Tensor) -> int:
        device, dtype = cell.device, cell.dtype
        G = self.core.twopi * torch.linalg.inv(cell).transpose(1, 2)
        Nk = torch.clamp((torch.norm(cell, dim=2) / self.dl).to(torch.int64), min=1)
        Nm = Nk.max(0).values
        ax = [torch.arange(-int(Nm[i]), int(Nm[i]) + 1, device=device) for i in range(3)]
        nvec = torch.stack(torch.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3).to(dtype)
        ksq = (torch.einsum('pd,gde->gpe', nvec, G) ** 2).sum(-1)
        box = (nvec.abs().unsqueeze(0) <= Nk.unsqueeze(1)).all(2)
        sph = (ksq > 0) & (ksq <= self.core.k_sq_max)
        nz = (nvec != 0).to(torch.int64)
        sign = torch.gather(nvec, 1, torch.argmax(nz, 1).unsqueeze(1)).squeeze(1)
        hemi = (sign > 0) | (nvec == 0).all(1)
        return int((box & sph & hemi.unsqueeze(0)).any(0).sum())

    def _budget(self, device: torch.device) -> float:
        if self.mem_budget is not None:
            return float(self.mem_budget)
        if device.type == 'cuda':
            free, _ = torch.cuda.mem_get_info(device)
            return self.safety * free
        return float('inf')

    def forward(
        self,
        q: torch.Tensor,
        r: torch.Tensor,
        cell: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if q.dim() == 1:
            q = q.unsqueeze(1)
        N, n_q = r.shape[0], q.shape[1]
        device, dtype = r.device, r.dtype
        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=device)
        n_graphs = cell.shape[0]

        # gate: estimate peak from N * P' (no [N,P] tensor built)
        Pp = self._grid_Pprime(cell)
        planes = 6 + 2 * n_q
        autograd = 2 if torch.is_grad_enabled() else 1
        const = planes * r.element_size() * autograd
        budget = self._budget(device)
        if N * Pp * const <= budget:
            return self.core(q=q, r=r, cell=cell, batch=batch)

        # fallback: tier cells into budget-fitting groups by recursively
        Nk = torch.clamp((torch.norm(cell, dim=2) / self.dl).to(torch.int64), min=1)
        natoms = torch.bincount(batch, minlength=n_graphs)
        dense = (2 * Nk + 1).prod(1)
        ratio = Pp / float((2 * Nk.max(0).values + 1).prod())

        def fits(mask: torch.Tensor) -> bool:
            gi = mask.nonzero(as_tuple=True)[0]
            p = float((2 * Nk[gi].max(0).values + 1).prod()) * ratio
            return float(natoms[gi].sum()) * p * const <= budget

        gid = torch.full((n_graphs,), -1, dtype=torch.long, device=device)
        remaining = torch.ones(n_graphs, dtype=torch.bool, device=device)
        g = 0
        while remaining.any():
            in_bulk = remaining.clone()
            rem = remaining.nonzero(as_tuple=True)[0]
            order = rem[torch.argsort(dense[rem], descending=True)].tolist()
            oi = 0
            while int(in_bulk.sum()) > 1 and not fits(in_bulk):
                while oi < len(order) and not in_bulk[order[oi]]:
                    oi += 1
                if oi >= len(order):
                    break
                in_bulk[order[oi]] = False
                oi += 1
            gid[in_bulk] = g
            remaining = remaining & ~in_bulk
            g += 1
        n_groups = g

        atom_gid = gid[batch]
        out_idx, out_e = [], []
        for grp in range(n_groups):
            graphs = (gid == grp).nonzero(as_tuple=True)[0]
            amask = atom_gid == grp
            remap = torch.full((n_graphs,), -1, dtype=torch.long, device=device)
            remap[graphs] = torch.arange(graphs.shape[0], device=device)
            e_sub = self.core(q[amask], r[amask], cell[graphs], remap[batch[amask]])
            out_idx.append(graphs)
            out_e.append(e_sub)
        out = torch.zeros(n_graphs, device=device, dtype=dtype)
        return out.index_copy(0, torch.cat(out_idx), torch.cat(out_e))
