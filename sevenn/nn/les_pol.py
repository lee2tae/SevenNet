"""
Polarizable LES, one-shot (non-self-consistent) linear response in the style of Cheng et al. (arXiv 2605.05746):
  E_i      = -dE_ewald/dmu_i (q, mu)        field at atom i from the permanent latent charges and dipoles
  mu_ind_i = (alpha_i / k_e) E_i             induced dipole (alpha in Angstrom^3), no induced-induced coupling
  E_pol    = -1/2 sum_i (alpha_i / k_e) |E_i|^2
  P        = sum_i (q_i r_i + mu_i + mu_ind_i)  ->  Z* = dP/dr
Everything runs on TritonEwaldQU3 (third-order autograd), so forces, Z* and their training gradients are plain autograd.
"""
import os
from typing import Optional

import torch
import torch.nn as nn

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType

from .ewald_qu import _HAS_TRITON, BatchedEwaldQU
from .les_qu import BornEffectiveChargeQU

K_E = 14.399645  # eV*Angstrom/e^2


def _make_ewald(les_args: Optional[dict], ewald_type: str = 'triton'):
    les_args = les_args or {}
    kw = dict(dl=les_args.get('dl', 2.0), sigma=les_args.get('sigma', 1.0),
              remove_self_interaction=les_args.get('remove_self_interaction', True))
    if ewald_type == 'triton':
        if not _HAS_TRITON:
            raise ImportError("ewald_type='triton' requires the triton package")
        from .ewald_qu3 import TritonEwaldQU3
        return TritonEwaldQU3(**kw)
    return BatchedEwaldQU(**kw)


def field(ewald, q, mu, r, cell, batch):
    """E_i = -dE_ewald/dmu_i  (N, 3), eV/(e*Angstrom); differentiable (create_graph)."""
    mt = mu if mu.requires_grad else mu.detach().requires_grad_(True)
    e = ewald(q, mt, r, cell, batch).sum()
    (g,) = torch.autograd.grad(e, mt, create_graph=True)
    return -g


class PolarizabilityReadout(nn.Module):
    """raw per-atom feature -> alpha_i = softplus(x + init_shift) >= 0 (Angstrom^3); no upper bound."""

    def __init__(self, data_key_in: str = KEY.LES_ALPHA_RAW, data_key_out: str = KEY.LES_ALPHA, init_shift: float = -4.0):
        super().__init__()
        self.key_input = data_key_in
        self.key_output = data_key_out
        self.init_shift = init_shift

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_input].reshape(-1)
        data[self.key_output] = torch.nn.functional.softplus(x + self.init_shift)
        return data


class LatentEwaldSumQUPol(nn.Module):
    """E_LR = E_ewald(q, mu) - 1/2 sum_i a_i |E_i|^2,  a_i = alpha_i / k_e.  3D cells only."""

    def __init__(self, les_args: Optional[dict] = None, data_key_q: str = KEY.LES_Q, data_key_mu: str = KEY.LES_MU,
                 data_key_alpha: str = KEY.LES_ALPHA, data_key_out: str = KEY.LR_ENERGY, ewald_type: str = 'triton'):
        super().__init__()
        self.key_q, self.key_mu, self.key_alpha, self.key_output = data_key_q, data_key_mu, data_key_alpha, data_key_out
        self.ewald = _make_ewald(les_args, ewald_type)
        self._is_batch_data = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        q = data[self.key_q]
        if q.dim() == 2:
            q = q.sum(dim=1)
        mu, pos = data[self.key_mu], data[KEY.POS]
        a = data[self.key_alpha].reshape(-1) / K_E
        batch = data[KEY.BATCH].long() if self._is_batch_data else torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        cell = data[KEY.CELL].view(-1, 3, 3)
        if bool((torch.linalg.det(cell).abs() <= 1e-6).any()):
            raise RuntimeError('LatentEwaldSumQUPol supports 3D-periodic cells only')
        E = field(self.ewald, q, mu, pos, cell, batch)
        mu_ind = a.unsqueeze(1) * E
        e_lr = self.ewald(q=q, mu=mu, r=pos, cell=cell, batch=batch)
        pol = torch.zeros(cell.shape[0], device=pos.device, dtype=pos.dtype)
        pol.index_add_(0, batch, -0.5 * a * (E * E).sum(1))
        e_lr = e_lr + pol
        data[KEY.LES_MU_IND] = mu_ind
        data[self.key_output] = e_lr if self._is_batch_data else e_lr.squeeze()
        d = os.environ.get('SEVENN_POL_DUMP')
        if d and not torch.isfinite(e_lr).all():
            _dump_batch(d, dict(q=q, mu=mu, alpha=a * K_E, pos=pos, cell=cell, batch=batch, mu_ind=mu_ind, e_lr=e_lr))
        return data


class PolarizableBornEffectiveChargeQU(BornEffectiveChargeQU):
    """BEC head with one-shot induced dipoles from the eps-scaled latent sources."""

    def __init__(self, les_args: Optional[dict] = None, data_key_alpha: str = KEY.LES_ALPHA, ewald_type: str = 'triton', **kwargs):
        super().__init__(**kwargs)
        self.data_key_alpha = data_key_alpha
        self.ewald = _make_ewald(les_args, ewald_type)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        r = data[self.data_key_pos]
        alpha = data[self.data_key_alpha].reshape(-1)
        if (
            self.compute_bec
            and not self.compute_dipole
            and KEY.LES_BEC_REF in data
            and bool(torch.isnan(data[KEY.LES_BEC_REF]).all())
        ):
            zeros = torch.zeros(r.shape[0], 3, 3, device=r.device, dtype=r.dtype)
            zeros = zeros + 0.0 * data[self.data_key_mu].sum() + 0.0 * alpha.sum()
            if self.epsilon_mode != 'fixed':
                zeros = zeros + 0.0 * data[self.data_key_epsilon].sum()
            data[self.data_key_out] = zeros
            return data

        q = data[self.data_key_q]
        if q.dim() == 1:
            q = q.unsqueeze(1)
        q = q.sum(dim=1, keepdim=True)
        mu = data[self.data_key_mu]

        if self._is_batch_data:
            batch = data[KEY.BATCH].long()
            n_graphs = int(batch.max().item()) + 1
        else:
            batch = torch.zeros(r.shape[0], dtype=torch.long, device=r.device)
            n_graphs = 1
        cell = data[self.data_key_cell].view(-1, 3, 3)
        if bool((torch.linalg.det(cell).abs() <= 1e-6).any()):
            raise RuntimeError('PolarizableBornEffectiveChargeQU supports 3D-periodic cells only')

        if self.epsilon_mode == 'learned_atomic':
            eps_atom = data[self.data_key_epsilon].reshape(-1, 1)
            s = eps_atom.sqrt()
            q_scaled = q * s
            mu = mu * s
            if self.remove_mean == 'q_projection':
                S = torch.zeros(n_graphs, 1, device=q.device, dtype=q.dtype)
                D = torch.zeros(n_graphs, 1, device=q.device, dtype=q.dtype)
                S.index_add_(0, batch, q_scaled)
                D.index_add_(0, batch, q * q)
                q = q_scaled - (q * q) * (S / D.clamp(min=1e-8))[batch]
            else:
                q = q_scaled
            factors = torch.ones(n_graphs, device=r.device, dtype=r.dtype)
        elif self.epsilon_mode == 'learned':
            factors = data[self.data_key_epsilon].reshape(-1) ** 0.5
        else:
            factors = torch.full((n_graphs,), self.normalization_factor, device=r.device, dtype=r.dtype)
        if self.remove_mean == 'uniform':
            sum_q = torch.zeros(n_graphs, 1, device=q.device, dtype=q.dtype)
            cnt = torch.zeros_like(sum_q)
            sum_q.index_add_(0, batch, q)
            cnt.index_add_(0, batch, torch.ones_like(q))
            q = q - (sum_q / cnt.clamp(min=1.0))[batch]

        fb = factors[batch].unsqueeze(1)
        q_src, mu_src = q * fb, mu * fb
        E = field(self.ewald, q_src.reshape(-1), mu_src, r, cell, batch)
        mu_ind = (alpha / K_E).unsqueeze(1) * E
        data[KEY.LES_MU_IND] = mu_ind
        mu_tot = mu_src + mu_ind

        p, phases, M = self._polarization(q, r, cell, batch, n_graphs, factors)
        mu_frac = torch.einsum('nd,nde->ne', mu_tot, torch.linalg.inv(cell)[batch])
        pm = torch.zeros(n_graphs, 3, device=r.device, dtype=r.dtype)
        pm.index_add_(0, batch, mu_frac)

        if self.compute_dipole:
            dip = torch.zeros(n_graphs, 3, device=r.device, dtype=r.dtype)
            dip.index_add_(0, batch, q_src * r + mu_tot)
            if self.output_index is not None:
                dip = dip[:, self.output_index]
            data[self.data_key_dipole] = dip if self._is_batch_data else dip.squeeze(0)

        if self.compute_bec:
            z = None
            if self._use_batched_grad:
                try:
                    z = self._bec_grad_batched(p, r, phases) + self._mu_grad_batched(pm, r)
                except torch.OutOfMemoryError:
                    raise
                except RuntimeError:
                    self._use_batched_grad = False
            if z is None:
                zq = (self._grad(p, r).transpose(1, 2).contiguous() * phases.unsqueeze(2).conj()).real
                z = zq + self._grad_real(pm, r).transpose(1, 2).contiguous()
            result = torch.einsum('nea,neb->nab', M[batch], z)
            if self.output_index is not None:
                result = result[:, self.output_index]
            result = self._asr_project(result, batch, n_graphs)
            data[self.data_key_out] = result
        return data


_DUMPED = [0]


def _dump_batch(d, tensors):
    """debug aid (SEVENN_POL_DUMP=<dir>): save the first non-finite batches seen by the polarizable energy."""
    if _DUMPED[0] >= 3:
        return
    os.makedirs(d, exist_ok=True)
    torch.save({k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in tensors.items()},
               os.path.join(d, f'nonfinite_batch_{os.getpid()}_{_DUMPED[0]}.pt'))
    _DUMPED[0] += 1
