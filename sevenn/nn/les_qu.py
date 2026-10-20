"""
LES with latent atomic dipoles (q + mu). 3D-periodic cells only.
  LatentDipoleReadout : node features (1o part) -> mu_i (N, 3)
  LatentEwaldSumQU    : E_LR = Ewald(q, mu)
  BornEffectiveChargeQU: P = sum_i (q_i r_i + mu_i), Z*_i = dP/dr_i
"""
from typing import Optional

import torch
import torch.nn as nn
from e3nn.o3 import Irreps

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType

from .ewald_qu import BatchedEwaldQU, _HAS_TRITON
from .les import BornEffectiveCharge
from .linear import IrrepsLinear


class FeatureTap(nn.Module):
    """Stores a copy of data[key_in] under key_out (pass-through)."""

    def __init__(self, data_key_in: str = KEY.NODE_FEATURE, data_key_out: str = KEY.LES_VEC_FEATURE):
        super().__init__()
        self.key_input = data_key_in
        self.key_output = data_key_out

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_output] = data[self.key_input]
        return data


class LatentDipoleReadout(nn.Module):
    """node features -> mu_i (N, 3). hidden_channels=[H, ...]: gated equivariant hidden layers
    (IrrepsLinear -> Hx0e + Hx1o gated by SiLU scalars) before the final linear to 1x1o."""

    def __init__(
        self,
        irreps_in: Irreps,
        data_key_in: str = KEY.NODE_FEATURE,
        data_key_out: str = KEY.LES_MU,
        hidden_channels: Optional[list] = None,
        zero_init: bool = False,
    ):
        super().__init__()
        if not any(ir.l == 1 and ir.p == -1 for _, ir in Irreps(irreps_in)):
            raise ValueError(
                f'LatentDipoleReadout needs 1o node features, got {irreps_in}'
            )
        self.key_output = data_key_out
        hidden = list(hidden_channels or [])
        key = data_key_in
        layers = []
        irreps = Irreps(irreps_in)
        for i, h in enumerate(hidden):
            from .equivariant_gate import EquivariantGate
            k_out = f'{data_key_out}_h{i}'
            gate = EquivariantGate(Irreps(f'{h}x0e+{h}x1o'), {'e': torch.nn.functional.silu}, {'e': torch.nn.functional.silu}, data_key_x=k_out)
            lin = IrrepsLinear(irreps_in=irreps, irreps_out=gate.get_gate_irreps_in(), data_key_in=key, data_key_out=k_out, biases=False)
            layers += [lin, gate]
            irreps, key = Irreps(f'{h}x0e+{h}x1o'), k_out
        self.linear = IrrepsLinear(irreps_in=irreps, irreps_out=Irreps('1x1o'), data_key_in=key, data_key_out=data_key_out, biases=False)
        self.hidden = nn.ModuleList(layers)
        self._zero_init = zero_init

    @property
    def layer_instantiated(self) -> bool:
        return self.linear.layer_instantiated and all(
            m.layer_instantiated for m in self.hidden if isinstance(m, IrrepsLinear)
        )

    def instantiate(self) -> None:
        for m in self.hidden:
            if isinstance(m, IrrepsLinear):
                m.instantiate()
        self.linear.instantiate()
        if self._zero_init:
            nn.init.zeros_(self.linear.linear.weight)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        for m in self.hidden:
            data = m(data)
        return self.linear(data)


class LatentEwaldSumQU(nn.Module):
    def __init__(
        self,
        les_args: Optional[dict] = None,
        data_key_q: str = KEY.LES_Q,
        data_key_mu: str = KEY.LES_MU,
        data_key_out: str = KEY.LR_ENERGY,
        ewald_type: str = 'triton',
    ):
        super().__init__()
        les_args = les_args or {}
        self.key_q = data_key_q
        self.key_mu = data_key_mu
        self.key_output = data_key_out
        kw = dict(
            dl=les_args.get('dl', 2.0),
            sigma=les_args.get('sigma', 1.0),
            remove_self_interaction=les_args.get('remove_self_interaction', True),
        )
        if ewald_type == 'triton':
            if not _HAS_TRITON:
                raise ImportError("ewald_type='triton' requires the triton package")
            from .ewald_qu import TritonEwaldQU
            self.ewald = TritonEwaldQU(**kw)
        else:
            self.ewald = BatchedEwaldQU(**kw)
        self._is_batch_data = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        q = data[self.key_q]
        if q.dim() == 2:
            q = q.sum(dim=1)
        mu = data[self.key_mu]
        pos = data[KEY.POS]
        if self._is_batch_data:
            batch = data[KEY.BATCH].long()
        else:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        cell = data[KEY.CELL].view(-1, 3, 3)
        if bool((torch.linalg.det(cell).abs() <= 1e-6).any()):
            raise RuntimeError('LatentEwaldSumQU supports 3D-periodic cells only')
        e_lr = self.ewald(q=q, mu=mu, r=pos, cell=cell, batch=batch)
        data[self.key_output] = e_lr if self._is_batch_data else e_lr.squeeze()
        return data


class BornEffectiveChargeQU(BornEffectiveCharge):
    def __init__(self, data_key_mu: str = KEY.LES_MU, **kwargs):
        super().__init__(**kwargs)
        self.data_key_mu = data_key_mu

    def _mu_grad_batched(self, pm: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        # pm (G, 3) real lattice-coordinate dipole sum -> (N, e, b)
        n_graphs = pm.shape[0]
        Y = pm.permute(1, 0)                                  # (3, G)
        cot = torch.eye(3, device=r.device, dtype=r.dtype)
        cot = cot.unsqueeze(-1).expand(3, 3, n_graphs)
        (g,) = torch.autograd.grad(
            [Y], [r], grad_outputs=[cot], retain_graph=True,
            create_graph=self.training, is_grads_batched=True, allow_unused=True,
        )
        if g is None:
            return torch.zeros(r.shape[0], 3, 3, device=r.device, dtype=r.dtype)
        return g.permute(1, 0, 2)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        r = data[self.data_key_pos]
        if (
            self.compute_bec
            and not self.compute_dipole
            and KEY.LES_BEC_REF in data
            and bool(torch.isnan(data[KEY.LES_BEC_REF]).all())
        ):
            zeros = torch.zeros(r.shape[0], 3, 3, device=r.device, dtype=r.dtype)
            zeros = zeros + 0.0 * data[self.data_key_mu].sum()
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
            raise RuntimeError('BornEffectiveChargeQU supports 3D-periodic cells only')

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
            factors = torch.full(
                (n_graphs,), self.normalization_factor, device=r.device, dtype=r.dtype
            )
        if self.remove_mean == 'uniform':
            sum_q = torch.zeros(n_graphs, 1, device=q.device, dtype=q.dtype)
            cnt = torch.zeros_like(sum_q)
            sum_q.index_add_(0, batch, q)
            cnt.index_add_(0, batch, torch.ones_like(q))
            q = q - (sum_q / cnt.clamp(min=1.0))[batch]

        p, phases, M = self._polarization(q, r, cell, batch, n_graphs, factors)
        # dipole part in lattice coordinates (no Berry phase needed)
        mu_frac = torch.einsum('nd,nde->ne', mu, torch.linalg.inv(cell)[batch])
        pm = torch.zeros(n_graphs, 3, device=r.device, dtype=r.dtype)
        pm.index_add_(0, batch, mu_frac)
        pm = pm * factors.unsqueeze(1)

        if self.compute_dipole:
            dip = torch.zeros(n_graphs, 3, device=r.device, dtype=r.dtype)
            dip.index_add_(0, batch, q * r + mu)
            dip = dip * factors.unsqueeze(1)
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
                    self._use_batched_grad = False  # e.g. TorchScript modules: no vmap batching rule
            if z is None:
                zq = (self._grad(p, r).transpose(1, 2).contiguous() * phases.unsqueeze(2).conj()).real
                z = zq + self._grad_real(pm, r).transpose(1, 2).contiguous()
            result = torch.einsum('nea,neb->nab', M[batch], z)
            if self.output_index is not None:
                result = result[:, self.output_index]
            result = self._asr_project(result, batch, n_graphs)
            data[self.data_key_out] = result
        return data
