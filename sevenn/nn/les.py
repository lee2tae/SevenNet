"""
LES (Latent Ewald Summation) modules for SevenNet.

Architecture:
  NODE_FEATURE (last conv. layer, all-scalar)
      │
      ├─→ [LatentChargeReadout] → LES_Q (N_atoms, n_charges)
      │
      └─→ [init_feature_reduce] → ATOMIC_ENERGY → [AtomReduce] → SR_ENERGY
                                                                       │
  LES_Q ──→ [LatentEwaldSum] ──→ LR_ENERGY ──→ [AddLREnergy] ─────────┘
                                                       │
                                              PRED_TOTAL_ENERGY
                                                       │
                                           [ForceStressOutput]

EdgePreprocess (first layer) applies an affine strain to pos and cell and
computes EDGE_VEC from the strained pos, connecting all three to the _strain
leaf.  ForceStressOutput then recovers:
  Forces: -d(E_total)/d(strained_pos)   SR + q-path LR + direct Ewald
  Stress: -d(E_total)/d(_strain)        SR virial + LR positional + LR cell

References:
  - LES library: https://github.com/ChengUCB/les
  - NequIP-LES:  https://github.com/ChengUCB/nequip-les
"""
from typing import Optional

import torch
import torch.nn as nn
from e3nn.o3 import Irreps

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType

from .linear import IrrepsLinear


class LatentChargeReadout(nn.Module):
    """
    Projects node features to per-atom latent charges.

    Architecture (controlled by ``hidden_channels``):
        hidden_channels=[] (default):
            irreps_in ──[IrrepsLinear]──► (N, n_charges)
        hidden_channels=[H, ...]:
            irreps_in ──[IrrepsLinear]──► (N, H) ──[SiLU + nn.Linear]──► (N, n_charges)

    The first layer is SevenNet's IrrepsLinear (a thin wrapper around
    e3nn.o3.Linear that operates on AtomGraphData dicts). Modality dependence
    flows in through the upstream conv stack's modality-aware features; this
    layer does not concatenate the modality one-hot itself.

    Args:
        irreps_in:       e3nn irreps of the input node features.
        n_charges:       number of latent charge channels per atom (default 1).
                         With n_charges > 1 the Ewald energy is the sum of
                         n_charges independent Coulomb interactions, one per
                         channel: E_LR = Σ_α E_Coulomb(q^α).
        hidden_channels: hidden layer widths, e.g. [128] for one hidden layer.
        zero_init:       zero-initialise all weights so E_LR = 0 at init.
                         Useful for transparent-wrapper tests; not for training.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        data_key_in: str = KEY.NODE_FEATURE,
        data_key_out: str = KEY.LES_Q,
        n_charges: int = 1,
        hidden_channels: Optional[list] = None,
        zero_init: bool = False,
    ):
        super().__init__()
        self.key_input = data_key_in
        self.key_output = data_key_out
        self.n_charges = n_charges

        if hidden_channels is None:
            hidden_channels = []
        self._hidden_channels = list(hidden_channels)

        first_out = hidden_channels[0] if hidden_channels else n_charges
        # Intermediate key only needed when a scalar MLP follows.
        self._intermediate_key = (
            f'{data_key_out}_intermediate' if hidden_channels else data_key_out
        )
        self.first_linear = IrrepsLinear(
            irreps_in=irreps_in,
            irreps_out=Irreps(f'{first_out}x0e'),
            data_key_in=data_key_in,
            data_key_out=self._intermediate_key,
            biases=False,
        )

        scalar_layers: list[nn.Module] = []
        if hidden_channels:
            dims = hidden_channels + [n_charges]
            for i in range(len(dims) - 1):
                scalar_layers.append(nn.SiLU())
                scalar_layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
        self.scalar_mlp = nn.Sequential(*scalar_layers)

        self._zero_init = zero_init
        if zero_init:
            for m in self.scalar_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)

    @property
    def layer_instantiated(self) -> bool:
        # AtomGraphSequential._instantiate_modules only walks top-level modules,
        # so we expose the inner IrrepsLinear's lazy-instantiation status here.
        return self.first_linear.layer_instantiated

    def instantiate(self) -> None:
        self.first_linear.instantiate()
        if self._zero_init:
            nn.init.zeros_(self.first_linear.linear.weight)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data = self.first_linear(data)
        if self._hidden_channels:
            data[self.key_output] = self.scalar_mlp(data[self._intermediate_key])
        return data


class LatentEwaldSum(nn.Module):
    """
    Computes long-range energy via Ewald summation on latent charges.

    Expects EdgePreprocess to have already run, which:
      - creates the _strain leaf and connects pos and cell to it
      - writes strained pos to data[KEY.POS]
      - writes strained cell to data[KEY.CELL]

    ForceStressOutput then differentiates w.r.t. strained pos (forces) and
    _strain (complete stress: SR virial + LR positional + LR cell/k-space).

    Args:
        les_args:         kwargs forwarded to Les().
        data_key_in:      per-atom latent charges (N_atoms, n_charges).
        data_key_out:     per-graph LR energy output.
        compute_bec:      if True, compute Born effective charges.
        bec_output_index: 0/1/2 for x/y/z component of BEC.
    """

    def __init__(
        self,
        les_args: Optional[dict] = None,
        data_key_in: str = KEY.LES_Q,
        data_key_out: str = KEY.LR_ENERGY,
        compute_bec: bool = False,
        bec_output_index: Optional[int] = None,
        ewald_type: str = 'batched',
    ):
        super().__init__()
        if les_args is None:
            les_args = {'use_atomwise': False}
        self.key_input = data_key_in
        self.key_output = data_key_out
        self.compute_bec = compute_bec
        self.bec_output_index = bec_output_index
        if compute_bec:
            raise ValueError(
                'compute_bec=True is not supported'
            )
        # ewald_type: 'batched' | 'flat' | 'auto' | 'triton' | 'cheng'
        self.ewald_type = ewald_type
        self._native_ewald = ewald_type in (
            'batched', 'flat', 'auto', 'triton'
        )
        dl = les_args.get('dl', 2.0)
        sigma = les_args.get('sigma', 1.0)
        rsi = les_args.get('remove_self_interaction', True)
        if ewald_type == 'batched':
            from .ewald import BatchedEwald
            self.ewald = BatchedEwald(dl=dl, sigma=sigma, remove_self_interaction=rsi)
        elif ewald_type == 'flat':
            from .ewald import FlatBatchedEwald
            self.ewald = FlatBatchedEwald(
                dl=dl, sigma=sigma, remove_self_interaction=rsi
            )
        elif ewald_type == 'auto':
            from .ewald import AutoBatchedEwald
            self.ewald = AutoBatchedEwald(
                dl=dl, sigma=sigma, remove_self_interaction=rsi
            )
        elif ewald_type == 'triton':
            try:
                from .ewald import TritonEwald
            except ImportError as e:
                raise ImportError(
                    "ewald_type='triton' requires the 'triton' package and a "
                    "CUDA GPU. Install triton or use 'batched'/'flat'/'auto'."
                ) from e
            self.ewald = TritonEwald(
                dl=dl, sigma=sigma, remove_self_interaction=rsi
            )
        elif ewald_type == 'cheng':
            try:
                from les import Les  # https://github.com/ChengUCB/les
            except ImportError as e:
                raise ImportError(
                    "The 'les' package is required for ewald_type='cheng'. "
                    "Install it"
                    "or use ewald_type='batched'/'hybrid' for the native kernel"
                ) from e
            self.les = Les(les_args)
        else:
            raise ValueError(f'Unknown ewald_type: {ewald_type}')
        self._is_batch_data = True  # set by AtomGraphSequential.set_is_batch_data()

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        q = data[self.key_input]   # (N_atoms, n_charges)
        pos = data[KEY.POS]        # strained pos from EdgePreprocess

        if self._is_batch_data:
            batch = data[KEY.BATCH].long()
            n_graphs = int(batch.max().item()) + 1
        else:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
            n_graphs = 1

        # Batched cell: SevenNet stores (3,3) per graph; PyG stacks to (3*n,3).
        # EdgePreprocess wrote the strained cell here, so les() receives a
        # tensor connected to _strain for correct stress computation.
        if KEY.CELL in data:
            cell = data[KEY.CELL].view(-1, 3, 3)  # (n_graphs, 3, 3)
        else:
            cell = torch.zeros((n_graphs, 3, 3), device=pos.device, dtype=pos.dtype)

        if self._native_ewald:
            # Native batched reciprocal-space sum (no per-structure loop).
            e_lr = self.ewald(q=q, r=pos, cell=cell, batch=batch)  # (n_graphs,)
        else:
            les_result = self.les(
                latent_charges=q,
                positions=pos,
                batch=batch,
                cell=cell,
                compute_energy=True,
                compute_bec=self.compute_bec,
                bec_output_index=self.bec_output_index,
            )
            e_lr = les_result['E_lr']  # (n_graphs,)
            if self.compute_bec:
                bec = les_result.get('BEC')
                if bec is not None:
                    data[KEY.LES_BEC] = bec

        assert e_lr is not None

        # Non-batch mode: squeeze to scalar to match SR_ENERGY from AtomReduce.
        data[self.key_output] = e_lr if self._is_batch_data else e_lr.squeeze()

        return data


class AddLREnergy(nn.Module):
    """Adds LR energy to SR energy: PRED_TOTAL_ENERGY = SR_ENERGY + LR_ENERGY."""

    def __init__(
        self,
        key_sr: str = KEY.SR_ENERGY,
        key_lr: str = KEY.LR_ENERGY,
        data_key_out: str = KEY.PRED_TOTAL_ENERGY,
    ):
        super().__init__()
        self.key_sr = key_sr
        self.key_lr = key_lr
        self.key_output = data_key_out

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_output] = data[self.key_sr] + data[self.key_lr]
        return data


class NeutralizeCharge(nn.Module):
    """
    Enforce charge-neutrality constraint per graph.

    Args:
        mode:
            'none'  → identity (skip; module not normally inserted)
            'shift' → q_i ← q_i − ⟨q⟩_graph             (uniform shift)
            'fukui' → q_i ← q_i − (Σq) · softplus(f_i) / Σ_j softplus(f_j)
                      (Fukui-style redistribution; needs KEY.LES_F input)
        data_key_q: per-atom latent charges (N, n_charges)
        data_key_f: per-atom Fukui factor (N, 1)  [only for 'fukui']
        eps:        denominator guard
    """

    def __init__(
        self,
        mode: str = 'none',
        data_key_q: str = KEY.LES_Q,
        data_key_f: str = KEY.LES_F,
        eps: float = 1e-12,
    ):
        super().__init__()
        if mode not in ('none', 'shift', 'fukui'):
            raise ValueError(
                f"Unknown neutralize_mode: {mode!r}. "
                "Choose from 'none' | 'shift' | 'fukui'."
            )
        self.mode = mode
        self.data_key_q = data_key_q
        self.data_key_f = data_key_f
        self.eps = eps
        self._is_batch_data = True  # set by AtomGraphSequential.set_is_batch_data

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        if self.mode == 'none':
            return data

        q = data[self.data_key_q]   # (N_atoms, n_charges)
        if self._is_batch_data:
            batch = data[KEY.BATCH].long()
            n_graphs = int(batch.max().item()) + 1
        else:
            batch = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
            n_graphs = 1

        n_ch = q.shape[1]

        if self.mode == 'shift':
            # per-graph mean
            sum_q = torch.zeros(n_graphs, n_ch, device=q.device, dtype=q.dtype)
            sum_q.scatter_add_(
                0, batch.unsqueeze(-1).expand(-1, n_ch), q
            )
            ones = torch.ones(q.shape[0], device=q.device, dtype=q.dtype)
            count = torch.zeros(n_graphs, device=q.device, dtype=q.dtype)
            count.scatter_add_(0, batch, ones)
            mean_q = sum_q / count.clamp(min=1.0).unsqueeze(-1)
            q = q - mean_q[batch]

        elif self.mode == 'fukui':
            f_raw = data[self.data_key_f]   # expected (N, 1) or (N,)
            if f_raw.dim() == 1:
                f_raw = f_raw.unsqueeze(-1)
            f = torch.nn.functional.softplus(f_raw)   # (N, 1) positive

            f_sum = torch.zeros(n_graphs, 1, device=q.device, dtype=q.dtype)
            f_sum.scatter_add_(0, batch.unsqueeze(-1), f)
            q_sum = torch.zeros(n_graphs, n_ch, device=q.device, dtype=q.dtype)
            q_sum.scatter_add_(
                0, batch.unsqueeze(-1).expand(-1, n_ch), q
            )
            # ratio (N, 1) broadcasts over n_ch
            ratio = f / (f_sum[batch] + self.eps)
            q = q - q_sum[batch] * ratio

        data[self.data_key_q] = q
        return data


class BornEffectiveCharge(nn.Module):
    """
    Native Born effective charge (BEC) readout.
    """

    def __init__(
        self,
        data_key_q: str = KEY.LES_Q,
        data_key_pos: str = KEY.POS,
        data_key_cell: str = KEY.CELL,
        data_key_out: str = KEY.LES_BEC,
        remove_mean: bool = True,
        epsilon_factor: float = 1.0,
        output_index: Optional[int] = None,
    ):
        super().__init__()
        self.data_key_q = data_key_q
        self.data_key_pos = data_key_pos
        self.data_key_cell = data_key_cell
        self.data_key_out = data_key_out
        self.remove_mean = remove_mean
        self.epsilon_factor = epsilon_factor
        self.normalization_factor = epsilon_factor ** 0.5
        self.output_index = output_index
        self._is_batch_data = True  # set by AtomGraphSequential.set_is_batch_data

    def _grad_real(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # d y / d x
        create = self.training
        if y.dim() == 1:
            g = torch.autograd.grad(
                [y], [x],
                grad_outputs=[torch.ones_like(y)],
                retain_graph=True,
                create_graph=create,
                allow_unused=True,
            )[0]
            return g if g is not None else torch.zeros_like(x)
        cols = []
        for i in range(y.shape[1]):
            g = torch.autograd.grad(
                [y[:, i]], [x],
                grad_outputs=[torch.ones_like(y[:, i])],
                retain_graph=True,
                create_graph=create,
                allow_unused=True,
            )[0]
            cols.append(g if g is not None else torch.zeros_like(x))
        return torch.stack(cols, dim=2)

    def _grad(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Complex y: split into real/imag parts (autograd is real-valued).
        if y.is_complex():
            return self._grad_real(y.real, x) + 1j * self._grad_real(y.imag, x)
        return self._grad_real(y, x)

    def _pol_pbc(self, r_now, q_now, box):
        # Berry-phase-style polarization for a periodic cell.
        r_frac = torch.matmul(r_now, torch.linalg.inv(box))
        phase = torch.exp(1j * 2.0 * torch.pi * r_frac)          # (n, 3)
        S = torch.sum(q_now * phase, dim=0)                      # (3,)
        pol = torch.matmul(box.to(S.dtype), S.unsqueeze(1)) / (1j * 2.0 * torch.pi)
        return pol.reshape(-1), phase

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        q = data[self.data_key_q]
        if q.dim() == 1:
            q = q.unsqueeze(1)
        q = q.sum(dim=1, keepdim=True)   # (N, 1) effective charge
        r = data[self.data_key_pos]      # (N, 3), strained, requires_grad

        if self._is_batch_data:
            batch = data[KEY.BATCH].long()
            n_graphs = int(batch.max().item()) + 1
        else:
            batch = torch.zeros(r.shape[0], dtype=torch.long, device=r.device)
            n_graphs = 1

        if self.data_key_cell in data:
            cell = data[self.data_key_cell].view(-1, 3, 3)
        else:
            cell = torch.zeros((n_graphs, 3, 3), device=r.device, dtype=r.dtype)

        all_P = []

        cdtype = torch.complex128 if r.dtype == torch.float64 else torch.complex64
        phase_shape = (r.shape[0],) if self.output_index is not None else r.shape
        phases = torch.zeros(phase_shape, dtype=cdtype, device=r.device)
        for i in range(n_graphs):
            mask = batch == i
            r_now, q_now = r[mask], q[mask]
            if self.remove_mean:
                q_now = q_now - q_now.mean(dim=0, keepdim=True)
            box = cell[i]
            if torch.linalg.det(box).abs() < 1e-6:
                pol = torch.sum(q_now * r_now, dim=0)            # (3,)
                phase = torch.ones_like(r_now, dtype=cdtype)
            else:
                pol, phase = self._pol_pbc(r_now, q_now, box)
            if self.output_index is not None:
                pol = pol[self.output_index]
                phase = phase[:, self.output_index]
            all_P.append(pol * self.normalization_factor)
            phases[mask] = phase.to(cdtype)

        P = torch.stack(all_P, dim=0)          # (n_graphs, 3) or (n_graphs,)

        if self.output_index is None:
            # grad -> (N, b, a); transpose so index1=P-component a, index2=r-component b
            bec = self._grad(P, r).transpose(1, 2).contiguous()   # (N, 3, 3)
            result = bec * phases.unsqueeze(2).conj()             # dephase over a
        else:
            bec = self._grad(P, r)                                # (N, 3)
            result = bec * phases.unsqueeze(1).conj()

        data[self.data_key_out] = result.real
        return data
