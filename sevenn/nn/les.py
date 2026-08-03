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
import warnings
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

    2D (slab) systems are recognized by the zero-row cell convention set by
    dataload._pbc_masked_cell: exactly one zero lattice vector and two
    independent nonzero ones. They are handled with EW3DC (Yeh-Berkowitz):
    the zero row is replaced on the fly by a vacuum-padded axis along the
    slab normal, the regular 3D Ewald kernel runs on that cell, and the
    analytic slab-dipole correction  E_c = norm_factor * M_n^2 / V  is added
    (M_n = sum_i q_i r_i.n_hat, per charge channel). The residual error
    decays as exp(-2*pi*gap/L_inplane), so the padding scales with the
    in-plane cell length.

    Args:
        les_args:     dl / sigma / remove_self_interaction settings, plus
                      slab_vacuum_factor (vacuum gap in units of the largest
                      in-plane cell length, default 1.5) and slab_min_pad
                      (minimum gap in Angstrom, default 8 * sigma).
        data_key_in:  per-atom latent charges (N_atoms, n_charges).
        data_key_out: per-graph LR energy output.
    """

    def __init__(
        self,
        les_args: Optional[dict] = None,
        data_key_in: str = KEY.LES_Q,
        data_key_out: str = KEY.LR_ENERGY,
        ewald_type: str = 'batched',
    ):
        super().__init__()
        if les_args is None:
            les_args = {}
        self.key_input = data_key_in
        self.key_output = data_key_out
        # ewald_type: 'batched' | 'flat' | 'auto' | 'triton'
        self.ewald_type = ewald_type
        dl = les_args.get('dl', 2.0)
        sigma = les_args.get('sigma', 1.0)
        rsi = les_args.get('remove_self_interaction', True)
        self.slab_vacuum_factor = les_args.get('slab_vacuum_factor', 1.5)
        self.slab_min_pad = les_args.get('slab_min_pad', 8.0 * sigma)
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
        else:
            raise ValueError(f'Unknown ewald_type: {ewald_type}')
        # open-boundary (non-pbc) counterpart
        if ewald_type == 'triton':
            from .ewald import TritonDirectSum
            self.direct = TritonDirectSum(
                sigma=sigma, remove_self_interaction=rsi
            )
        else:
            from .ewald import DirectSum
            self.direct = DirectSum(sigma=sigma, remove_self_interaction=rsi)
        self._is_batch_data = True  # set by AtomGraphSequential.set_is_batch_data()

    def _extend_slab_cells(self, q, pos, cell, batch, n_graphs):
        """
        EW3DC for slab graphs (exactly one zero cell row).
        Returns Yeh-Berkowitz dipole correction per graph or None
        """
        with torch.no_grad():
            row_norm = torch.linalg.norm(cell, dim=2)               # (G, 3)
            cand = ((row_norm > 1e-6).sum(dim=1) == 2) & (
                torch.linalg.det(cell).abs() <= 1e-6
            )
            if not bool(cand.any()):
                return cell, None
            i0 = torch.argmin(row_norm, dim=1)                      # zero row

        g = torch.arange(n_graphs, device=cell.device)
        a_j = cell[g, (i0 + 1) % 3]
        a_k = cell[g, (i0 + 2) % 3]
        nvec = torch.linalg.cross(a_j, a_k)                         # (G, 3)
        with torch.no_grad():
            is_2d = cand & (torch.linalg.norm(nvec, dim=1) > 1e-6)
            if not bool(is_2d.any()):
                return cell, None
        # unit fallback keeps norm/div grads NaN-free on non-slab graphs
        ez = torch.zeros_like(nvec)
        ez[:, 2] = 1.0
        nvec = torch.where(is_2d.unsqueeze(1), nvec, ez)
        n_hat = nvec / torch.linalg.norm(nvec, dim=1, keepdim=True)

        z = (pos * n_hat[batch]).sum(dim=1)                         # (N,)
        with torch.no_grad():
            fmax = torch.finfo(z.dtype).max
            zmax = torch.full(
                (n_graphs,), -fmax, device=z.device, dtype=z.dtype
            ).scatter_reduce(0, batch, z, 'amax')
            zmin = torch.full(
                (n_graphs,), fmax, device=z.device, dtype=z.dtype
            ).scatter_reduce(0, batch, z, 'amin')
            pad = (
                self.slab_vacuum_factor * row_norm.max(dim=1).values
            ).clamp(min=self.slab_min_pad)
            lz = (zmax - zmin).clamp(min=0.0) + pad                 # (G,)

        row_sel = torch.nn.functional.one_hot(i0, 3).bool()         # (G, 3)
        mask = (is_2d.unsqueeze(1) & row_sel).unsqueeze(2)          # (G, 3, 1)
        new_row = (lz.unsqueeze(1) * n_hat).unsqueeze(1)            # (G, 1, 3)
        cell = torch.where(mask, new_row, cell)

        # M_n from mean-centered normal coordinates (origin-independent)
        cnt = torch.zeros(n_graphs, device=z.device, dtype=z.dtype)
        cnt.scatter_add_(0, batch, torch.ones_like(z))
        zsum = torch.zeros(n_graphs, device=z.device, dtype=z.dtype)
        zsum.scatter_add_(0, batch, z)
        zc = z - (zsum / cnt.clamp(min=1.0))[batch]
        mn = torch.zeros(n_graphs, q.shape[1], device=z.device, dtype=z.dtype)
        mn.index_add_(0, batch, q * zc.unsqueeze(1))                # (G, n_q)
        vol = torch.linalg.det(cell).abs()
        vol = torch.where(is_2d, vol, torch.ones_like(vol))
        e_corr = self.ewald.norm_factor * (mn ** 2).sum(dim=1) / vol
        e_corr = torch.where(is_2d, e_corr, torch.zeros_like(e_corr))
        return cell, e_corr

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        q = data[self.key_input]   # (N_atoms, n_charges)
        pos = data[KEY.POS]        # strained pos from EdgePreprocess
        if q.dim() == 1:
            q = q.unsqueeze(1)

        if self._is_batch_data:
            batch = data[KEY.BATCH].long()
            n_graphs = int(batch.max().item()) + 1
        else:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
            n_graphs = 1

        # per-graph (3,3) cells, PyG-stacked to (3n,3)
        # EdgePreprocess requires KEY.CELL (which is the first layer of LES modules)
        # So this RuntimeError should never happen if the model is built correctly, but we check anyway.
        if KEY.CELL not in data:
            raise RuntimeError(
                f'{KEY.CELL} missing from graph data; LES requires it.'
            )
        cell = data[KEY.CELL].view(-1, 3, 3)  # (n_graphs, 3, 3)

        # slab (2D) graphs: vacuum-pad the cell + Yeh-Berkowitz correction
        cell, e_slab = self._extend_slab_cells(q, pos, cell, batch, n_graphs)

        # reciprocal sum for periodic graphs, direct sum for open (det ~ 0)
        is_pbc = torch.linalg.det(cell).abs() > 1e-6           # (n_graphs,)
        if bool(is_pbc.all()):
            e_lr = self.ewald(q=q, r=pos, cell=cell, batch=batch)
        elif not bool(is_pbc.any()):
            e_lr = self.direct(q=q, r=pos, batch=batch, n_graphs=n_graphs)
        else:
            atom_pbc = is_pbc[batch]
            remap_p = torch.cumsum(is_pbc.long(), 0) - 1
            remap_o = torch.cumsum((~is_pbc).long(), 0) - 1
            e_p = self.ewald(
                q=q[atom_pbc],
                r=pos[atom_pbc],
                cell=cell[is_pbc],
                batch=remap_p[batch[atom_pbc]],
            )
            e_o = self.direct(
                q=q[~atom_pbc],
                r=pos[~atom_pbc],
                batch=remap_o[batch[~atom_pbc]],
                n_graphs=n_graphs - int(is_pbc.sum().item()),
            )
            e_lr = torch.zeros(
                n_graphs, device=pos.device, dtype=pos.dtype
            )
            e_lr[is_pbc] = e_p
            e_lr[~is_pbc] = e_o

        if e_slab is not None:
            e_lr = e_lr + e_slab

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


class EpsilonFactorReadout(nn.Module):
    """
    Epsilon factor from per-atom raw values (KEY.LES_EPS_ATOMIC):
        mode='graph': eps_g = act(mean_i(x_i))   -> (n_graphs,)
        mode='atom':  eps_i = act(x_i)           -> (N_atoms,)
    """

    _SOFTPLUS_INV_ONE = 0.5413248546129181

    def __init__(
        self,
        data_key_in: str = KEY.LES_EPS_ATOMIC,
        data_key_out: str = KEY.LES_EPS,
        mode: str = 'graph',
        min_one: bool = False,
    ):
        super().__init__()
        if mode not in ('graph', 'atom'):
            raise ValueError(
                f"Unknown EpsilonFactorReadout mode: {mode!r}. "
                "Choose from 'graph' | 'atom'."
            )
        self.key_input = data_key_in
        self.key_output = data_key_out
        self.mode = mode
        self.min_one = min_one
        self._is_batch_data = True  # set by AtomGraphSequential.set_is_batch_data

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.min_one:
            return 1.0 + torch.nn.functional.softplus(x)
        return torch.nn.functional.softplus(x + self._SOFTPLUS_INV_ONE)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_input].reshape(-1)   # (N_atoms,)
        if self.mode == 'atom':
            data[self.key_output] = self._activate(x)
            return data

        if self._is_batch_data:
            batch = data[KEY.BATCH].long()
            n_graphs = int(batch.max().item()) + 1
        else:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            n_graphs = 1

        sum_x = torch.zeros(n_graphs, device=x.device, dtype=x.dtype)
        sum_x.scatter_add_(0, batch, x)
        count = torch.zeros(n_graphs, device=x.device, dtype=x.dtype)
        count.scatter_add_(0, batch, torch.ones_like(x))
        mean_x = sum_x / count.clamp(min=1.0)

        data[self.key_output] = self._activate(mean_x)
        return data


class BornEffectiveCharge(nn.Module):
    """
    Native Born effective charge (BEC) and dipole readout.

    epsilon_mode:
        'fixed'   → P is scaled by sqrt(epsilon_factor) (build-time constant).
        'learned' → P is scaled by sqrt(data[KEY.LES_EPS]) per graph, predicted
                    by EpsilonFactorReadout upstream.
        'learned_atomic' → per-atom screening: q_i is scaled by sqrt(eps_i)
                    BEFORE neutralization, so the screened charges are
                    re-neutralized and the acoustic sum rule survives.
                    data[KEY.LES_EPS] holds per-atom eps (N,).
    """

    def __init__(
        self,
        data_key_q: str = KEY.LES_Q,
        data_key_pos: str = KEY.POS,
        data_key_cell: str = KEY.CELL,
        data_key_out: str = KEY.LES_BEC,
        data_key_dipole: str = KEY.LES_DIPOLE,
        data_key_epsilon: str = KEY.LES_EPS,
        remove_mean: str = 'uniform',
        epsilon_factor: float = 1.0,
        epsilon_mode: str = 'fixed',
        output_index: Optional[int] = None,
        compute_bec: bool = False,
        compute_dipole: bool = False,
    ):
        super().__init__()
        if epsilon_mode not in ('fixed', 'learned', 'learned_atomic'):
            raise ValueError(
                f"Unknown epsilon_mode: {epsilon_mode!r}. "
                "Choose from 'fixed' | 'learned' | 'learned_atomic'."
            )
        if isinstance(remove_mean, bool):
            remove_mean = 'uniform' if remove_mean else 'none'
        if remove_mean not in ('none', 'uniform', 'q_projection'):
            raise ValueError(
                f"Unknown remove_mean: {remove_mean!r}. "
                "Choose from 'none' | 'uniform' | 'q_projection'."
            )
        if remove_mean == 'q_projection' and epsilon_mode != 'learned_atomic':
            raise ValueError(
                "remove_mean='q_projection' requires "
                "epsilon_mode='learned_atomic'."
            )
        self.data_key_q = data_key_q
        self.data_key_pos = data_key_pos
        self.data_key_cell = data_key_cell
        self.data_key_out = data_key_out
        self.data_key_dipole = data_key_dipole
        self.data_key_epsilon = data_key_epsilon
        self.remove_mean = remove_mean
        self.epsilon_factor = epsilon_factor
        self.normalization_factor = epsilon_factor ** 0.5
        self.epsilon_mode = epsilon_mode
        self.output_index = output_index
        self.compute_bec = compute_bec
        self.compute_dipole = compute_dipole
        self._is_batch_data = True  # set by AtomGraphSequential.set_is_batch_data
        # One vmapped VJP for all BEC components; flips off permanently if an
        # op in the backward graph has no vmap batching rule.
        self._use_batched_grad = True

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

    def _polarization(self, q, r, cell, batch, n_graphs, factors):
        # Berry-phase-style polarization, vectorized over graphs.
        cdtype = torch.complex128 if r.dtype == torch.float64 else torch.complex64
        eye = torch.eye(3, device=r.device, dtype=r.dtype)
        with torch.no_grad():
            row_norm = torch.linalg.norm(cell, dim=2)            # (G, 3)
            row_nz = row_norm > 1e-6
            pbc3 = torch.linalg.det(cell).abs() >= 1e-6          # (G,)
            cand = (row_nz.sum(dim=1) == 2) & ~pbc3
            i0 = torch.argmin(row_norm, dim=1)                   # zero row
        g = torch.arange(n_graphs, device=r.device)
        a_j = cell[g, (i0 + 1) % 3]
        a_k = cell[g, (i0 + 2) % 3]
        nvec = torch.linalg.cross(a_j, a_k)                      # (G, 3)
        with torch.no_grad():
            is_2d = cand & (torch.linalg.norm(nvec, dim=1) > 1e-6)
        ez = torch.zeros_like(nvec)
        ez[:, 2] = 1.0
        nvec = torch.where(is_2d.unsqueeze(1), nvec, ez)
        n_hat = nvec / torch.linalg.norm(nvec, dim=1, keepdim=True)

        # per-direction Berry mask and invertible cell
        dmask = torch.where(
            is_2d.unsqueeze(1), row_nz, pbc3.unsqueeze(1).expand(-1, 3)
        )                                                        # (G, 3)
        row_sel = torch.nn.functional.one_hot(i0, 3).bool()      # (G, 3)
        safe_cell = torch.where(pbc3.view(-1, 1, 1), cell, eye)
        slab_cell = torch.where(
            (is_2d.unsqueeze(1) & row_sel).unsqueeze(2),
            n_hat.unsqueeze(1), cell,
        )
        safe_cell = torch.where(is_2d.view(-1, 1, 1), slab_cell, safe_cell)

        r_frac = torch.einsum('nd,nde->ne', r, torch.linalg.inv(safe_cell)[batch])
        phase = torch.exp(1j * 2.0 * torch.pi * r_frac)          # (N, 3)
        phase = torch.where(dmask[batch], phase, torch.ones_like(phase))
        S = torch.zeros(n_graphs, 3, device=r.device, dtype=cdtype)
        S.index_add_(0, batch, q * phase)
        p_pbc = S / (1j * 2.0 * torch.pi)                        # (G, 3) lattice
        p_open = torch.zeros(n_graphs, 3, device=r.device, dtype=r.dtype)
        p_open.index_add_(0, batch, q * r_frac)                  # (G, 3) cartesian
        p = torch.where(dmask, p_pbc, p_open.to(cdtype))
        M = safe_cell                                            # (G, 3, 3)
        return p * factors.unsqueeze(1), phase, M

    def _bec_grad_batched(self, p, r, phase):
        # All lattice-component x {real, imag} cotangents in one vmapped
        # backward. Returns z (N, e, b), phase-corrected per lattice dir.
        n_graphs = p.shape[0]
        Y = torch.stack([p.real, p.imag]).permute(0, 2, 1).reshape(6, n_graphs)
        cot = torch.eye(6, device=r.device, dtype=r.dtype)
        cot = cot.unsqueeze(-1).expand(6, 6, n_graphs)
        (g,) = torch.autograd.grad(
            [Y], [r],
            grad_outputs=[cot],
            retain_graph=True,
            create_graph=self.training,
            is_grads_batched=True,
        )                                                         # (6, N, 3)
        bec = (g[0:3] + 1j * g[3:6]).permute(1, 0, 2)             # (N, e, b)
        return (bec * phase.conj().unsqueeze(2)).real

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        r = data[self.data_key_pos]      # (N, 3), strained, requires_grad

        # Skip guard: batch has BEC refs but none labeled (all NaN filler).
        if (
            self.compute_bec
            and not self.compute_dipole
            and KEY.LES_BEC_REF in data
            and bool(torch.isnan(data[KEY.LES_BEC_REF]).all())
        ):
            shape = (
                (r.shape[0], 3) if self.output_index is not None
                else (r.shape[0], 3, 3)
            )
            data[self.data_key_out] = torch.zeros(
                shape, device=r.device, dtype=r.dtype
            )
            return data

        q = data[self.data_key_q]
        if q.dim() == 1:
            q = q.unsqueeze(1)
        q = q.sum(dim=1, keepdim=True)   # (N, 1) effective charge

        if self._is_batch_data:
            batch = data[KEY.BATCH].long()
            n_graphs = int(batch.max().item()) + 1
        else:
            batch = torch.zeros(r.shape[0], dtype=torch.long, device=r.device)
            n_graphs = 1

        # redundant like the one in LatentEwaldSum; kept for the message
        if self.data_key_cell not in data:
            raise RuntimeError(
                f'{self.data_key_cell} missing from graph data; LES BEC '
                'requires it. Build graphs with with_shift=True.'
            )
        cell = data[self.data_key_cell].view(-1, 3, 3)

        if self.epsilon_mode == 'learned_atomic':
            # scale BEFORE neutralize so the acoustic sum rule survives
            eps_atom = data[self.data_key_epsilon].reshape(-1, 1)  # (N, 1)
            q_scaled = q * eps_atom.sqrt()
            if self.remove_mean == 'q_projection':
                # min-norm correction to sqrt(eps) - 1 s.t. sum q = 0
                S = torch.zeros(n_graphs, 1, device=q.device, dtype=q.dtype)
                D = torch.zeros(n_graphs, 1, device=q.device, dtype=q.dtype)
                S.index_add_(0, batch, q_scaled)
                D.index_add_(0, batch, q * q)
                q = q_scaled - (q * q) * (S / D.clamp(min=1e-8))[batch]
            else:
                q = q_scaled
            factors = torch.ones(n_graphs, device=r.device, dtype=r.dtype)
        elif self.epsilon_mode == 'learned':
            factors = data[self.data_key_epsilon].reshape(-1) ** 0.5  # (n_graphs,)
        else:
            factors = torch.full(
                (n_graphs,), self.normalization_factor,
                device=r.device, dtype=r.dtype,
            )

        if self.remove_mean == 'uniform':
            sum_q = torch.zeros(n_graphs, 1, device=q.device, dtype=q.dtype)
            cnt = torch.zeros(n_graphs, 1, device=q.device, dtype=q.dtype)
            sum_q.index_add_(0, batch, q)
            cnt.index_add_(0, batch, torch.ones_like(q))
            q = q - (sum_q / cnt.clamp(min=1.0))[batch]

        p, phases, M = self._polarization(q, r, cell, batch, n_graphs, factors)

        if self.compute_dipole:
            dip = torch.einsum('gea,ge->ga', M.to(p.dtype), p).real
            if self.output_index is not None:
                dip = dip[:, self.output_index]           # (n_graphs,)
            data[self.data_key_dipole] = (
                dip if self._is_batch_data else dip.squeeze(0)
            )

        if self.compute_bec:
            z = None                                      # (N, e, b) per-lattice
            if self._use_batched_grad:
                try:
                    z = self._bec_grad_batched(p, r, phases)
                except torch.OutOfMemoryError:
                    raise  # fallback loop uses MORE memory; do not mask OOM
                except RuntimeError as e:
                    self._use_batched_grad = False
                    warnings.warn(
                        'Batched BEC autograd failed, falling back to the '
                        f'per-component loop: {e}\nFor faster BEC training, '
                        'call e3nn.set_optimization_defaults(jit_script_fx='
                        'False) before building the model.'
                    )
            if z is None:
                # grad -> (N, b, e); transpose so index1=lattice e, index2=r-comp b
                bec = self._grad(p, r).transpose(1, 2).contiguous()   # (N, 3, 3)
                z = (bec * phases.unsqueeze(2).conj()).real
            # Z*_ab = sum_e M_ea z_eb
            result = torch.einsum('nea,neb->nab', M[batch], z)
            if self.output_index is not None:
                result = result[:, self.output_index]     # (N, 3)
            data[self.data_key_out] = result

        return data
