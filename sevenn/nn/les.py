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
                                           [LESForceStressOutput]

Forces:
  F_edge   = -d(E_total)/d(EDGE_VEC)  SR forces + q-path LR forces
  F_lr_pos = -d(E_LR)/d(POS)          direct Ewald positional forces

Stress:
  σ_edge = -(1/V) Σ_ij F_ij ⊗ r_ij   edge virial (SR + q-path LR)
  σ_lr   = -(1/V) d(E_LR)/d(LES_STRAIN)  complete Ewald stress via
             affine strain applied to pos and cell before les()

EDGE_VEC and POS are independent precomputed leaves, so the three
gradient paths are non-overlapping.

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
from .util import broadcast


class LatentChargeReadout(nn.Module):
    """
    Projects node features to per-atom latent charges.

    Architecture (controlled by ``hidden_channels``):
        hidden_channels=[] (default):
            irreps_in ──[IrrepsLinear]──► (N, n_charges)
        hidden_channels=[H, ...]:
            irreps_in ──[IrrepsLinear]──► (N, H) ──[SiLU + nn.Linear]──► (N, n_charges)

    The first layer is SevenNet's IrrepsLinear so that ``set_num_modalities``
    can make the q-readout per-channel: each modality gets an independent set
    of weights via the standard one-hot concatenation trick. The downstream
    scalar MLP then sees modality-conditioned features.

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
        return self.first_linear.layer_instantiated

    def instantiate(self) -> None:
        self.first_linear.instantiate()
        if self._zero_init:
            nn.init.zeros_(self.first_linear.linear.weight)

    def set_num_modalities(self, num_modalities: int) -> None:
        """Make the q-readout modality-aware: each channel gets its own weights."""
        self.first_linear.set_num_modalities(num_modalities)

    @property
    def _is_batch_data(self) -> bool:
        return self.first_linear._is_batch_data

    @_is_batch_data.setter
    def _is_batch_data(self, value: bool) -> None:
        # AtomGraphSequential.set_is_batch_data only walks top-level modules;
        # propagate the flag to the inner IrrepsLinear so its
        # _patch_modal_to_data picks the correct batched/non-batched branch.
        self.first_linear._is_batch_data = value

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        # IrrepsLinear._patch_modal_to_data concats the modality one-hot into
        # data[key_input] in place. With key_in == key_out the linear's output
        # immediately overwrites it; here key_in != key_out, so we must restore
        # the original NODE_FEATURE for downstream modality-aware layers
        # (e.g. reduce_input_to_hidden) which would otherwise re-concat the
        # one-hot onto an already-augmented tensor.
        saved_input = data[self.key_input]
        data = self.first_linear(data)
        data[self.key_input] = saved_input
        if self._hidden_channels:
            data[self.key_output] = self.scalar_mlp(data[self._intermediate_key])
        return data


class LatentEwaldSum(nn.Module):
    """
    Computes long-range energy via Ewald summation on latent charges.

    Sets up two differentiable leaves for LESForceStressOutput:

    POS (KEY.POS):
        requires_grad is enabled so d(E_LR)/d(pos) gives direct Ewald forces.

    LES_STRAIN (KEY.LES_STRAIN):
        Zero (n_graphs, 3, 3) leaf. Its symmetric part is applied to both pos
        and cell before les(), so d(E_LR)/d(les_strain) captures the complete
        Ewald stress (positional virial + cell contribution) in one autograd
        call. Formula: σ_lr = -(1/V) * strain_grad.

    Args:
        les_args:         kwargs forwarded to Les().
        data_key_in:      per-atom latent charges (N_atoms, 1).
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
    ):
        super().__init__()
        try:
            from les import Les  # https://github.com/ChengUCB/les
        except ImportError as e:
            raise ImportError(
                "The 'les' package is required for LES support. "
                'Install it with: pip install git+https://github.com/ChengUCB/les.git'
            ) from e

        if les_args is None:
            les_args = {'use_atomwise': False}
        self.key_input = data_key_in
        self.key_output = data_key_out
        self.compute_bec = compute_bec
        self.bec_output_index = bec_output_index
        self.les = Les(les_args)
        self._is_batch_data = True  # set by AtomGraphSequential.set_is_batch_data()

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        q = data[self.key_input]   # (N_atoms, n_charges)
        pos = data[KEY.POS]        # (N_atoms, 3)

        # Enable pos gradients for direct Ewald force (Path 2).
        if torch.is_grad_enabled() and pos.is_leaf and not pos.requires_grad:
            pos.requires_grad_(True)

        if self._is_batch_data:
            batch = data[KEY.BATCH].long()
            n_graphs = int(batch.max().item()) + 1
        else:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
            n_graphs = 1

        # Batched cell: SevenNet stores (3,3) per graph; PyG stacks to (3*n,3).
        if KEY.CELL in data:
            cell = data[KEY.CELL].view(-1, 3, 3)  # (n_graphs, 3, 3)
        else:
            cell = torch.zeros((n_graphs, 3, 3), device=pos.device, dtype=pos.dtype)

        if torch.is_grad_enabled():
            # Strain leaf for LR stress (Path 3).
            # Apply symmetric strain to pos and cell so that
            # d(E_LR)/d(les_strain) gives the complete affine-deformation
            # Ewald stress in one autograd call.
            les_strain = torch.zeros(
                (n_graphs, 3, 3), dtype=pos.dtype, device=pos.device,
            )
            les_strain.requires_grad_(True)
            data[KEY.LES_STRAIN] = les_strain

            sym_strain = 0.5 * (les_strain + les_strain.transpose(-1, -2))
            pos = pos + torch.bmm(pos.unsqueeze(-2), sym_strain[batch]).squeeze(-2)
            cell = cell + torch.bmm(cell, sym_strain)

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
        assert e_lr is not None

        # Non-batch mode: squeeze to scalar to match SR_ENERGY from AtomReduce.
        data[self.key_output] = e_lr if self._is_batch_data else e_lr.squeeze()

        if self.compute_bec:
            bec = les_result.get('BEC')
            if bec is not None:
                data[KEY.LES_BEC] = bec

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


class LESForceStressOutput(nn.Module):
    """
    Force and stress output for LES models. Replaces ForceStressOutputFromEdge.

    Three gradient paths in a single torch.autograd.grad call:
      Path 1  d(E_total)/d(EDGE_VEC)  SR + q-path LR forces; edge virial stress
      Path 2  d(E_LR)/d(POS)          direct Ewald positional forces
      Path 3  d(E_LR)/d(LES_STRAIN)   complete Ewald stress (pos + cell)

    All three are computed in one call because separate calls would free the
    graph after Path 1 (retain_graph=False when create_graph=False at inference),
    causing Path 2/3 to fail.

    Atomic virial is not supported: direct Ewald forces (Path 2) depend on
    absolute positions and have no pairwise decomposition.
    """

    def __init__(
        self,
        data_key_edge: str = KEY.EDGE_VEC,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        data_key_energy: str = KEY.PRED_TOTAL_ENERGY,
        data_key_pos: str = KEY.POS,
        data_key_force: str = KEY.PRED_FORCE,
        data_key_stress: str = KEY.PRED_STRESS,
        data_key_cell_volume: str = KEY.CELL_VOLUME,
        use_atomic_virial: bool = False,
    ):
        super().__init__()
        if use_atomic_virial:
            raise NotImplementedError(
                'Atomic virial is not supported for LES models. '
                'Direct Ewald forces (Path 2) have no pairwise decomposition. '
                'Use total stress instead.'
            )
        self.key_edge = data_key_edge
        self.key_edge_idx = data_key_edge_idx
        self.key_energy = data_key_energy
        self.key_pos = data_key_pos
        self.key_force = data_key_force
        self.key_stress = data_key_stress
        self.key_cell_volume = data_key_cell_volume
        self.use_atomic_virial = False
        self._is_batch_data = True  # set by AtomGraphSequential.set_is_batch_data()

    def get_grad_key(self) -> str:
        return self.key_edge

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        tot_num = torch.sum(data[KEY.NUM_ATOMS])
        rij = data[self.key_edge]
        energy = [(data[self.key_energy]).sum()]
        edge_idx = data[self.key_edge_idx]
        pos = data[self.key_pos]

        has_les_strain = KEY.LES_STRAIN in data
        grad_inputs = [rij, pos]
        if has_les_strain:
            grad_inputs.append(data[KEY.LES_STRAIN])

        grads = torch.autograd.grad(
            energy,
            grad_inputs,
            create_graph=self.training,
            allow_unused=True,
        )

        fij = grads[0]          # d(E_total)/d(EDGE_VEC): SR + q-path LR forces
        pos_grad = grads[1]     # d(E_LR)/d(POS): direct Ewald forces
        strain_grad = grads[2] if has_les_strain else None  # d(E_LR)/d(LES_STRAIN): Ewald stress

        force = torch.zeros(tot_num, 3, dtype=rij.dtype, device=rij.device)

        # ── Path 1: edge gradient → forces + edge virial stress ──
        if fij is not None:
            pf = torch.zeros(tot_num, 3, dtype=fij.dtype, device=fij.device)
            nf = torch.zeros(tot_num, 3, dtype=fij.dtype, device=fij.device)
            _edge_src = broadcast(edge_idx[0], fij, 0)
            _edge_dst = broadcast(edge_idx[1], fij, 0)
            pf.scatter_reduce_(0, _edge_src, fij, reduce='sum')
            nf.scatter_reduce_(0, _edge_dst, fij, reduce='sum')
            force = pf - nf

            diag = rij * fij
            s12 = rij[..., 0] * fij[..., 1]
            s23 = rij[..., 1] * fij[..., 2]
            s31 = rij[..., 2] * fij[..., 0]
            _virial = torch.cat(
                [diag, s12.unsqueeze(-1), s23.unsqueeze(-1), s31.unsqueeze(-1)],
                dim=-1,
            )
            _s = torch.zeros(tot_num, 6, dtype=fij.dtype, device=fij.device)
            _edge_dst6 = broadcast(edge_idx[1], _virial, 0)
            _s.scatter_reduce_(0, _edge_dst6, _virial, reduce='sum')

            if self._is_batch_data:
                batch = data[KEY.BATCH]
                nbatch = int(batch.max().cpu().item()) + 1
                sout = torch.zeros(
                    (nbatch, 6), dtype=_virial.dtype, device=_virial.device
                )
                _batch = broadcast(batch, _s, 0)
                sout.scatter_reduce_(0, _batch, _s, reduce='sum')
            else:
                sout = torch.sum(_s, dim=0)

            data[self.key_stress] = (
                torch.neg(sout) / data[self.key_cell_volume].unsqueeze(-1)
            )

        # ── Path 2: position gradient → direct Ewald forces ──
        if pos_grad is not None:
            force = force - pos_grad

        data[self.key_force] = force

        # ── Path 3: strain gradient → complete LR stress ──
        # σ_lr = -(1/V) * strain_grad  (affine deformation: pos + cell both strained)
        if strain_grad is not None:
            volume = data[self.key_cell_volume]
            if self._is_batch_data:
                lr_stress_3x3 = (
                    torch.neg(strain_grad) / volume.unsqueeze(-1).unsqueeze(-1)
                )
                lr_stress_voigt = torch.stack(
                    [
                        lr_stress_3x3[:, 0, 0],
                        lr_stress_3x3[:, 1, 1],
                        lr_stress_3x3[:, 2, 2],
                        lr_stress_3x3[:, 0, 1],
                        lr_stress_3x3[:, 1, 2],
                        lr_stress_3x3[:, 0, 2],
                    ],
                    dim=-1,
                )  # (n_graphs, 6)
            else:
                lr_stress_3x3 = torch.neg(strain_grad.squeeze(0)) / volume  # (3, 3)
                lr_stress_voigt = torch.stack(
                    [
                        lr_stress_3x3[0, 0],
                        lr_stress_3x3[1, 1],
                        lr_stress_3x3[2, 2],
                        lr_stress_3x3[0, 1],
                        lr_stress_3x3[1, 2],
                        lr_stress_3x3[0, 2],
                    ]
                )  # (6,)

            if self.key_stress in data:
                data[self.key_stress] = data[self.key_stress] + lr_stress_voigt
            else:
                data[self.key_stress] = lr_stress_voigt

        return data
