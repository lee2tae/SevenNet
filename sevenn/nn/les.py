"""
LES (Latent Ewald Summation) modules for SevenNet.

Architecture:
  NODE_FEATURE (last conv. layer, all-scalar)
      │
      ├─→ [LatentChargeReadout]  → LES_Q  (N_atoms, 1)
      │        (e3nn Linear, zero-init)
      │
      └─→ [init_feature_reduce] → SCALED_ATOMIC_ENERGY → ATOMIC_ENERGY
                                                               │
                                                  [AtomReduce (sum)]
                                                               │
                                                          SR_ENERGY  (n_graphs,)
                                                               │
  LES_Q ──→ [LatentEwaldSum] ──→ LR_ENERGY (n_graphs,)        │
                  (les lib)                                     │
                                                  [AddLREnergy (+)]
                                                               │
                                                    PRED_TOTAL_ENERGY
                                                               │
                                              [LESForceStressOutput]

Force computation decomposition:
  F_total = F_edge  +  F_lr_pos
  ├── F_edge   : d(E_total)/d(EDGE_VEC) via edge scatter
  │              Captures SR forces + q-path LR forces
  │              (q-path: LR_ENERGY → LES_Q → NODE_FEAT → EDGE_VEC)
  └── F_lr_pos : -d(LR_ENERGY)/d(POS)
                 Captures direct Ewald positional forces
                 (reciprocal-space: LR_ENERGY → POS directly)

Stress computation:
  Edge virial  : d(E_total)/d(EDGE_VEC) → virial formula (SR + q-path LR)
  Ewald stress : -d(LR_ENERGY)/d(LES_CELL) → cell-gradient formula
                 LES_CELL is a cloned differentiable copy of the cell stored
                 by LatentEwaldSum so LESForceStressOutput can differentiate
                 the direct reciprocal-space cell dependence.
                 Formula: σ_lr = -(1/V) * cell^T @ (dE_lr/dcell)
                 consistent with SevenNet's ForceStressOutput strain formula.

References:
  - LES library: https://github.com/ChengUCB/les
  - NequIP-LES: https://github.com/ChengUCB/nequip-les
"""
from typing import Optional

import torch
import torch.nn as nn
from e3nn.o3 import Irreps, Linear

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType
from .util import broadcast


class LatentChargeReadout(nn.Module):
    """
    Projects node features to one scalar latent charge per atom.

    Architecture (controlled by ``hidden_channels``):

        Single layer (default, ``hidden_channels=[]``):
            irreps_in  ──[e3nn Linear]──►  (N, 1)   charge

        Multi-layer MLP (e.g. ``hidden_channels=[128]``):
            irreps_in  ──[e3nn Linear]──►  (N, 128)
                       ──[SiLU]──────────►  (N, 128)
                       ──[nn.Linear]─────►  (N, 1)   charge

    The first layer is always an e3nn ``Linear`` so that arbitrary input
    irreps (mixed scalar/vector/tensor) are handled correctly.  Subsequent
    layers operate on plain scalars and use standard ``nn.Linear`` (no bias)
    with SiLU activations between them.

    Args:
        irreps_in:        e3nn irreps of the input node features.
        hidden_channels:  list of hidden scalar widths, e.g. ``[128]`` for
                          one hidden layer of width 128.  Empty list (default)
                          gives a direct single-layer projection.
        zero_init:        if ``True``, zero-initialise all weights so that
                          E_LR = 0 at the start.  Useful for transparent-wrapper
                          verification but NOT for real fine-tuning (zero charges
                          → zero Ewald gradient → no learning).
    """

    def __init__(
        self,
        irreps_in: Irreps,
        data_key_in: str = KEY.NODE_FEATURE,
        data_key_out: str = KEY.LES_Q,
        hidden_channels: Optional[list] = None,
        zero_init: bool = False,
    ):
        super().__init__()
        self.key_input = data_key_in
        self.key_output = data_key_out

        if hidden_channels is None:
            hidden_channels = []

        # First layer: irreps_in → first_dim x 0e  (e3nn, no bias)
        first_out = hidden_channels[0] if hidden_channels else 1
        self.first_linear = Linear(
            irreps_in, Irreps(f'{first_out}x0e'), biases=False
        )

        # Subsequent scalar layers: SiLU → nn.Linear per transition
        scalar_layers: list[nn.Module] = []
        if hidden_channels:
            dims = hidden_channels + [1]
            for i in range(len(dims) - 1):
                scalar_layers.append(nn.SiLU())
                scalar_layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
        self.scalar_mlp = nn.Sequential(*scalar_layers)

        if zero_init:
            nn.init.zeros_(self.first_linear.weight)
            for m in self.scalar_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = self.first_linear(data[self.key_input])  # (N_atoms, first_out)
        x = self.scalar_mlp(x)                       # (N_atoms, 1) after MLP
        data[self.key_output] = x
        return data


class LatentEwaldSum(nn.Module):
    """
    Computes long-range energy via Ewald summation on learned latent charges.

    Wraps the ``les`` library (https://github.com/ChengUCB/les).  Latent
    charges ``LES_Q`` (N_atoms, 1) are treated as point charges and fed into
    the Ewald sum, which decomposes the interaction energy into a real-space
    (short-range) and a reciprocal-space (long-range) part.

    Two differentiable leaf tensors are created for force/stress computation:

    POS leaf:
        ``data[KEY.POS].requires_grad_(True)`` is set so that
        ``LESForceStressOutput`` can differentiate the direct positional
        dependence of the Ewald sum (reciprocal-space term) w.r.t. positions.

    LES_CELL leaf (``KEY.LES_CELL``):
        A cloned copy of the cell with ``requires_grad=True`` is stored in
        data. ``LESForceStressOutput`` differentiates ``LR_ENERGY`` w.r.t.
        this to compute the direct Ewald cell-vector stress contribution.
        The formula ``σ_lr = -(1/V) * cell^T @ (dE_lr/dcell)`` is consistent
        with SevenNet's ``ForceStressOutput`` strain approach.

    The ``_is_batch_data`` flag is controlled by
    ``AtomGraphSequential.set_is_batch_data()`` at inference time.

    Args:
        les_args: keyword dict forwarded to ``Les()``.
        data_key_in: data key for per-atom latent charges (N_atoms, 1).
        data_key_out: data key for per-graph LR energy written by this module.
        compute_bec: if True, compute Born effective charges.
        bec_output_index: 0/1/2 to return only the x/y/z component of BEC.
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
                "Install it with: pip install git+https://github.com/ChengUCB/les.git"
            ) from e

        if les_args is None:
            les_args = {'use_atomwise': False}
        self.key_input = data_key_in
        self.key_output = data_key_out
        self.compute_bec = compute_bec
        self.bec_output_index = bec_output_index
        self.les = Les(les_args)
        # Controlled by AtomGraphSequential.set_is_batch_data()
        self._is_batch_data = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        q = data[self.key_input]   # (N_atoms, 1)
        pos = data[KEY.POS]        # (N_atoms, 3)

        # Enable positional gradients so LESForceStressOutput can compute
        # d(LR_ENERGY)/d(POS) — the direct Ewald positional force term.
        # Only acts on leaf tensors (precomputed positions in training pipeline).
        if torch.is_grad_enabled() and pos.is_leaf and not pos.requires_grad:
            pos.requires_grad_(True)

        if self._is_batch_data:
            batch = data[KEY.BATCH].long()           # (N_atoms,)
            n_graphs = int(batch.max().item()) + 1
        else:
            batch = torch.zeros(
                pos.shape[0], dtype=torch.long, device=pos.device
            )
            n_graphs = 1

        # SevenNet stores cell as (3, 3) per structure.  PyG's Collater
        # concatenates along dim-0 when batching, so batched cell is
        # (3*n_graphs, 3).  Reshaping to (-1, 3, 3) handles both cases.
        if KEY.CELL in data:
            cell = data[KEY.CELL].view(-1, 3, 3)  # (n_graphs, 3, 3)
        else:
            cell = torch.zeros(
                (n_graphs, 3, 3), device=pos.device, dtype=pos.dtype
            )

        # Create a differentiable copy of the cell for Ewald stress computation.
        # LESForceStressOutput differentiates LR_ENERGY w.r.t. les_cell to get
        # the direct Ewald cell-vector stress: σ_lr = -(1/V) * cell^T @ (dE/dcell).
        # Using detach().clone() ensures a fresh leaf regardless of whether cell
        # is a view or has upstream operations (e.g. from EdgePreprocess strain).
        les_cell = cell
        if torch.is_grad_enabled():
            les_cell = cell.detach().clone().requires_grad_(True)
            data[KEY.LES_CELL] = les_cell

        les_result = self.les(
            latent_charges=q,
            positions=pos,
            batch=batch,
            cell=les_cell,
            compute_energy=True,
            compute_bec=self.compute_bec,
            bec_output_index=self.bec_output_index,
        )

        e_lr = les_result['E_lr']  # (n_graphs,)
        assert e_lr is not None

        # Non-batch mode: return scalar to match AtomReduce's scalar SR_ENERGY
        data[self.key_output] = e_lr if self._is_batch_data else e_lr.squeeze()

        if self.compute_bec:
            bec = les_result.get('BEC')
            if bec is not None:
                data[KEY.LES_BEC] = bec

        return data


class AddLREnergy(nn.Module):
    """
    Adds long-range energy to short-range energy to produce PRED_TOTAL_ENERGY.

        PRED_TOTAL_ENERGY = SR_ENERGY + LR_ENERGY
    """

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
    Force and stress computation for LES-equipped models.

    Three-path force computation:

    Path 1 — edge gradient  (d(E_total)/d(EDGE_VEC)):
        Captures SR forces and q-path LR forces (LR_ENERGY → LES_Q → node
        features → EDGE_VEC).  Also provides the edge-virial for stress.

    Path 2 — position gradient  (d(E_total)/d(POS) = d(E_LR)/d(POS)):
        Captures the direct Ewald positional force that is NOT expressible as
        an edge contribution.  Specifically the reciprocal-space term
        E_k ~ |Σ_i q_i exp(ik·r_i)|² depends on absolute positions r_i.
        ``LatentEwaldSum`` enables this by calling ``pos.requires_grad_(True)``
        before the les() call.  In the training graph d(E_SR)/d(POS) = 0
        (EDGE_VEC is a precomputed leaf independent of POS), so
        d(E_total)/d(POS) reduces to d(E_LR)/d(POS).

    Total force: F = F_edge + F_lr_pos  (no double-counting: EDGE_VEC and POS
    are independent precomputed leaf tensors, so the q-path LR forces are
    captured exclusively by Path 1 and the direct Ewald forces exclusively by
    Path 2).

    Stress:

    Edge virial (Path 1):
        Same edge-based virial formula as ForceStressOutputFromEdge.
        Captures SR stress + q-path LR stress contribution.

    Ewald cell stress (Path 3 — d(LR_ENERGY)/d(LES_CELL)):
        ``LatentEwaldSum`` stores a cloned differentiable copy of the cell as
        ``KEY.LES_CELL``.  Differentiating ``LR_ENERGY`` w.r.t. that tensor
        gives the direct Ewald cell-vector stress contribution:

            σ_lr = -(1/V) * cell^T @ (dE_lr/dcell)

        This is derived from the strain formula used in
        ``ForceStressOutput``: under a strain ε applied as
        ``cell' = cell + cell @ sym_ε``,
        ``dE/dε = cell^T @ (dE/dcell')`` at ε = 0.
        The combined (edge-virial + cell-gradient) stress is stored in
        ``KEY.PRED_STRESS``.
    """

    def __init__(
        self,
        data_key_edge: str = KEY.EDGE_VEC,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        data_key_energy: str = KEY.PRED_TOTAL_ENERGY,
        data_key_pos: str = KEY.POS,
        data_key_force: str = KEY.PRED_FORCE,
        data_key_stress: str = KEY.PRED_STRESS,
        data_key_atomic_virial: str = KEY.PRED_ATOMIC_VIRIAL,
        data_key_cell_volume: str = KEY.CELL_VOLUME,
        use_atomic_virial: bool = False,
    ):
        super().__init__()
        self.key_edge = data_key_edge
        self.key_edge_idx = data_key_edge_idx
        self.key_energy = data_key_energy
        self.key_pos = data_key_pos
        self.key_force = data_key_force
        self.key_stress = data_key_stress
        self.key_atomic_virial = data_key_atomic_virial
        self.key_cell_volume = data_key_cell_volume
        self.use_atomic_virial = use_atomic_virial
        # Controlled by AtomGraphSequential.set_is_batch_data()
        self._is_batch_data = True

    def get_grad_key(self) -> str:
        # Primary grad key: AtomGraphSequential sets EDGE_VEC.requires_grad=True
        return self.key_edge

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        tot_num = torch.sum(data[KEY.NUM_ATOMS])
        rij = data[self.key_edge]        # requires_grad=True (from AtomGraphSequential)
        energy = [(data[self.key_energy]).sum()]
        edge_idx = data[self.key_edge_idx]
        pos = data[self.key_pos]

        # ── Single backward pass for all three gradient paths ──
        #
        # Three separate torch.autograd.grad calls would fail during inference:
        # create_graph=False → retain_graph=False, so Path 1 frees les()'s
        # saved tensors (via the charge path E_total→LR_ENERGY→les()→LES_Q→EDGE_VEC),
        # and Path 2/3 raise RuntimeError when they try to reuse the freed graph.
        # A single call computes all gradients in one backward traversal.
        #
        # In the training graph, EDGE_VEC and POS are independent precomputed
        # leaves, so:
        #   d(E_total)/d(EDGE_VEC) = d(E_SR)/d(EDGE_VEC) + d(E_LR)/d(EDGE_VEC)
        #                            (SR forces + q-path LR forces)
        #   d(E_total)/d(POS)      = d(E_LR)/d(POS)
        #                            (direct Ewald positional forces only;
        #                             d(E_SR)/d(POS)=0 as EDGE_VEC is a leaf)
        #   d(E_total)/d(LES_CELL) = d(E_LR)/d(LES_CELL)
        #                            (Ewald cell-vector stress contribution)
        has_les_cell = KEY.LES_CELL in data
        grad_inputs = [rij, pos]
        if has_les_cell:
            grad_inputs.append(data[KEY.LES_CELL])

        grads = torch.autograd.grad(
            energy,
            grad_inputs,
            create_graph=self.training,
            allow_unused=True,
        )

        fij = grads[0]       # d(E_total)/d(EDGE_VEC)
        pos_grad = grads[1]  # d(E_total)/d(POS) = d(E_LR)/d(POS) in training
        cell_grad = grads[2] if has_les_cell else None  # d(E_LR)/d(LES_CELL)

        force = torch.zeros(tot_num, 3, dtype=rij.dtype, device=rij.device)

        # ── Path 1: edge gradient → SR forces + q-path LR forces + edge virial ──
        if fij is not None:
            pf = torch.zeros(tot_num, 3, dtype=fij.dtype, device=fij.device)
            nf = torch.zeros(tot_num, 3, dtype=fij.dtype, device=fij.device)
            _edge_src = broadcast(edge_idx[0], fij, 0)
            _edge_dst = broadcast(edge_idx[1], fij, 0)
            pf.scatter_reduce_(0, _edge_src, fij, reduce='sum')
            nf.scatter_reduce_(0, _edge_dst, fij, reduce='sum')
            force = pf - nf

            # Stress via virial formula from edge gradient
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

            if self.use_atomic_virial:
                data[self.key_atomic_virial] = torch.neg(_s)

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

        # ── Path 2: position gradient → direct Ewald LR positional forces ──
        if pos_grad is not None:
            force = force - pos_grad  # F = -dE/dpos

        data[self.key_force] = force

        # ── Path 3: cell gradient → direct Ewald LR cell stress ──
        # σ_lr = -(1/V) * cell^T @ (dE_lr/dcell)
        # Derived from the strain formula: dE/dε = cell^T @ (dE/dcell') at ε=0
        # (consistent with SevenNet's ForceStressOutput).
        if cell_grad is not None:
            les_cell = data[KEY.LES_CELL]
            volume = data[self.key_cell_volume]
            if self._is_batch_data:
                lr_stress_3x3 = (
                    torch.neg(les_cell.transpose(-2, -1) @ cell_grad)
                    / volume.unsqueeze(-1).unsqueeze(-1)
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
                lr_stress_3x3 = (
                    torch.neg(les_cell.transpose(-2, -1) @ cell_grad).squeeze(0)
                    / volume
                )  # (3, 3)
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
