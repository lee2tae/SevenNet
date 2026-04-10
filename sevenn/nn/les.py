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

    A single e3nn Linear layer (irreps_in → 1x0e) with zero weight
    initialization, so the LR contribution starts at zero and is learned
    gradually without disrupting a pre-trained SR baseline.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        data_key_in: str = KEY.NODE_FEATURE,
        data_key_out: str = KEY.LES_Q,
    ):
        super().__init__()
        self.key_input = data_key_in
        self.key_output = data_key_out
        # Project to scalar; no bias (consistent with NequIP-LES ScalarMLP)
        self.linear = Linear(irreps_in, Irreps('1x0e'), biases=False)
        # Zero-init: guarantees E_lr = 0 at start of training
        nn.init.zeros_(self.linear.weight)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_output] = self.linear(data[self.key_input])
        return data


class LatentEwaldSum(nn.Module):
    """
    Computes long-range energy via Ewald summation on learned latent charges.

    Wraps the ``les`` library (https://github.com/ChengUCB/les).  Latent
    charges ``LES_Q`` (N_atoms, 1) are treated as point charges and fed into
    the Ewald sum, which decomposes the interaction energy into a real-space
    (short-range) and a reciprocal-space (long-range) part.

    Critically, this module sets ``data[KEY.POS].requires_grad_(True)`` so
    that ``LESForceStressOutput`` can differentiate the direct positional
    dependence of the Ewald sum (reciprocal-space term) w.r.t. positions.
    Without this, forces from the ``E_k ~ exp(ik·r_i)`` term are missing.

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
            data[KEY.POS] = pos  # update so LESForceStressOutput sees same tensor

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

    Two-path force computation:

    Path 1 — edge gradient  (d(E_total)/d(EDGE_VEC)):
        Captures SR forces and q-path LR forces (LR_ENERGY → LES_Q → node
        features → EDGE_VEC).  Also provides the virial for stress.

    Path 2 — position gradient  (d(LR_ENERGY)/d(POS)):
        Captures the direct Ewald positional force that is NOT expressible as
        an edge contribution.  Specifically the reciprocal-space term
        E_k ~ |Σ_i q_i exp(ik·r_i)|² depends on absolute positions r_i.
        ``LatentEwaldSum`` enables this by calling
        ``pos.requires_grad_(True)`` before the les() call.

    Total force: F = F_edge + F_lr_pos  (no double-counting because in the
    training graph, EDGE_VEC and POS are independent leaf tensors — the graph
    construction precomputes EDGE_VEC, so POS → EDGE_VEC path does not exist
    in the autograd graph).

    Stress: edge-based virial only (direct Ewald stress correction is small
    and requires the strain-tensor machinery used by ForceStressOutput; not
    implemented here).
    """

    def __init__(
        self,
        data_key_edge: str = KEY.EDGE_VEC,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        data_key_energy: str = KEY.PRED_TOTAL_ENERGY,
        data_key_lr_energy: str = KEY.LR_ENERGY,
        data_key_pos: str = KEY.POS,
        data_key_force: str = KEY.PRED_FORCE,
        data_key_stress: str = KEY.PRED_STRESS,
        data_key_cell_volume: str = KEY.CELL_VOLUME,
    ):
        super().__init__()
        self.key_edge = data_key_edge
        self.key_edge_idx = data_key_edge_idx
        self.key_energy = data_key_energy
        self.key_lr_energy = data_key_lr_energy
        self.key_pos = data_key_pos
        self.key_force = data_key_force
        self.key_stress = data_key_stress
        self.key_cell_volume = data_key_cell_volume
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

        # ── Path 1: edge gradient → SR forces + q-path LR forces + stress ──
        edge_grad = torch.autograd.grad(
            energy,
            [rij],
            create_graph=self.training,
            allow_unused=True,
        )
        fij = edge_grad[0]

        force = torch.zeros(tot_num, 3, dtype=rij.dtype, device=rij.device)

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
        # LatentEwaldSum set pos.requires_grad_(True), so this captures the
        # reciprocal-space Ewald force that path 1 misses.
        pos = data[self.key_pos]
        if self.key_lr_energy in data and pos.requires_grad:
            lr_energy = data[self.key_lr_energy]
            pos_grad = torch.autograd.grad(
                [lr_energy.sum()],
                [pos],
                create_graph=self.training,
                allow_unused=True,
            )[0]
            if pos_grad is not None:
                force = force - pos_grad  # F = -dE/dpos

        data[self.key_force] = force
        return data
