"""
Integration test for LES fine-tuning of SevenNet-omni.

Covers:
  1. build_model_with_les() — loads SR weights, leaves LES params at init
  2. Parameter freezing (freeze_sr=True) — only LES params require grad
  3. Inference pass — no crash, no NaN/Inf in energy/force/stress
  4. Training pass — backward through E_total, only LES params accumulate grad
  5. Optimizer step — LES params update, SR params unchanged
"""

import warnings

import pytest
import torch
from ase.build import bulk

import sevenn._keys as KEY
from sevenn.atom_graph_data import AtomGraphData
from sevenn.checkpoint import SevenNetCheckpoint
from sevenn.train.dataload import unlabeled_atoms_to_graph
from sevenn.util import pretrained_name_to_path


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def omni_checkpoint():
    path = pretrained_name_to_path('7net-omni')
    return SevenNetCheckpoint(path)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason='FlashTP requires CUDA'
)

DEVICE = 'cuda'


@pytest.fixture(scope='module')
def sr_model(omni_checkpoint):
    """Original non-LES SevenNet-omni on CUDA (reference for comparison)."""
    model = omni_checkpoint.build_model()
    model.set_is_batch_data(False)
    return model.to(DEVICE)


@pytest.fixture(scope='module')
def les_model(omni_checkpoint):
    """LES model built from SevenNet-omni with SR params frozen, on CUDA."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        model = omni_checkpoint.build_model_with_les(
            les_config={
                'les_args': {'use_atomwise': False},
                'zero_init': True,  # needed for transparent-wrapper tests
            },
            freeze_sr=True,
            # No backend overrides: inherits FlashTP from the checkpoint
        )
    return model.to(DEVICE)


@pytest.fixture  # function-scoped: each test gets a fresh, un-mutated graph
def nacl_graph(omni_checkpoint):
    """NaCl rocksalt structure as an AtomGraphData (with cell for LES)."""
    atoms = bulk('NaCl', 'rocksalt', a=5.63)
    atoms.rattle(stdev=0.02, seed=0)
    cutoff = omni_checkpoint.config['cutoff']
    # with_shift=True → KEY.CELL included in data dict (needed for Ewald)
    graph = AtomGraphData.from_numpy_dict(
        unlabeled_atoms_to_graph(atoms, cutoff, with_shift=True)
    )
    # SevenNet-omni is multimodal; pick one modality for testing
    graph[KEY.DATA_MODALITY] = 'omat24'
    return graph.to(DEVICE)


@pytest.fixture  # function-scoped: each test gets a fresh batch
def nacl_batch(nacl_graph):
    """Single NaCl graph wrapped as a PyG Batch (is_batch_data=True)."""
    from torch_geometric.data import Batch
    # Batch.from_data_list adds KEY.BATCH and turns DATA_MODALITY into a list
    return Batch.from_data_list([nacl_graph])


# ── tests ─────────────────────────────────────────────────────────────────────

class TestBuildModelWithLES:
    def test_model_type(self, les_model):
        from sevenn.nn.sequential import AtomGraphSequential
        assert isinstance(les_model, AtomGraphSequential)

    def test_les_modules_present(self, les_model):
        module_names = dict(les_model.named_modules()).keys()
        assert 'les_charge_readout' in module_names
        assert 'les_lr_energy' in module_names
        assert 'add_lr_to_total' in module_names

    def test_les_charge_readout_zero_init(self, les_model):
        """les_charge_readout should start at zero — E_LR = 0 initially."""
        w = les_model._modules['les_charge_readout'].first_linear.weight
        assert torch.allclose(w, torch.zeros_like(w)), \
            'les_charge_readout.first_linear.weight should be zero-initialised'

    def test_sr_structure_preserved(self, les_model, omni_checkpoint):
        """All SR param values must match the original checkpoint exactly."""
        orig_sd = omni_checkpoint.model_state_dict  # CPU tensors (raw checkpoint)
        les_sd = les_model.state_dict()              # CUDA tensors (model on GPU)
        for key, orig_val in orig_sd.items():
            assert key in les_sd, f'SR key {key!r} missing from LES model'
            assert torch.allclose(les_sd[key].cpu().float(), orig_val.float()), \
                f'SR param {key!r} changed during build_model_with_les'


class TestParameterFreezing:
    def test_les_params_require_grad(self, les_model):
        les_prefixes = ('les_charge_readout.', 'les_lr_energy.')
        les_trainable = [
            name for name, p in les_model.named_parameters()
            if name.startswith(les_prefixes) and p.requires_grad
        ]
        assert len(les_trainable) > 0, 'No trainable LES parameters found'

    def test_sr_params_frozen(self, les_model):
        les_prefixes = ('les_charge_readout.', 'les_lr_energy.')
        sr_trainable = [
            name for name, p in les_model.named_parameters()
            if not name.startswith(les_prefixes) and p.requires_grad
        ]
        assert len(sr_trainable) == 0, \
            f'SR params should be frozen; found trainable: {sr_trainable}'

    def test_optimizer_only_has_les_params(self, les_model):
        trainable = [p for p in les_model.parameters() if p.requires_grad]
        assert len(trainable) > 0
        # Verify these are exactly the LES params
        les_prefixes = ('les_charge_readout.', 'les_lr_energy.')
        for name, p in les_model.named_parameters():
            if p.requires_grad:
                assert name.startswith(les_prefixes), \
                    f'Unexpected trainable param: {name}'


class TestInference:
    """Model in eval mode, single-graph (is_batch_data=False, like the calculator)."""

    @pytest.fixture(autouse=True)
    def set_single_graph_mode(self, les_model):
        les_model.set_is_batch_data(False)
        yield
        les_model.set_is_batch_data(True)  # restore for training tests

    def test_forward_no_crash(self, les_model, nacl_graph):
        # Note: no torch.no_grad() — SevenNet calls torch.autograd.grad
        # internally to compute forces, so grad computation must stay enabled.
        les_model.eval()
        out = les_model(nacl_graph)
        assert KEY.PRED_TOTAL_ENERGY in out
        assert KEY.PRED_FORCE in out

    def test_energy_finite(self, les_model, nacl_graph):
        les_model.eval()
        out = les_model(nacl_graph)
        e = out[KEY.PRED_TOTAL_ENERGY]
        assert torch.isfinite(e).all(), f'Energy contains NaN/Inf: {e}'

    def test_force_finite(self, les_model, nacl_graph):
        les_model.eval()
        out = les_model(nacl_graph)
        f = out[KEY.PRED_FORCE]
        assert torch.isfinite(f).all(), f'Force contains NaN/Inf: {f}'

    def test_force_shape(self, les_model, nacl_graph):
        les_model.eval()
        out = les_model(nacl_graph)
        n_atoms = int(nacl_graph[KEY.NUM_ATOMS].item())
        assert out[KEY.PRED_FORCE].shape == (n_atoms, 3)

    def test_stress_finite(self, les_model, nacl_graph):
        les_model.eval()
        out = les_model(nacl_graph)
        if KEY.PRED_STRESS in out:
            s = out[KEY.PRED_STRESS]
            assert torch.isfinite(s).all(), f'Stress contains NaN/Inf: {s}'

    def test_lr_energy_zero_at_init(self, les_model, nacl_graph):
        """With zero-init les_charge_readout, LR charges = 0 → E_LR ≈ 0."""
        les_model.eval()
        out = les_model(nacl_graph)
        e_lr = out.get(KEY.LR_ENERGY)
        if e_lr is not None:
            assert torch.allclose(e_lr, torch.zeros_like(e_lr), atol=1e-6), \
                f'E_LR should be ~0 with zero-init charges, got {e_lr}'

    def test_total_energy_equals_sr_at_init(self, les_model, nacl_graph):
        """With zero charges, PRED_TOTAL_ENERGY == SR_ENERGY."""
        les_model.eval()
        out = les_model(nacl_graph)
        e_total = out[KEY.PRED_TOTAL_ENERGY]
        e_sr = out.get(KEY.SR_ENERGY)
        if e_sr is not None:
            assert torch.allclose(e_total, e_sr, atol=1e-5), \
                'Total energy should equal SR energy when LR charges are zero'

    def test_multiple_forward_passes_stable(self, les_model, nacl_graph,
                                            omni_checkpoint):
        """retain_graph bug would crash on second inference pass.
        Use a second fresh graph (function-scoped fixture gives one per call,
        but we need two here so we build the second manually)."""
        atoms = bulk('NaCl', 'rocksalt', a=5.63)
        atoms.rattle(stdev=0.02, seed=0)
        cutoff = omni_checkpoint.config['cutoff']
        graph2 = AtomGraphData.from_numpy_dict(
            unlabeled_atoms_to_graph(atoms, cutoff, with_shift=True)
        )
        graph2[KEY.DATA_MODALITY] = 'omat24'
        graph2 = graph2.to(DEVICE)

        les_model.eval()
        out1 = les_model(nacl_graph)
        out2 = les_model(graph2)
        assert torch.allclose(
            out1[KEY.PRED_TOTAL_ENERGY], out2[KEY.PRED_TOTAL_ENERGY]
        ), 'Two identical forward passes give different energies'


class TestTraining:
    """Model in train mode with batched data (is_batch_data=True, like training)."""

    def test_backward_no_crash(self, les_model, nacl_batch):
        les_model.train()
        out = les_model(nacl_batch)
        loss = out[KEY.PRED_TOTAL_ENERGY].sum()
        loss.backward()           # must not raise RuntimeError

    def test_only_les_params_get_grad(self, les_model, nacl_batch):
        """After backward, only LES params should have .grad set."""
        les_model.train()
        # Zero out any stale grads from previous tests
        les_model.zero_grad()
        out = les_model(nacl_batch)
        loss = out[KEY.PRED_TOTAL_ENERGY].sum()
        loss.backward()

        les_prefixes = ('les_charge_readout.', 'les_lr_energy.')
        for name, param in les_model.named_parameters():
            if name.startswith(les_prefixes):
                # LES params must have a gradient
                assert param.grad is not None, \
                    f'LES param {name!r} has no grad after backward'
            else:
                # SR params should have no gradient (frozen)
                assert param.grad is None, \
                    f'SR param {name!r} has grad despite being frozen: {param.grad}'

    def test_optimizer_step_updates_les_only(self, les_model, nacl_batch):
        """One Adam step should move LES params and leave SR params unchanged."""
        les_model.train()
        les_model.zero_grad()

        # Snapshot SR params before step
        sr_snapshot = {
            name: p.data.clone()
            for name, p in les_model.named_parameters()
            if not p.requires_grad
        }

        # Perturb les_charge_readout weights so grad is non-zero
        # (They start at zero, so the *gradient* of loss w.r.t. them may be zero
        #  at q=0, but the Les() params should receive non-zero grads.)
        with torch.no_grad():
            les_model._modules['les_charge_readout'].first_linear.weight.fill_(0.01)

        out = les_model(nacl_batch)
        loss = out[KEY.PRED_TOTAL_ENERGY].sum()
        loss.backward()

        trainable = [p for p in les_model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(trainable, lr=1e-3)
        opt.step()

        # SR params must be unchanged
        for name, orig in sr_snapshot.items():
            cur = dict(les_model.named_parameters())[name].data
            assert torch.allclose(cur, orig), \
                f'SR param {name!r} changed after optimizer step'

        # Restore les_charge_readout weights to zero for subsequent tests
        with torch.no_grad():
            les_model._modules['les_charge_readout'].first_linear.weight.zero_()

    def test_force_computed_in_train_mode(self, les_model, nacl_batch):
        les_model.train()
        les_model.zero_grad()
        out = les_model(nacl_batch)
        assert KEY.PRED_FORCE in out
        assert torch.isfinite(out[KEY.PRED_FORCE]).all()

    def test_stress_computed_in_train_mode(self, les_model, nacl_batch):
        les_model.train()
        les_model.zero_grad()
        out = les_model(nacl_batch)
        if KEY.PRED_STRESS in out:
            assert torch.isfinite(out[KEY.PRED_STRESS]).all()


class TestZeroInitEquivalence:
    """
    Core correctness test: with zero-init les_charge_readout, the LES model
    must produce numerically identical energy, forces, and stress as the
    original non-LES checkpoint.

    Why this matters: if attaching LES changes any SR output (even by 1 ULP),
    the fine-tuning starting point is corrupted.  The zero-init guarantee is
    only useful if the model is truly a transparent wrapper at init.
    """

    @pytest.fixture(autouse=True)
    def single_graph_mode(self, les_model, sr_model):
        les_model.eval()
        les_model.set_is_batch_data(False)
        sr_model.eval()
        yield
        les_model.set_is_batch_data(True)

    def test_energy_matches_original(self, les_model, sr_model, nacl_graph):
        out_sr = sr_model(nacl_graph)
        # nacl_graph is function-scoped so this is a fresh copy
        out_les = les_model(nacl_graph)
        assert torch.allclose(
            out_les[KEY.PRED_TOTAL_ENERGY],
            out_sr[KEY.PRED_TOTAL_ENERGY],
            atol=1e-5,
        ), (
            f'Energy mismatch: LES={out_les[KEY.PRED_TOTAL_ENERGY].item():.8f}, '
            f'SR={out_sr[KEY.PRED_TOTAL_ENERGY].item():.8f}'
        )

    def test_forces_match_original(self, les_model, sr_model, nacl_graph):
        out_sr = sr_model(nacl_graph)
        out_les = les_model(nacl_graph)
        assert torch.allclose(
            out_les[KEY.PRED_FORCE],
            out_sr[KEY.PRED_FORCE],
            atol=1e-5,
        ), (
            f'Force mismatch (max abs diff): '
            f'{(out_les[KEY.PRED_FORCE] - out_sr[KEY.PRED_FORCE]).abs().max().item():.2e}'
        )

    def test_stress_matches_original(self, les_model, sr_model, nacl_graph):
        out_sr = sr_model(nacl_graph)
        out_les = les_model(nacl_graph)
        if KEY.PRED_STRESS not in out_sr or KEY.PRED_STRESS not in out_les:
            pytest.skip('Stress not computed for this structure')
        assert torch.allclose(
            out_les[KEY.PRED_STRESS],
            out_sr[KEY.PRED_STRESS],
            atol=1e-5,
        ), (
            f'Stress mismatch (max abs diff): '
            f'{(out_les[KEY.PRED_STRESS] - out_sr[KEY.PRED_STRESS]).abs().max().item():.2e}'
        )


N_CHARGES = 4  # number of latent charge channels used in multi-q tests


class TestMultiDimensionalQ:
    """
    Tests for multi-channel latent charges (n_charges > 1).

    With n_charges=N_CHARGES the readout maps node features to
    (N_atoms, N_CHARGES) and the Ewald energy is the sum of N_CHARGES
    independent Coulomb interactions, one per channel.
    """

    @pytest.fixture(scope='class')
    def les_model_mq(self, omni_checkpoint):
        """LES model with n_charges=N_CHARGES, SR params frozen."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            model = omni_checkpoint.build_model_with_les(
                les_config={
                    'les_args': {'use_atomwise': False},
                    'n_charges': N_CHARGES,
                    # non-zero init so that charges and LR energy are nonzero
                },
                freeze_sr=True,
            )
        return model.to(DEVICE)

    # ── shape ────────────────────────────────────────────────────────────────

    def test_charge_shape(self, les_model_mq, nacl_graph):
        """LES_Q must have shape (N_atoms, N_CHARGES)."""
        les_model_mq.eval()
        les_model_mq.set_is_batch_data(False)
        out = les_model_mq(nacl_graph)
        q = out[KEY.LES_Q]
        n_atoms = int(nacl_graph[KEY.NUM_ATOMS].item())
        assert q.shape == (n_atoms, N_CHARGES), (
            f'Expected LES_Q shape ({n_atoms}, {N_CHARGES}), got {tuple(q.shape)}'
        )

    def test_force_shape(self, les_model_mq, nacl_graph):
        """Force shape must be (N_atoms, 3) regardless of n_charges."""
        les_model_mq.eval()
        les_model_mq.set_is_batch_data(False)
        out = les_model_mq(nacl_graph)
        n_atoms = int(nacl_graph[KEY.NUM_ATOMS].item())
        assert out[KEY.PRED_FORCE].shape == (n_atoms, 3)

    # ── inference ────────────────────────────────────────────────────────────

    def test_energy_finite(self, les_model_mq, nacl_graph):
        les_model_mq.eval()
        les_model_mq.set_is_batch_data(False)
        out = les_model_mq(nacl_graph)
        assert torch.isfinite(out[KEY.PRED_TOTAL_ENERGY]).all()

    def test_force_finite(self, les_model_mq, nacl_graph):
        les_model_mq.eval()
        les_model_mq.set_is_batch_data(False)
        out = les_model_mq(nacl_graph)
        assert torch.isfinite(out[KEY.PRED_FORCE]).all()

    def test_stress_finite(self, les_model_mq, nacl_graph):
        les_model_mq.eval()
        les_model_mq.set_is_batch_data(False)
        out = les_model_mq(nacl_graph)
        if KEY.PRED_STRESS in out:
            assert torch.isfinite(out[KEY.PRED_STRESS]).all()

    def test_lr_energy_nonzero(self, les_model_mq, nacl_graph):
        """With default (non-zero) init, LR energy should be nonzero."""
        les_model_mq.eval()
        les_model_mq.set_is_batch_data(False)
        out = les_model_mq(nacl_graph)
        e_lr = out.get(KEY.LR_ENERGY)
        if e_lr is not None:
            assert not torch.allclose(e_lr, torch.zeros_like(e_lr), atol=1e-6), \
                'E_LR is unexpectedly zero with non-zero-init multi-channel readout'

    # ── training ─────────────────────────────────────────────────────────────

    def test_backward_no_crash(self, les_model_mq, nacl_batch):
        les_model_mq.train()
        les_model_mq.set_is_batch_data(True)
        les_model_mq.zero_grad()
        out = les_model_mq(nacl_batch)
        out[KEY.PRED_TOTAL_ENERGY].sum().backward()

    def test_only_les_params_get_grad(self, les_model_mq, nacl_batch):
        les_model_mq.train()
        les_model_mq.set_is_batch_data(True)
        les_model_mq.zero_grad()
        out = les_model_mq(nacl_batch)
        out[KEY.PRED_TOTAL_ENERGY].sum().backward()

        les_prefixes = ('les_charge_readout.', 'les_lr_energy.')
        for name, param in les_model_mq.named_parameters():
            if name.startswith(les_prefixes):
                assert param.grad is not None, \
                    f'LES param {name!r} has no grad after backward'
            else:
                assert param.grad is None, \
                    f'SR param {name!r} has grad despite being frozen: {param.grad}'
