"""
Unit tests for the LES-legacy architecture.

LES-legacy specific design under test:
  - EdgePreprocess (first model layer, is_stress=True) restores the
    pos -> EDGE_VEC autograd connection so that ForceStressOutput
    captures all gradient paths through a single _strain leaf.
  - ForceStressOutput (positional + strain gradient, not edge virial)
  - LatentEwaldSum reads strained pos/cell written back by EdgePreprocess

All tests run on CPU with a small model built from scratch.
The 'les' package (https://github.com/ChengUCB/les) must be installed.
"""

import pytest
import torch
from ase.build import bulk
from torch_geometric.data import Batch

import sevenn._keys as KEY
import sevenn.train.dataload as dl
from sevenn.atom_graph_data import AtomGraphData
from sevenn.model_build import build_E3_equivariant_model
from sevenn.nn.edge_embedding import EdgePreprocess
from sevenn.nn.force_output import ForceStressOutput
from sevenn.util import chemical_species_preprocess

# ── skip if les not installed ─────────────────────────────────────────────────

try:
    import les as _les_pkg  # noqa: F401
    HAS_LES = True
except ImportError:
    HAS_LES = False

pytestmark = pytest.mark.skipif(not HAS_LES, reason='les package not installed')

# ── constants ─────────────────────────────────────────────────────────────────

CUTOFF = 4.0
DELTA = 5e-4   # Angstrom, perturbation for numerical gradient checks
ATOL_FD = 1e-2  # tolerance for FD vs autograd comparison


# ── config helpers ────────────────────────────────────────────────────────────

def _base_config():
    """Minimal SevenNet config for fast CPU testing."""
    config = {
        'cutoff': CUTOFF,
        'channel': 4,
        'radial_basis': {'radial_basis_name': 'bessel'},
        'cutoff_function': {'cutoff_function_name': 'poly_cut'},
        'interaction_type': 'nequip',
        'lmax': 1,
        'is_parity': True,
        'num_convolution_layer': 2,
        'weight_nn_hidden_neurons': [16],
        'act_radial': 'silu',
        'act_scalar': {'e': 'silu', 'o': 'tanh'},
        'act_gate': {'e': 'silu', 'o': 'tanh'},
        'conv_denominator': 10.0,
        'train_denominator': False,
        'self_connection_type': 'nequip',
        'shift': 0.0,
        'scale': 1.0,
        'train_shift_scale': False,
        'irreps_manual': False,
        'lmax_edge': -1,
        'lmax_node': -1,
        'readout_as_fcn': False,
        'use_bias_in_linear': False,
        '_normalize_sph': True,
    }
    config.update(**chemical_species_preprocess(['Na', 'Cl']))
    return config


def _les_config(zero_init=True, n_charges=1):
    cfg = _base_config()
    cfg['use_les'] = True
    cfg['les_config'] = {
        'les_args': {'use_atomwise': False},
        'n_charges': n_charges,
        'zero_init': zero_init,
    }
    return cfg


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def nacl_atoms():
    atoms = bulk('NaCl', 'rocksalt', a=5.63)
    atoms.rattle(stdev=0.01, seed=42)
    return atoms


@pytest.fixture(scope='module')
def nacl_graph(nacl_atoms):
    """NaCl AtomGraphData with cell info for Ewald summation."""
    return AtomGraphData.from_numpy_dict(
        dl.unlabeled_atoms_to_graph(nacl_atoms, CUTOFF, with_shift=True)
    )


@pytest.fixture(scope='module')
def les_model_zero():
    """LES model, zero-init charges -> E_LR = 0 at construction."""
    return build_E3_equivariant_model(_les_config(zero_init=True), parallel=False)


@pytest.fixture(scope='module')
def les_model():
    """LES model with non-zero charges -> E_LR != 0."""
    return build_E3_equivariant_model(_les_config(zero_init=False), parallel=False)


# ── graph helpers ──────────────────────────────────────────────────────────────

def _fresh(graph):
    """Clone graph for a single forward pass (EdgePreprocess mutates data)."""
    return graph.clone()


def _run(model, graph, batch=False):
    model.eval()
    model.set_is_batch_data(batch)
    return model(_fresh(graph))


def _energy(model, graph):
    return _run(model, graph)[KEY.PRED_TOTAL_ENERGY].item()


def _perturbed(graph, atom_idx, direction, delta):
    """Graph with one Cartesian coordinate shifted by delta."""
    g = _fresh(graph)
    pos = g[KEY.POS].clone()
    pos[atom_idx, direction] += delta
    g[KEY.POS] = pos
    return g


def _strained(graph, alpha, beta, delta):
    """
    Graph with symmetric strain delta applied to the (alpha, beta) component.

    Applies new_pos = pos + pos @ eps_sym, new_cell = cell + cell @ eps_sym.
    CELL_SHIFT (integer PBC shifts) is unchanged for small delta.
    CELL_VOLUME is updated for correct stress normalisation.
    """
    g = _fresh(graph)
    pos = g[KEY.POS].clone().float()
    cell = g[KEY.CELL].view(3, 3).clone().float()

    eps = torch.zeros(3, 3)
    eps[alpha, beta] += 0.5 * delta
    eps[beta, alpha] += 0.5 * delta  # symmetric

    g[KEY.POS] = pos + pos @ eps
    new_cell = cell + cell @ eps
    g[KEY.CELL] = new_cell
    g[KEY.CELL_VOLUME] = torch.det(new_cell).abs()
    return g


# ── architecture tests ─────────────────────────────────────────────────────────

class TestLESLegacyArchitecture:
    """Verify that build_E3_equivariant_model produces the correct layer structure."""

    def test_edge_preprocess_is_first_layer(self, les_model):
        first_name, first_mod = next(iter(les_model._modules.items()))
        assert first_name == 'edge_preprocess'
        assert isinstance(first_mod, EdgePreprocess)

    def test_edge_preprocess_is_stress_true(self, les_model):
        assert les_model._modules['edge_preprocess'].is_stress is True

    def test_force_output_is_ForceStressOutput(self, les_model):
        fo = les_model._modules['force_output']
        assert isinstance(fo, ForceStressOutput)

    def test_les_modules_present(self, les_model):
        names = set(les_model._modules.keys())
        for expected in ('les_charge_readout', 'les_lr_energy', 'add_lr_to_total'):
            assert expected in names, f'Missing module: {expected}'

    def test_sr_energy_reduce_present(self, les_model):
        # LES uses reduce_sr_energy, not the non-LES reduce_total_enegy
        assert 'reduce_sr_energy' in les_model._modules
        assert 'reduce_total_enegy' not in les_model._modules

    def test_parallel_raises(self):
        with pytest.raises(NotImplementedError):
            build_E3_equivariant_model(_les_config(), parallel=True)

    def test_strain_leaf_created_during_forward(self, les_model, nacl_graph):
        """EdgePreprocess must write _strain to data on every forward pass."""
        les_model.eval()
        les_model.set_is_batch_data(False)
        g = _fresh(nacl_graph)
        les_model(g)
        assert '_strain' in g, '_strain leaf not found in data after forward'
        # Must be a true autograd leaf so that d(E)/d(_strain) is well-defined.
        assert g['_strain'].requires_grad
        assert g['_strain'].is_leaf

    def test_is_batch_data_propagates(self, les_model):
        """set_is_batch_data must reach top-level LES modules and EdgePreprocess."""
        les_model.set_is_batch_data(False)
        for name in ('edge_preprocess', 'les_lr_energy', 'force_output'):
            assert les_model._modules[name]._is_batch_data is False, (
                f'{name}._is_batch_data not set to False'
            )

        les_model.set_is_batch_data(True)
        for name in ('edge_preprocess', 'les_lr_energy', 'force_output'):
            assert les_model._modules[name]._is_batch_data is True, (
                f'{name}._is_batch_data not restored to True'
            )

    def test_state_dict_round_trip(self):
        """save → load → forward must reproduce original output bit-for-bit."""
        model1 = build_E3_equivariant_model(
            _les_config(zero_init=False), parallel=False
        )
        model2 = build_E3_equivariant_model(
            _les_config(zero_init=False), parallel=False
        )
        # Different random init → outputs would normally differ.
        sd = model1.state_dict()
        missing, unexpected = model2.load_state_dict(sd, strict=True)
        assert not missing, f'Unexpected missing keys: {missing}'
        assert not unexpected, f'Unexpected extra keys: {unexpected}'

        atoms = bulk('NaCl', 'rocksalt', a=5.63)
        graph = AtomGraphData.from_numpy_dict(
            dl.unlabeled_atoms_to_graph(atoms, CUTOFF, with_shift=True)
        )
        for m in (model1, model2):
            m.eval()
            m.set_is_batch_data(False)
        out1 = model1(_fresh(graph))
        out2 = model2(_fresh(graph))
        assert torch.allclose(
            out1[KEY.PRED_TOTAL_ENERGY], out2[KEY.PRED_TOTAL_ENERGY], atol=1e-6
        )
        assert torch.allclose(out1[KEY.PRED_FORCE], out2[KEY.PRED_FORCE], atol=1e-6)
        assert torch.allclose(
            out1[KEY.PRED_STRESS], out2[KEY.PRED_STRESS], atol=1e-6
        )


# ── non-batch inference ────────────────────────────────────────────────────────

class TestNonBatchInference:

    @pytest.fixture(autouse=True)
    def setup(self, les_model):
        les_model.eval()
        les_model.set_is_batch_data(False)

    def test_energy_finite(self, les_model, nacl_graph):
        out = _run(les_model, nacl_graph)
        assert torch.isfinite(out[KEY.PRED_TOTAL_ENERGY])

    def test_force_finite(self, les_model, nacl_graph):
        out = _run(les_model, nacl_graph)
        assert torch.isfinite(out[KEY.PRED_FORCE]).all()

    def test_stress_finite(self, les_model, nacl_graph):
        out = _run(les_model, nacl_graph)
        assert torch.isfinite(out[KEY.PRED_STRESS]).all()

    def test_energy_shape(self, les_model, nacl_graph):
        out = _run(les_model, nacl_graph)
        assert out[KEY.PRED_TOTAL_ENERGY].shape == ()

    def test_force_shape(self, les_model, nacl_graph):
        out = _run(les_model, nacl_graph)
        n = int(nacl_graph[KEY.NUM_ATOMS].item())
        assert out[KEY.PRED_FORCE].shape == (n, 3)

    def test_stress_shape(self, les_model, nacl_graph):
        out = _run(les_model, nacl_graph)
        assert out[KEY.PRED_STRESS].shape == (6,)

    def test_total_equals_sr_plus_lr(self, les_model, nacl_graph):
        out = _run(les_model, nacl_graph)
        assert torch.allclose(
            out[KEY.PRED_TOTAL_ENERGY],
            out[KEY.SR_ENERGY] + out[KEY.LR_ENERGY],
            atol=1e-6,
        )

    def test_lr_energy_is_scalar(self, les_model, nacl_graph):
        """Non-batch LR_ENERGY must be a scalar tensor so AddLREnergy doesn't
        accidentally broadcast a (1,) tensor onto the scalar SR_ENERGY."""
        out = _run(les_model, nacl_graph)
        assert out[KEY.LR_ENERGY].shape == (), (
            f'LR_ENERGY should be a scalar, got shape {tuple(out[KEY.LR_ENERGY].shape)}'
        )
        assert out[KEY.SR_ENERGY].shape == ()


# ── batch inference ────────────────────────────────────────────────────────────

class TestBatchInference:

    @pytest.fixture
    def batch(self, nacl_graph):
        return Batch.from_data_list([_fresh(nacl_graph), _fresh(nacl_graph)])

    @pytest.fixture(autouse=True)
    def setup(self, les_model):
        les_model.eval()
        les_model.set_is_batch_data(True)

    def test_energy_shape(self, les_model, batch):
        out = les_model(batch)
        assert out[KEY.PRED_TOTAL_ENERGY].shape == (2,)

    def test_force_shape(self, les_model, batch, nacl_graph):
        out = les_model(batch)
        n = int(nacl_graph[KEY.NUM_ATOMS].item())
        assert out[KEY.PRED_FORCE].shape == (2 * n, 3)

    def test_stress_shape(self, les_model, batch):
        out = les_model(batch)
        assert out[KEY.PRED_STRESS].shape == (2, 6)

    def test_all_finite(self, les_model, batch):
        out = les_model(batch)
        for key in (KEY.PRED_TOTAL_ENERGY, KEY.PRED_FORCE, KEY.PRED_STRESS):
            assert torch.isfinite(out[key]).all(), f'{key} contains NaN/Inf'


# ── batch == sequential consistency ───────────────────────────────────────────

class TestBatchConsistency:
    """Batch output must match running the same graph twice in non-batch mode."""

    def _seq_outputs(self, model, graph):
        model.eval()
        model.set_is_batch_data(False)
        o1 = _run(model, graph)
        o2 = _run(model, graph)
        return o1, o2

    def _batch_output(self, model, graph):
        model.eval()
        model.set_is_batch_data(True)
        batch = Batch.from_data_list([_fresh(graph), _fresh(graph)])
        return model(batch)

    def test_energy_consistent(self, les_model, nacl_graph):
        o1, o2 = self._seq_outputs(les_model, nacl_graph)
        ob = self._batch_output(les_model, nacl_graph)
        e_seq = torch.stack([o1[KEY.PRED_TOTAL_ENERGY], o2[KEY.PRED_TOTAL_ENERGY]])
        assert torch.allclose(e_seq, ob[KEY.PRED_TOTAL_ENERGY], atol=1e-5)

    def test_force_consistent(self, les_model, nacl_graph):
        o1, o2 = self._seq_outputs(les_model, nacl_graph)
        ob = self._batch_output(les_model, nacl_graph)
        f_seq = torch.cat([o1[KEY.PRED_FORCE], o2[KEY.PRED_FORCE]])
        assert torch.allclose(f_seq, ob[KEY.PRED_FORCE], atol=1e-5)

    def test_stress_consistent(self, les_model, nacl_graph):
        o1, o2 = self._seq_outputs(les_model, nacl_graph)
        ob = self._batch_output(les_model, nacl_graph)
        s_seq = torch.stack([o1[KEY.PRED_STRESS], o2[KEY.PRED_STRESS]])
        assert torch.allclose(s_seq, ob[KEY.PRED_STRESS], atol=1e-5)


# ── training backward ──────────────────────────────────────────────────────────

class TestTraining:

    @pytest.fixture(autouse=True)
    def setup(self, les_model):
        les_model.train()
        les_model.set_is_batch_data(False)
        les_model.zero_grad()

    def test_backward_energy(self, les_model, nacl_graph):
        out = les_model(_fresh(nacl_graph))
        out[KEY.PRED_TOTAL_ENERGY].sum().backward()

    def test_backward_force(self, les_model, nacl_graph):
        """Force loss requires create_graph=True inside ForceStressOutput."""
        out = les_model(_fresh(nacl_graph))
        loss = out[KEY.PRED_TOTAL_ENERGY].sum() + out[KEY.PRED_FORCE].sum()
        loss.backward()

    def test_backward_stress(self, les_model, nacl_graph):
        out = les_model(_fresh(nacl_graph))
        loss = (out[KEY.PRED_TOTAL_ENERGY].sum()
                + out[KEY.PRED_FORCE].sum()
                + out[KEY.PRED_STRESS].sum())
        loss.backward()

    def test_params_receive_grad(self, les_model, nacl_graph):
        out = les_model(_fresh(nacl_graph))
        (out[KEY.PRED_TOTAL_ENERGY].sum() + out[KEY.PRED_FORCE].sum()).backward()
        params_with_grad = [n for n, p in les_model.named_parameters()
                            if p.grad is not None and p.grad.abs().max() > 0]
        assert len(params_with_grad) > 0, 'No parameters received a non-zero gradient'


# ── numerical gradient: forces ─────────────────────────────────────────────────

class TestNumericalForce:
    """
    PRED_FORCE must equal -dE/d(pos) via central finite differences.

    This validates that EdgePreprocess correctly restores the pos -> EDGE_VEC
    autograd connection so ForceStressOutput captures all gradient paths
    (SR through EDGE_VEC chain + LR direct from les()) in one call.
    """

    @pytest.fixture(autouse=True)
    def setup(self, les_model):
        les_model.eval()
        les_model.set_is_batch_data(False)

    def _fd_force(self, model, graph, atom_idx, direction):
        """Central-difference: -(E(pos+δ) - E(pos-δ)) / 2δ."""
        ep = model(_perturbed(graph, atom_idx, direction, +DELTA))[KEY.PRED_TOTAL_ENERGY].item()
        em = model(_perturbed(graph, atom_idx, direction, -DELTA))[KEY.PRED_TOTAL_ENERGY].item()
        return -(ep - em) / (2 * DELTA)

    @pytest.mark.parametrize('atom_idx,direction', [(0, 0), (0, 1), (1, 2)])
    def test_force_vs_fd(self, les_model, nacl_graph, atom_idx, direction):
        fd = self._fd_force(les_model, nacl_graph, atom_idx, direction)
        f_model = _run(les_model, nacl_graph)[KEY.PRED_FORCE][atom_idx, direction].item()
        assert abs(fd - f_model) < ATOL_FD, (
            f'Force mismatch at atom {atom_idx} dir {direction}: '
            f'FD={fd:.6f}  model={f_model:.6f}'
        )


# ── numerical gradient: stress ─────────────────────────────────────────────────

class TestNumericalStress:
    """
    PRED_STRESS must equal -(1/V) dE/dε via central finite differences.

    The _strain leaf created by EdgePreprocess must capture SR virial +
    LR positional + LR cell contributions so that the total stress is
    reproduced by a single d(E)/d(_strain) call.
    """

    # Voigt index -> (alpha, beta)
    VOIGT = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]

    @pytest.fixture(autouse=True)
    def setup(self, les_model):
        les_model.eval()
        les_model.set_is_batch_data(False)

    def _fd_stress(self, model, graph, voigt_idx):
        """Central-difference stress: -(E(+δ) - E(-δ)) / (2δ V_0)."""
        alpha, beta = self.VOIGT[voigt_idx]
        ep = model(_strained(graph, alpha, beta, +DELTA))[KEY.PRED_TOTAL_ENERGY].item()
        em = model(_strained(graph, alpha, beta, -DELTA))[KEY.PRED_TOTAL_ENERGY].item()
        vol = graph[KEY.CELL_VOLUME].item()
        return -(ep - em) / (2 * DELTA * vol)

    @pytest.mark.parametrize('voigt_idx', [0, 1, 2])  # diagonal (xx, yy, zz)
    def test_stress_vs_fd(self, les_model, nacl_graph, voigt_idx):
        fd = self._fd_stress(les_model, nacl_graph, voigt_idx)
        s_model = _run(les_model, nacl_graph)[KEY.PRED_STRESS][voigt_idx].item()
        assert abs(fd - s_model) < ATOL_FD, (
            f'Stress mismatch at Voigt index {voigt_idx}: '
            f'FD={fd:.6f}  model={s_model:.6f}'
        )


# ── SR isolation: zero-init ────────────────────────────────────────────────────

class TestSRIsolation:
    """
    With zero-init charges (LES_Q = 0 -> E_LR = 0), PRED_TOTAL_ENERGY
    must equal SR_ENERGY.  Validates that LES adds no spurious energy at init.
    """

    @pytest.fixture(autouse=True)
    def setup(self, les_model_zero):
        les_model_zero.eval()
        les_model_zero.set_is_batch_data(False)

    def test_lr_energy_zero(self, les_model_zero, nacl_graph):
        out = _run(les_model_zero, nacl_graph)
        e_lr = out[KEY.LR_ENERGY]
        assert torch.allclose(e_lr, torch.zeros_like(e_lr), atol=1e-6), \
            f'E_LR should be 0 with zero-init charges, got {e_lr.item()}'

    def test_total_equals_sr(self, les_model_zero, nacl_graph):
        out = _run(les_model_zero, nacl_graph)
        assert torch.allclose(out[KEY.PRED_TOTAL_ENERGY], out[KEY.SR_ENERGY], atol=1e-6)


# ── LR contribution to force/stress ────────────────────────────────────────────

class TestLRContribution:
    """
    With non-zero charges, the LR Ewald term must produce non-trivial
    contributions to forces and stress. Compares against a zero-init
    (SR-only) model that uses the same SR weights.

    Catches bugs where the LR gradient path is silently broken (e.g.
    LatentEwaldSum not connected to the autograd graph, or pos write-back
    from EdgePreprocess missing).
    """

    @pytest.fixture(scope='class')
    def shared_config(self):
        return _les_config(zero_init=False, n_charges=1)

    @pytest.fixture(scope='class')
    def les_model_lr(self, shared_config):
        return build_E3_equivariant_model(shared_config, parallel=False)

    @pytest.fixture(scope='class')
    def les_model_sr_only(self, shared_config, les_model_lr):
        """Same SR weights as les_model_lr, but charges forced to zero."""
        cfg = dict(shared_config)
        cfg['les_config'] = {**cfg['les_config'], 'zero_init': True}
        m = build_E3_equivariant_model(cfg, parallel=False)
        # Copy SR weights so only the LR term differs.
        src_sd = les_model_lr.state_dict()
        dst_sd = m.state_dict()
        for k in dst_sd:
            if not k.startswith(('les_charge_readout.', 'les_lr_energy.')):
                dst_sd[k] = src_sd[k]
        m.load_state_dict(dst_sd, strict=True)
        return m

    def test_lr_changes_force(self, les_model_lr, les_model_sr_only, nacl_graph):
        for m in (les_model_lr, les_model_sr_only):
            m.eval()
            m.set_is_batch_data(False)
        f_lr = les_model_lr(_fresh(nacl_graph))[KEY.PRED_FORCE]
        f_sr = les_model_sr_only(_fresh(nacl_graph))[KEY.PRED_FORCE]
        diff = (f_lr - f_sr).abs().max().item()
        assert diff > 1e-5, (
            f'LR term should contribute non-trivially to forces; max diff = {diff:.2e}'
        )

    def test_lr_changes_stress(self, les_model_lr, les_model_sr_only, nacl_graph):
        for m in (les_model_lr, les_model_sr_only):
            m.eval()
            m.set_is_batch_data(False)
        s_lr = les_model_lr(_fresh(nacl_graph))[KEY.PRED_STRESS]
        s_sr = les_model_sr_only(_fresh(nacl_graph))[KEY.PRED_STRESS]
        diff = (s_lr - s_sr).abs().max().item()
        assert diff > 1e-5, (
            f'LR term should contribute non-trivially to stress; max diff = {diff:.2e}'
        )


# ── multi-charge channels ──────────────────────────────────────────────────────

class TestMultiCharge:
    """n_charges > 1: LES_Q shape and basic sanity."""

    N_CHARGES = 3

    @pytest.fixture(scope='class')
    def les_model_mq(self):
        return build_E3_equivariant_model(
            _les_config(zero_init=False, n_charges=self.N_CHARGES), parallel=False
        )

    def test_charge_shape(self, les_model_mq, nacl_graph):
        les_model_mq.eval()
        les_model_mq.set_is_batch_data(False)
        out = les_model_mq(_fresh(nacl_graph))
        n = int(nacl_graph[KEY.NUM_ATOMS].item())
        assert out[KEY.LES_Q].shape == (n, self.N_CHARGES)

    def test_energy_finite(self, les_model_mq, nacl_graph):
        les_model_mq.eval()
        les_model_mq.set_is_batch_data(False)
        out = les_model_mq(_fresh(nacl_graph))
        assert torch.isfinite(out[KEY.PRED_TOTAL_ENERGY])

    def test_force_shape(self, les_model_mq, nacl_graph):
        les_model_mq.eval()
        les_model_mq.set_is_batch_data(False)
        out = les_model_mq(_fresh(nacl_graph))
        n = int(nacl_graph[KEY.NUM_ATOMS].item())
        assert out[KEY.PRED_FORCE].shape == (n, 3)

    def test_backward_no_crash(self, les_model_mq, nacl_graph):
        les_model_mq.train()
        les_model_mq.set_is_batch_data(False)
        les_model_mq.zero_grad()
        out = les_model_mq(_fresh(nacl_graph))
        (out[KEY.PRED_TOTAL_ENERGY].sum() + out[KEY.PRED_FORCE].sum()).backward()


# ── Ewald summation physics ───────────────────────────────────────────────────

class TestEwaldPhysics:
    """
    Physics-level sanity checks on the Ewald sum itself.

    Tests that probe the analytic structure of Coulomb interactions:
      - translation invariance        (PBC handled correctly)
      - quadratic scaling in q        (E ~ Σ qᵢqⱼ / rᵢⱼ)
      - sign invariance E(-q) = E(q)  (q² symmetry)
      - zero-charge ⇒ zero-energy

    For the scaling/sign/zero tests, LatentEwaldSum is invoked directly with
    synthetic q to isolate Ewald physics from the GNN's charge readout.
    """

    @pytest.fixture(scope='class')
    def ewald_module(self):
        """Bare LatentEwaldSum in non-batch mode."""
        from sevenn.nn.les import LatentEwaldSum
        m = LatentEwaldSum()
        m._is_batch_data = False
        m.eval()
        return m

    @staticmethod
    def _ewald_data(graph, q):
        """Minimal data dict for direct LatentEwaldSum invocation."""
        return {
            KEY.POS: graph[KEY.POS].clone(),
            KEY.CELL: graph[KEY.CELL].clone(),
            KEY.LES_Q: q,
        }

    def test_translation_invariance(self, les_model, nacl_graph):
        """
        Uniform shift of all positions must leave E_LR unchanged.

        Under a pure translation Δ:
          - edge_vec = pos[dst] - pos[src] + S·cell is invariant (so SR q is too)
          - Ewald structure factor: |Σ qᵢ exp(i k·(rᵢ+Δ))|² = |S(k)|²  ✓
        Any failure here points to broken PBC handling inside Les().
        """
        les_model.eval()
        les_model.set_is_batch_data(False)

        e0 = les_model(_fresh(nacl_graph))[KEY.LR_ENERGY].detach().item()

        shifted = _fresh(nacl_graph)
        shifted[KEY.POS] = shifted[KEY.POS] + torch.tensor([0.3, -0.7, 1.1])
        e1 = les_model(shifted)[KEY.LR_ENERGY].detach().item()

        # Use a meaningful tolerance: energy scale of E_LR, not absolute.
        scale = max(abs(e0), 1e-6)
        assert abs(e0 - e1) / scale < 1e-3, (
            f'E_LR not translation-invariant: '
            f'E(pos) = {e0:.6e}, E(pos+Δ) = {e1:.6e}, ΔE/E = {(e0-e1)/scale:.3e}'
        )

    def test_quadratic_in_charge(self, ewald_module, nacl_graph):
        """E_LR(α·q) = α²·E_LR(q): Coulomb is bilinear in charges."""
        n = int(nacl_graph[KEY.NUM_ATOMS].item())
        q = torch.linspace(-0.4, 0.4, n).unsqueeze(-1)
        q = q - q.mean()  # enforce charge neutrality

        e1 = ewald_module(self._ewald_data(nacl_graph, q))[KEY.LR_ENERGY].item()

        alpha = 2.5
        e2 = ewald_module(
            self._ewald_data(nacl_graph, alpha * q)
        )[KEY.LR_ENERGY].item()

        expected = alpha ** 2 * e1
        rel_err = abs(e2 - expected) / max(abs(expected), 1e-12)
        assert rel_err < 1e-4, (
            f'Ewald not quadratic in q: E(αq) = {e2:.6e}, '
            f'α²·E(q) = {expected:.6e}, rel_err = {rel_err:.3e}'
        )

    def test_sign_invariance(self, ewald_module, nacl_graph):
        """E_LR(-q) = E_LR(q): Coulomb energy depends on qᵢqⱼ pairs."""
        n = int(nacl_graph[KEY.NUM_ATOMS].item())
        q = torch.linspace(-0.4, 0.4, n).unsqueeze(-1)
        q = q - q.mean()

        e_pos = ewald_module(self._ewald_data(nacl_graph, q))[KEY.LR_ENERGY].item()
        e_neg = ewald_module(self._ewald_data(nacl_graph, -q))[KEY.LR_ENERGY].item()

        scale = max(abs(e_pos), 1e-6)
        assert abs(e_pos - e_neg) / scale < 1e-5, (
            f'E(-q) ≠ E(q): {e_pos:.6e} vs {e_neg:.6e}'
        )

    def test_zero_charge_zero_energy(self, ewald_module, nacl_graph):
        """q = 0 must give E_LR = 0 (no charge → no Coulomb)."""
        n = int(nacl_graph[KEY.NUM_ATOMS].item())
        q = torch.zeros(n, 1)
        e = ewald_module(self._ewald_data(nacl_graph, q))[KEY.LR_ENERGY].item()
        assert abs(e) < 1e-10, f'E_LR(q=0) should be 0, got {e:.3e}'
