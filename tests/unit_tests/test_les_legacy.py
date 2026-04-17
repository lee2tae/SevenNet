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
        assert g['_strain'].requires_grad


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
