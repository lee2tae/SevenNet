"""
Real-space (cell-size independent) BEC for LES models; les_config: bec_mode: realspace.

Z*_i = q'_i I + sum_e (d_{i,n(e)} - d_{i,c(e)}) V_e + sum_j d mu'_j / d r_i
V_e  = sum_j (r_j - r_ref(e)) (x) d q'_j / d v_e,  v_e = r_n - r_c (n = EDGE_IDX[1] sends, c = EDGE_IDX[0] receives)
Multi-hop displacements are resolved per hop: each convolution after the first is re-run with its edge weights
scaled by -v_e and back-propagated through its node input only. Global scalars of the charge pipeline (neutralization,
q_projection) are frozen; gauge 'mid' puts their compensating charge at edge midpoints (exact ASR), 'iref' at the
displaced atom (the L -> inf limit of the phase-corrected LES BEC).
"""
from collections import OrderedDict

import torch

import sevenn._keys as KEY

from .convolution import IrrepsConvolution, message_gather
from .les_qu import FeatureTap

RS_Q_RAW = 'les_rs_q_raw'


def rs_conv_taps(block: OrderedDict, t: int) -> OrderedDict:
    out = OrderedDict()
    for k, m in block.items():
        if k == f'{t}_convolution':
            out[f'{t}_rs_tap_in'] = FeatureTap(KEY.NODE_FEATURE, f'_rs_x_in_{t}')
            out[k] = m
            out[f'{t}_rs_tap_out'] = FeatureTap(KEY.NODE_FEATURE, f'_rs_out_{t}')
        else:
            out[k] = m
    return out


def bind_realspace_bec(model):
    m = model._modules.get('les_bec')
    if m is not None and hasattr(m, '_parent'):
        m._parent = [model]
    return model


def _conv_apply(conv, x, filt, weight, edge_idx):
    src, dst = edge_idx[1], edge_idx[0]
    if isinstance(conv, IrrepsConvolution):
        out = message_gather(x, dst, conv.convolution(x[src], filt, weight))
    else:
        out = conv.convolution(x, filt, weight, src.to(torch.int32), dst.to(torch.int32))
    return out.div(conv.denominator)


def _seg(x, batch, n):
    return torch.zeros((n,) + x.shape[1:], device=x.device, dtype=x.dtype).index_add_(0, batch, x)


def _edge_to_atom(W, eidx, n_atoms):
    out = W.new_zeros((n_atoms,) + W.shape[1:])
    out.index_add_(0, eidx[1], W)
    out.index_add_(0, eidx[0], -W)
    return out


def realspace_bec_cls(base):
    class RealSpaceBEC(base):
        def __init__(self, conv_layers=(), neutralize_mode='none', gauge='mid', **kwargs):
            super().__init__(**kwargs)
            self.conv_layers = list(conv_layers)
            self.neutralize_mode = neutralize_mode
            self.gauge = gauge
            self._parent = []

        def _charges(self, data, batch, n_graphs):
            # q' and its position-local linearisation: dq'_j = a_j dq_raw_j + b_j df_j + c_j ds_j (global scalars
            # held fixed for d/dr only; a, b, c stay in the graph so parameter gradients are exact)
            q_raw = data[RS_Q_RAW]
            q_raw = q_raw.reshape(q_raw.shape[0], -1).sum(1)
            f, lam1 = None, None
            q = q_raw
            if self.neutralize_mode == 'fukui':
                f = torch.nn.functional.softplus(data[KEY.LES_F].reshape(-1))
                lam1 = (_seg(q_raw, batch, n_graphs) / (_seg(f, batch, n_graphs) + 1e-12))[batch]
                q = q_raw - lam1 * f
            elif self.neutralize_mode == 'shift':
                q = q_raw - (_seg(q_raw, batch, n_graphs) / _seg(torch.ones_like(q_raw), batch, n_graphs))[batch]
            s_loc = None
            if self.epsilon_mode == 'learned_atomic':
                s = s_loc = data[self.data_key_epsilon].reshape(-1).sqrt()
            elif self.epsilon_mode == 'learned':
                s = data[self.data_key_epsilon].reshape(-1).sqrt()[batch]
            else:
                s = torch.full_like(q, self.normalization_factor)
            qs = q * s
            a = s
            if self.remove_mean == 'q_projection':
                lam2 = (_seg(qs, batch, n_graphs) / _seg(q * q, batch, n_graphs).clamp(min=1e-8))[batch]
                qs = qs - lam2 * q * q
                a = s - 2 * lam2 * q
            elif self.remove_mean == 'uniform':
                qs = qs - (_seg(qs, batch, n_graphs) / _seg(torch.ones_like(qs), batch, n_graphs))[batch]
            outs, cots = [q_raw], [a]
            if f is not None:
                outs.append(f); cots.append(-lam1 * a)
            if s_loc is not None:
                outs.append(s_loc); cots.append(q)
            mu = data[KEY.LES_MU] if KEY.LES_MU in data else None
            return qs, outs, cots, mu, s, s_loc

        def forward(self, data):
            if self.compute_dipole:
                cb, self.compute_bec = self.compute_bec, False
                data = super().forward(data)
                self.compute_bec = cb
            if not self.compute_bec:
                return data
            v = data[KEY.EDGE_VEC]
            eidx = data[KEY.EDGE_IDX]
            n_atoms = data[KEY.POS].shape[0]
            if self._is_batch_data:
                batch = data[KEY.BATCH].long()
                n_graphs = int(batch.max().item()) + 1
            else:
                batch = torch.zeros(n_atoms, dtype=torch.long, device=v.device)
                n_graphs = 1
            qs, q_outs, q_cots, mu, s, s_loc = self._charges(data, batch, n_graphs)
            if KEY.LES_BEC_REF in data and bool(torch.isnan(data[KEY.LES_BEC_REF]).all()):
                z = v.new_zeros(n_atoms, 3, 3) + 0.0 * qs.sum()
                if mu is not None:
                    z = z + 0.0 * mu.sum()
                data[self.data_key_out] = z
                return data

            cg = self.training
            convs = [self._parent[0]._modules[f'{t}_convolution'] for t in self.conv_layers]
            outs_c = [data[f'_rs_out_{t}'] for t in self.conv_layers]
            grads = torch.autograd.grad(q_outs, [v] + outs_c, q_cots, retain_graph=True, create_graph=cg,
                                        allow_unused=True)
            g = torch.zeros_like(v) if grads[0] is None else grads[0]

            e3 = torch.cat([eidx + a * n_atoms for a in range(3)], dim=1)
            vw = torch.cat([-v[:, a].detach() for a in range(3)])[:, None]
            nb = 6 if mu is not None else 3
            outs, cots = [], []
            for t, conv, G in zip(self.conv_layers, convs, grads[1:]):
                if G is None:
                    continue
                x_in = data[f'_rs_x_in_{t}']
                x3 = x_in.detach().repeat(3, 1).requires_grad_(True)
                w3 = conv.weight_nn(data[conv.key_weight_input]).repeat(3, 1) * vw
                out3 = _conv_apply(conv, x3, data[conv.key_filter].repeat(3, 1), w3, e3)
                (H3,) = torch.autograd.grad(out3, x3, G.repeat(3, 1), create_graph=cg)
                c = H3.new_zeros((nb,) + x_in.shape)
                c[:3] = H3.view(3, n_atoms, -1)
                outs.append(x_in)
                cots.append(c)
            if mu is not None:                       # mu'_j = s_j mu_j
                c = mu.new_zeros((nb,) + mu.shape)
                c[3:] = torch.eye(3, device=v.device, dtype=v.dtype)[:, None, :] * s[None, :, None]
                outs.append(mu)
                cots.append(c)
                if s_loc is not None:
                    c = s_loc.new_zeros(nb, n_atoms)
                    c[3:] = mu.T
                    outs.append(s_loc)
                    cots.append(c)
            Gb = None
            if outs:
                (Gb,) = torch.autograd.grad(outs, v, cots, retain_graph=True, create_graph=cg,
                                            is_grads_batched=True, allow_unused=True)
            if Gb is None:
                Gb = v.new_zeros(nb, v.shape[0], 3)
            U = Gb[:3].permute(1, 0, 2)
            vd = v.detach()[:, :, None]
            if self.gauge == 'iref':
                R = v.new_zeros(n_atoms, 3, 3)
                R.index_add_(0, eidx[1], U - vd * g[:, None, :])
                R.index_add_(0, eidx[0], -U)
            else:
                R = _edge_to_atom(U - 0.5 * vd * g[:, None, :], eidx, n_atoms)
            z = qs[:, None, None] * torch.eye(3, device=v.device, dtype=v.dtype) + R
            if mu is not None:
                z = z + _edge_to_atom(Gb[3:].permute(1, 0, 2), eidx, n_atoms)
            if self.output_index is not None:
                z = z[:, self.output_index]
            data[self.data_key_out] = self._asr_project(z, batch, n_graphs)
            return data

    return RealSpaceBEC
