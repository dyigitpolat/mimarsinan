"""A Linear on the last axis of a rank-k per-sample input (k >= 3) is the rank-2 case with its leading axes flattened.

On the base the FC mapping lowered a per-sample input of rank 1, or rank 2 with the fan-in as an axis, and refused
every other (``PerceptronFanInError``): ConvNeXt's channels-last MLP (a Linear on ``(H, W, C)``), Swin's MLPs and patch
merging, MaxViT's MLP. ``PerceptronMapper`` now reshapes the per-sample sources to ``(prod(leading axes), fan-in)``,
maps them as the rank-2 case (``map_fc`` shares one weight bank over the positions) and reshapes the outputs back. The
MAC counters already count a Linear at every leading position (``module_macs``); the softcore flowchart's estimate
counts the positions as instances.
"""

from __future__ import annotations

import contextlib
import hashlib
import io

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING, SPIKING_PACKAGING
from mimarsinan.mapping.verification.onchip_fraction import estimate_onchip_fraction
from mimarsinan.mapping.verification.onchip_majority import count_host_ops
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.torch_mapping.converter import trace_and_convert
from operand_completeness.harness import (
    CONTRACTS,
    assert_converted_exact,
    assert_value_program_exact,
    native,
    quietly,
)

PACKAGINGS = {"mvm": MVM_PACKAGING, "spiking": SPIKING_PACKAGING}


class ConvNeXtBlock(nn.Module):
    """``flatten``: the same block with the Linears' leading axes flattened by hand (the rank-2 twin)."""

    def __init__(self, dim: int = 8, flatten: bool = False):
        super().__init__()
        self.flatten = flatten
        self.dw = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.pw2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), 0.5))

    def forward(self, x):
        y = self.norm(self.dw(x).permute(0, 2, 3, 1))
        if self.flatten:
            y = self.pw2(F.gelu(self.pw1(y.reshape(-1, 16, 8)))).reshape(-1, 4, 4, 8) * self.gamma
        else:
            y = self.pw2(F.gelu(self.pw1(y))) * self.gamma
        return x + y.permute(0, 3, 1, 2)


class TinyConvNeXt(nn.Module):
    """The round-12 reviewer's ``TinyConvNeXt`` (``r12_families.py``) on (1, 8, 8): the block runs on (8, 4, 4)."""

    def __init__(self, flatten: bool = False):
        super().__init__()
        self.stem = nn.Conv2d(1, 8, 2, 2)
        self.b = ConvNeXtBlock(flatten=flatten)
        self.head = nn.Linear(8, 10)

    def forward(self, x):
        return self.head(self.b(self.stem(x)).mean([-2, -1]))


class ChannelsLastMlp(nn.Module):
    """A rank-3 per-sample input (4, 4, 8): the first weight layer is a Linear on its last axis (an encoding layer)."""

    def __init__(self):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 5)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x))).mean(dim=(1, 2))


class ReluMlp(nn.Module):
    """Two Linears with a ReLU each on a (4, 4, 8) input, every one a perceptron under both contracts; ``flatten``: the
    leading axes flattened by hand (the rank-2 twin)."""

    def __init__(self, flatten: bool = False):
        super().__init__()
        self.flatten = flatten
        self.fc1, self.act1, self.fc2, self.act2 = nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 5), nn.ReLU()

    def forward(self, x):
        y = x.reshape(-1, 16, 8) if self.flatten else x
        y = self.act2(self.fc2(self.act1(self.fc1(y))))
        return (y.reshape(-1, 4, 4, 5) if self.flatten else y).mean(dim=(1, 2))


class TokenMlp(nn.Module):
    """A rank-2 per-sample input (5, 6): the case the FC mapping lowered on the base."""

    def __init__(self):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(6, 6), nn.ReLU(), nn.Linear(6, 4)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x))).mean(dim=1)


def _flow(model: nn.Module, shape, packaging):
    torch.manual_seed(0)
    return quietly(trace_and_convert, model.eval(), shape, packaging=packaging).flow


def _ir(flow):
    with contextlib.redirect_stdout(io.StringIO()):
        return IRMapping(q_max=1.0, firing_mode="Default", max_axons=None, max_neurons=None,
                         allow_coalescing=False, hardware_bias=False).map(flow.get_mapper_repr())


def _digest(graph) -> list:
    rows = []
    for node in graph.nodes:
        sources = [(s.node_id, s.index) for s in np.asarray(node.input_sources, dtype=object).flatten()]
        matrix = node.core_matrix if isinstance(node, NeuralCore) else None
        weights = None if matrix is None else matrix.tobytes()
        bank = getattr(node, "weight_bank_id", None)
        rows.append((type(node).__name__, node.id, sources, weights, bank))
    outputs = [(s.node_id, s.index) for s in np.asarray(graph.output_sources, dtype=object).flatten()]
    banks = sorted((k, np.asarray(v.core_matrix).tobytes()) for k, v in graph.weight_banks.items())
    return [rows, outputs, banks]


def _analytic_macs(model: nn.Module, shape) -> int:
    total = []

    def hook(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            k = module.kernel_size[0] * module.kernel_size[1]
            total.append(output[0].numel() * (module.in_channels // module.groups) * k)
        elif isinstance(module, nn.Linear):
            total.append(output[0].numel() * module.in_features)
    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model(torch.zeros(1, *shape))
    for handle in handles:
        handle.remove()
    return sum(total)


# ----------------------------------------------------------------------------- exactness
def test_the_value_program_of_a_channels_last_linear_is_exact():
    assert_value_program_exact(native(TinyConvNeXt), (1, 8, 8))


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_rank_3_encoding_linear_converts_exactly_and_maps(contract):
    flow = assert_converted_exact(native(ChannelsLastMlp), (4, 4, 8), contract)
    graph = _ir(flow)
    assert graph.nodes


def test_the_value_program_of_a_rank_3_encoding_linear_is_exact():
    assert_value_program_exact(native(ChannelsLastMlp), (4, 4, 8))


# ----------------------------------------------------------------------------- the rank-2 twin
@pytest.mark.parametrize("factory,shape,contract", [
    (TinyConvNeXt, (1, 8, 8), "mvm"), (ReluMlp, (4, 4, 8), "mvm"), (ReluMlp, (4, 4, 8), "spiking"),
], ids=["convnext_block-mvm", "relu_mlp-mvm", "relu_mlp-spiking"])
def test_a_rank_3_linear_maps_as_its_flattened_rank_2_twin(factory, shape, contract):
    """The same weights with the leading axes flattened by hand map to the same IR: node for node, every source,
    every core matrix and weight bank. (Each Linear is a perceptron here; under the spiking contract the ConvNeXt
    block's Linears before GELU and the layer scale are host ops, which the FC mapping does not lower.)"""
    torch.manual_seed(0)
    rank3 = factory().eval()
    twin = factory(flatten=True).eval()
    twin.load_state_dict(rank3.state_dict())
    a = _ir(_flow(rank3, shape, PACKAGINGS[contract]))
    b = _ir(_flow(twin, shape, PACKAGINGS[contract]))
    assert _digest(a) == _digest(b)


def test_the_layout_mapping_counts_the_cores_of_the_ir():
    flow = _flow(native(TinyConvNeXt).float(), (1, 8, 8), MVM_PACKAGING)
    layout = LayoutIRMapping(max_axons=None, max_neurons=None, allow_coalescing=False, hardware_bias=False)
    with contextlib.redirect_stdout(io.StringIO()):
        layout.map(flow.get_mapper_repr())
    cores = [node for node in _ir(flow).nodes if isinstance(node, NeuralCore)]
    assert len(layout.layout_softcores) == len(cores) > 2 * 16  # pw1 and pw2 at each of the 16 positions


# ----------------------------------------------------------------------------- MAC counters
@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("factory,shape", [(TinyConvNeXt, (1, 8, 8)), (ChannelsLastMlp, (4, 4, 8))],
                         ids=["convnext_block", "channels_last_mlp"])
def test_both_mac_counters_count_every_position_of_a_rank_3_linear(contract, factory, shape):
    model = native(factory).float()
    flow = _flow(model, shape, PACKAGINGS[contract])
    est = estimate_onchip_fraction(flow, shape, 10, metric="macs")
    assert count_host_ops(_ir(flow)) == est.host
    assert est.total == _analytic_macs(model, shape)


def test_a_host_rank_3_encoding_linear_counts_every_position():
    """Under the spiking contract the first Linear is the subsumed encoding layer, a host op per position (the base
    lowered it as one op of fan-in shape over all 128 inputs and counted 78 MACs), and the bare last Linear is a host
    op over the whole rank-3 input."""
    flow = _flow(native(ChannelsLastMlp).float(), (4, 4, 8), SPIKING_PACKAGING)
    graph = _ir(flow)
    encoding = [op for op in graph.get_compute_ops() if str(op.name).startswith("fc1")]
    assert len(encoding) == 16 and all(tuple(op.input_shape) == (8,) for op in encoding), encoding
    assert count_host_ops(graph) == 16 * 8 * 6 + 16 * 6 * 5


# ----------------------------------------------------------------------------- flowchart estimate
@pytest.mark.parametrize("out_shape,instances", [((6,), 1), ((5, 6), 5), ((4, 4, 6), 16), (None, 1)])
def test_the_flowchart_estimate_counts_a_linears_positions(out_shape, instances):
    mapper = PerceptronMapper(None, Perceptron(6, 8, normalization=nn.Identity()))
    assert mapper.flowchart_node_estimate(out_shape).fc_spec.instances == instances


# ----------------------------------------------------------------------------- rank 2 unchanged
_TOKEN_MLP_IR_SHA256 = "d7e505c6f8d659aedbff59ffa2b8329909970bb2248bf8b575b40462d49254fc"
"""sha256 of ``_digest`` of the token MLP's IR (MVM, unbounded cores), measured on the base tree (before rank-k)."""


def test_the_ir_of_a_rank_2_linear_is_the_bases():
    flow = _flow(native(TokenMlp).float(), (5, 6), MVM_PACKAGING)
    digest = hashlib.sha256(repr(_digest(_ir(flow))).encode()).hexdigest()
    assert digest == _TOKEN_MLP_IR_SHA256
