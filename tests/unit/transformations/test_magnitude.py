"""D4: structured magnitude pruning shrinks cores -> fewer reprogram phases.

The contract verified here (tests-first):

1. CORE REDUCTION (the deployment payoff): at sparsity ``s > 0`` the structurally
   pruned perceptron chain maps to STRICTLY FEWER hard cores than the dense chain,
   measured through the consumed ``estimate_cores_needed`` lower bound. Because the
   pruning is STRUCTURED (whole output neurons removed + the matching downstream
   axons), the per-segment diagonal bound ``max(ceil(Σ axons / max_axons),
   ceil(Σ neurons / max_neurons), ...)`` drops — unstructured zero-masking would
   leave the crossbar shapes (and thus the core count) unchanged.

2. DENSE DEFAULT (s == 0) is BYTE-IDENTICAL: every ``nn.Linear`` object and its
   parameter tensors are the SAME objects, untouched.

3. The pruned model FORWARD still runs and is shape-correct (logits dim preserved).
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.verification.capacity import estimate_cores_needed
from mimarsinan.transformations.pruning.magnitude import (
    ChannelPruningResult,
    kept_output_channels,
    prune_perceptron_chain,
)


class _Perceptron(nn.Module):
    """Minimal perceptron stand-in: the transform only touches ``.layer``."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.layer = nn.Linear(in_features, out_features, bias=bias)
        self.input_features = in_features
        self.output_channels = out_features

    def forward(self, x):
        return torch.relu(self.layer(x))


class _Chain(nn.Module):
    def __init__(self, widths, bias=True):
        super().__init__()
        self.perceptrons = nn.ModuleList(
            _Perceptron(widths[i], widths[i + 1], bias=bias)
            for i in range(len(widths) - 1)
        )

    def get_perceptrons(self):
        return list(self.perceptrons)

    def forward(self, x):
        for p in self.perceptrons:
            x = p(x)
        return x


def _seeded_chain(widths, seed=0, bias=True):
    torch.manual_seed(seed)
    return _Chain(widths, bias=bias)


def _src(specs):
    return np.array(
        [IRSource(node_id=nid, index=idx) for nid, idx in specs], dtype=object
    )


def _ir_from_chain(chain):
    """One NeuralCore per perceptron, wired sequentially -> a single neural segment.

    ``get_input_count`` = #input axons = ``len(input_sources)`` = the layer's
    ``in_features``; ``get_output_count`` = ``core_matrix.shape[1]`` = ``out_features``.
    So the IR core shapes track the (possibly pruned) ``nn.Linear`` shapes exactly,
    which is the quantity ``estimate_cores_needed`` sums.
    """
    nodes = []
    prev_id = -2  # network input
    for i, p in enumerate(chain.get_perceptrons()):
        in_f = p.layer.in_features
        out_f = p.layer.out_features
        sources = _src([(prev_id, j) for j in range(in_f)])
        nodes.append(
            NeuralCore(
                id=i,
                name=f"L{i}",
                input_sources=sources,
                core_matrix=np.ones((in_f, out_f), dtype=np.float64),
                threshold=1.0,
                latency=0,
            )
        )
        prev_id = i
    last = len(nodes) - 1
    return IRGraph(nodes=nodes, output_sources=_src([(last, 0)]))


def _cores(max_axons, max_neurons, count):
    return [
        {
            "max_axons": max_axons,
            "max_neurons": max_neurons,
            "count": count,
            "has_bias": True,
        }
    ]


# Wide intermediate widths so the diagonal (axon/neuron-sum) bound, not a fixed
# floor, governs the core count -> structured pruning visibly drops it.
_WIDTHS = [64, 256, 256, 256, 10]
_PLATFORM = {"cores": _cores(64, 64, 4096)}


class TestCoreReduction:
    def test_pruned_chain_maps_to_fewer_cores(self):
        dense = _seeded_chain(_WIDTHS, seed=1)
        pruned = _seeded_chain(_WIDTHS, seed=1)

        dense_cores = estimate_cores_needed(_ir_from_chain(dense), _PLATFORM).cores_needed

        result = prune_perceptron_chain(pruned.get_perceptrons(), sparsity=0.5)
        assert result.pruned is True
        pruned_cores = estimate_cores_needed(_ir_from_chain(pruned), _PLATFORM).cores_needed

        assert pruned_cores < dense_cores, (
            f"structured pruning must drop cores: dense={dense_cores} "
            f"pruned={pruned_cores}"
        )

    def test_pruning_drops_reprogram_phase_count_under_scheduling(self):
        """The D4 payoff end to end: fewer cores -> fewer reprogram phases.

        On a SCHEDULED chip with a tight budget each oversized segment needs
        ``ceil(segment_bound / budget)`` reprogram passes; structured pruning
        lowers ``segment_bound`` and so lowers the total ``phase_count``.
        """
        tight = {
            "cores": _cores(64, 64, 4),
            "allow_scheduling": True,
        }
        dense = _seeded_chain(_WIDTHS, seed=8)
        pruned = _seeded_chain(_WIDTHS, seed=8)
        dense_phases = estimate_cores_needed(_ir_from_chain(dense), tight).phase_count

        prune_perceptron_chain(pruned.get_perceptrons(), sparsity=0.5)
        pruned_phases = estimate_cores_needed(_ir_from_chain(pruned), tight).phase_count

        assert pruned_phases < dense_phases, (
            f"pruning must cut reprogram phases: dense={dense_phases} "
            f"pruned={pruned_phases}"
        )

    def test_intermediate_widths_actually_shrink(self):
        """The structural shrink is real: out_features and downstream in_features drop."""
        chain = _seeded_chain(_WIDTHS, seed=2)
        prune_perceptron_chain(chain.get_perceptrons(), sparsity=0.5)
        ps = chain.get_perceptrons()
        # intermediate layers lose ~half their output neurons
        assert ps[0].layer.out_features == 256 - math.floor(256 * 0.5)
        assert ps[1].layer.out_features == 256 - math.floor(256 * 0.5)
        assert ps[2].layer.out_features == 256 - math.floor(256 * 0.5)
        # downstream in_features track the upstream pruned outputs
        assert ps[1].layer.in_features == ps[0].layer.out_features
        assert ps[2].layer.in_features == ps[1].layer.out_features
        assert ps[3].layer.in_features == ps[2].layer.out_features
        # logits (last output) and network input (first input) are exempt
        assert ps[-1].layer.out_features == 10
        assert ps[0].layer.in_features == 64


class TestDenseDefaultByteIdentical:
    def test_sparsity_zero_is_byte_identical(self):
        chain = _seeded_chain(_WIDTHS, seed=3)
        layers_before = [p.layer for p in chain.get_perceptrons()]
        weights_before = [l.weight.clone() for l in layers_before]
        biases_before = [l.bias.clone() for l in layers_before]

        result = prune_perceptron_chain(chain.get_perceptrons(), sparsity=0.0)

        assert result.pruned is False
        assert result.sparsity == 0.0
        layers_after = [p.layer for p in chain.get_perceptrons()]
        for before_obj, after_obj, w0, b0 in zip(
            layers_before, layers_after, weights_before, biases_before
        ):
            # SAME object identity (no replacement) and bit-exact tensors
            assert before_obj is after_obj
            assert torch.equal(after_obj.weight, w0)
            assert torch.equal(after_obj.bias, b0)

    def test_sparsity_zero_estimate_unchanged(self):
        a = _seeded_chain(_WIDTHS, seed=4)
        b = _seeded_chain(_WIDTHS, seed=4)
        prune_perceptron_chain(b.get_perceptrons(), sparsity=0.0)
        assert (
            estimate_cores_needed(_ir_from_chain(a), _PLATFORM).cores_needed
            == estimate_cores_needed(_ir_from_chain(b), _PLATFORM).cores_needed
        )


class TestPrunedForwardRuns:
    def test_pruned_forward_shape_correct(self):
        chain = _seeded_chain(_WIDTHS, seed=5)
        x = torch.randn(8, _WIDTHS[0])
        out_before = chain(x)
        assert out_before.shape == (8, _WIDTHS[-1])

        prune_perceptron_chain(chain.get_perceptrons(), sparsity=0.5)
        out_after = chain(x)
        assert out_after.shape == (8, _WIDTHS[-1])
        assert torch.isfinite(out_after).all()

    def test_pruned_forward_runs_biasless(self):
        chain = _seeded_chain(_WIDTHS, seed=6, bias=False)
        x = torch.randn(4, _WIDTHS[0])
        prune_perceptron_chain(chain.get_perceptrons(), sparsity=0.25)
        out = chain(x)
        assert out.shape == (4, _WIDTHS[-1])
        assert torch.isfinite(out).all()


class TestKeptOutputChannels:
    def test_keeps_high_magnitude_drops_low(self):
        # row 0 = tiny, row 2 = large -> at s=0.5 (drop 1 of 3... floor=1) drop row 0
        w = torch.tensor([[0.01, 0.01], [1.0, 1.0], [5.0, 5.0]])
        mask = kept_output_channels(w, sparsity=0.5)
        assert mask.tolist() == [False, True, True]

    def test_never_drops_all(self):
        w = torch.randn(8, 4)
        mask = kept_output_channels(w, sparsity=1.0)
        assert int(mask.sum().item()) == 1  # keeps at least one

    def test_zero_sparsity_keeps_all(self):
        w = torch.randn(8, 4)
        mask = kept_output_channels(w, sparsity=0.0)
        assert bool(mask.all())


class TestResultDataclass:
    def test_kept_counts_reported(self):
        chain = _seeded_chain([32, 100, 10], seed=7)
        result = prune_perceptron_chain(chain.get_perceptrons(), sparsity=0.3)
        assert isinstance(result, ChannelPruningResult)
        # intermediate layer kept 100 - floor(30) = 70; last layer all 10
        assert result.kept_output_counts == [70, 10]
