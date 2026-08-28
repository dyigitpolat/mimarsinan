"""The two fail-loud grid gates, which hold whether or not splitting is on.

The advisory `rule_bias_grid_dominance` PREDICTS the collapse correctly but
evaluates at Torch Mapping, before conversion grows the ratio — a clean read
there does not certify the WQ entry (the rule's own detail text says so). So
the same predicate is re-read where the tensors are final, and the artifact-side
twin is read on what QuantizationVerificationStep already computes.
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.advisories.rules_graph_scale import (
    BIAS_DOMINANCE_LEVEL_FLOOR,
    worst_bias_grid_dominance,
)
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.pipelining.pipeline_steps.quantization.quantization_verification_step import (
    RetainedLevelCollapseError,
    assert_grid_retains_levels,
    integer_grid_stats,
)
from mimarsinan.pipelining.pipeline_steps.quantization.weight_quantization_step import (
    BiasGridDominanceError,
    refuse_bias_dominated_grid,
)

BITS = 4
Q_MAX = (2 ** (BITS - 1)) - 1


def _perceptron(w_max, b_max, name="fc0"):
    p = Perceptron(4, 6, normalization=nn.Identity())
    p.set_activation_scale(1.0)
    p.name = name
    with torch.no_grad():
        p.layer.weight.data.zero_()
        p.layer.weight.data[0, 0] = w_max
        p.layer.bias.data.zero_()
        p.layer.bias.data[0] = b_max
    return p


class _Repr:
    def __init__(self, perceptrons):
        self._perceptrons = perceptrons

    def get_perceptrons(self):
        return self._perceptrons


class TestWorstDominance:
    def test_a_balanced_graph_reports_nothing(self):
        assert worst_bias_grid_dominance(_Repr([_perceptron(0.1, 0.2)]), BITS) is None

    def test_the_measured_ratio_is_reported_with_its_perceptron(self):
        worst = worst_bias_grid_dominance(_Repr([_perceptron(0.1, 0.704)]), BITS)
        assert worst is not None
        ratio, name = worst
        assert name == "fc0"
        assert ratio == pytest.approx(7.04, rel=1e-3)

    def test_the_limit_is_the_rules_own_q_max_over_the_level_floor(self):
        limit = Q_MAX / BIAS_DOMINANCE_LEVEL_FLOOR
        assert worst_bias_grid_dominance(
            _Repr([_perceptron(1.0, limit * 0.99)]), BITS
        ) is None
        assert worst_bias_grid_dominance(
            _Repr([_perceptron(1.0, limit * 1.01)]), BITS
        ) is not None

    def test_the_worst_hop_wins(self):
        worst = worst_bias_grid_dominance(
            _Repr([_perceptron(0.1, 0.5, "a"), _perceptron(0.1, 0.9, "b")]), BITS
        )
        assert worst is not None and worst[1] == "b"


class TestWqEntryRefusal:
    def test_a_bias_dominated_shared_grid_is_refused_by_name(self):
        with pytest.raises(BiasGridDominanceError) as excinfo:
            refuse_bias_dominated_grid(
                _Repr([_perceptron(0.1, 0.704)]), BITS, weight_only_grid=False
            )
        message = str(excinfo.value)
        assert "fc0" in message
        assert "7.0" in message
        assert "bias_row_splitting" in message

    def test_the_weight_only_grid_cures_it_and_the_gate_stands_down(self):
        refuse_bias_dominated_grid(
            _Repr([_perceptron(0.1, 0.704)]), BITS, weight_only_grid=True
        )

    def test_a_healthy_graph_passes_on_the_shared_grid(self):
        refuse_bias_dominated_grid(
            _Repr([_perceptron(0.1, 0.2)]), BITS, weight_only_grid=False
        )


class TestRetainedLevelRefusal:
    def _stats(self, ints):
        return integer_grid_stats(torch.tensor(ints, dtype=torch.float32), Q_MAX)

    def test_the_measured_collapse_is_refused(self):
        """Platform-J's artifact: {-1, 0, 1} and 99.6% zeros at wb=4."""
        stats = self._stats([0.0] * 249 + [1.0, -1.0, 0.0])
        assert stats["effective_levels"] == 3
        with pytest.raises(RetainedLevelCollapseError, match="fc0"):
            assert_grid_retains_levels("fc0", stats)

    def test_a_grid_whose_largest_weight_keeps_two_levels_passes(self):
        assert_grid_retains_levels("fc0", self._stats([0.0] * 200 + [2.0, -1.0]))

    def test_a_fully_pruned_layer_is_not_a_collapse(self):
        """All-zero integers are a pruning outcome, not a starved grid: the
        gate judges the LARGEST weight's retained levels, and there is none."""
        assert_grid_retains_levels("fc0", self._stats([0.0] * 64))

    def test_the_refusal_names_the_remedy_and_reports_the_zero_fraction(self):
        with pytest.raises(RetainedLevelCollapseError) as excinfo:
            assert_grid_retains_levels("fc0", self._stats([0.0] * 99 + [1.0]))
        message = str(excinfo.value)
        assert "bias_row_splitting" in message
        assert "0.99" in message
