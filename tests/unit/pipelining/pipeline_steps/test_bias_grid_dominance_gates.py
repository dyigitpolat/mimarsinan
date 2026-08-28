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
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.pipelining.pipeline_steps.quantization.weight_quantization_step import (
    BiasGridDominanceError,
    BiasRowSplitEventSerialError,
    refuse_bias_dominated_grid,
    refuse_split_under_event_serial_soma,
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


def _soma(granularity, membrane_bits):
    return SomaLaw.resolve({
        "firing_granularity": granularity,
        "membrane_bits": membrane_bits,
        "membrane_signed": False,
        "firing_mode": "Novena",
        "thresholding_mode": "<=",
    })


class TestEventSerialSplitRefusal:
    """A k>1 split is refused BEFORE the training budget on the one soma point
    where it is not a value-preserving re-encoding (measured; the evidence lives
    in tests/unit/pipelining/test_streamed_bias_row_exactness.py)."""

    def _needs_many_rows(self):
        # max|b|/max|w| = 7.04 -> s_w = floor(q_max/max|w|) = 69 -> k = 7 rows.
        return _Repr([_perceptron(0.1, 0.704)])

    def test_the_event_serial_soma_with_a_register_refuses(self):
        with pytest.raises(BiasRowSplitEventSerialError) as excinfo:
            refuse_split_under_event_serial_soma(
                self._needs_many_rows(), BITS, _soma("per_event", 8)
            )
        message = str(excinfo.value)
        assert "fc0" in message and "7 always-on rows" in message
        assert "per_cycle" in message

    def test_an_unbounded_membrane_is_allowed(self):
        refuse_split_under_event_serial_soma(
            self._needs_many_rows(), BITS, _soma("per_event", 0)
        )

    def test_per_cycle_firing_is_allowed(self):
        refuse_split_under_event_serial_soma(
            self._needs_many_rows(), BITS, _soma("per_cycle", 8)
        )

    def test_a_vehicle_whose_bound_is_one_row_passes_everywhere(self):
        # max|b| * s_w <= q_max (0.05 * 69 = 3.45 <= 7): one row carries it.
        refuse_split_under_event_serial_soma(
            _Repr([_perceptron(0.1, 0.05)]), BITS, _soma("per_event", 8)
        )
