"""The bias-row split's event-serial refusal (ODIN, bias_row_splitting).

[U-H port] The lineage file also tested two default-path grid refusals (the WQ-entry bias-dominance refusal and the
artifact-side retained-level refusal, commit 8f614db2). They are not ported: they add refusals to every release-head
run, which the release head governs (final_units/uh/report.md, the dropped-commit table). What stays is the refusal
that arms only on a per-event soma with a fixed-width membrane.
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.pipelining.pipeline_steps.quantization.weight_quantization_step import (
    BiasRowSplitEventSerialError,
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
