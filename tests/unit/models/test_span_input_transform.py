"""[D6a] The span plan applies boundary transforms; callers never index columns.

The value executor used to rebuild the input-sourced column span from a
tagged tuple (``("slice", (d0, d1)) | ("index", tensor)``) and slice around
the plan that already owned those columns. The plan now takes the transform.
"""

import numpy as np
import torch

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources
from mimarsinan.models.spiking.signal_spans import SpanFillPlan


def _plan(sources):
    return SpanFillPlan(compress_spike_sources(sources), torch.device("cpu"))


def _entry_sources(n):
    """All columns read the segment input (an entry core)."""
    return [SpikeSource(-2, i, is_input=True, is_off=False) for i in range(n)]


class TestInputTransformHook:
    def test_transform_applies_to_input_columns(self):
        plan = _plan(_entry_sources(4))
        out = torch.empty(2, 4)
        x = torch.arange(8.0).reshape(2, 4)
        plan.apply(out, input_spikes=x, buffers={}, on_value=1.0,
                   input_transform=lambda t: t * 10.0)
        torch.testing.assert_close(out, x * 10.0)

    def test_absent_transform_is_the_identity(self):
        plan = _plan(_entry_sources(4))
        a, b = torch.empty(2, 4), torch.empty(2, 4)
        x = torch.arange(8.0).reshape(2, 4)
        plan.apply(a, input_spikes=x, buffers={}, on_value=1.0)
        plan.apply(b, input_spikes=x, buffers={}, on_value=1.0,
                   input_transform=None)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(a, x)

    def test_transform_leaves_non_input_columns_untouched(self):
        # An always-on column must keep its on_value, not be transformed.
        sources = [SpikeSource(-2, 0, is_input=True, is_off=False),
                   SpikeSource(-3, 0, is_input=False, is_off=False,
                               is_always_on=True)]
        plan = _plan(sources)
        out = torch.empty(1, 2)
        plan.apply(out, input_spikes=torch.tensor([[5.0]]), buffers={},
                   on_value=1.0, input_transform=lambda t: t * 0.0)
        assert float(out[0, 0]) == 0.0   # transformed input column
        assert float(out[0, 1]) == 1.0   # always-on column preserved

    def test_no_input_group_means_the_transform_never_fires(self):
        plan = _plan([SpikeSource(0, 0, is_input=False, is_off=False)])
        called = []
        out = torch.empty(1, 1)
        plan.apply(out, input_spikes=torch.zeros(1, 1),
                   buffers={0: torch.tensor([[3.0]])}, on_value=1.0,
                   input_transform=lambda t: called.append(1) or t)
        assert not called
        assert float(out[0, 0]) == 3.0
