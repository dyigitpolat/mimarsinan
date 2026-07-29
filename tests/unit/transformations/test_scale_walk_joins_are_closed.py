"""W0.8b findings 4 and 6 -- the scale-walk device invariant at EVERY join.

W0.8 claimed the device invariant was "made real instead of claimed" and closed it
at one join: ``normalize_fan_in_scales`` (the ComputeOp fan-in). The walk has three
multi-source joins, not one, and it enumerates them by construction --
``propagate_source_scale`` with more than one recorded source:

  * ``ComputeOpMapper``          -> ``normalize_fan_in_scales``   (closed by W0.8)
  * ``ConcatMapper``             -> ``torch.cat(parts)``          (LEFT OPEN)
  * ``_ResidualConcatMapper``    -> ``torch.cat(parts)``          (LEFT OPEN)

Both open ones are reachable from the exact precondition W0.8's own commit message
identified as the real hole: a parameterless ``InputMapper`` root emits a CPU unit
scale, and ``torch.cat`` of a CPU tensor with a device one raises. So the claim was
false at two of three joins.

Finding 6 is in the same file: ``assign_per_input_scales`` -- the ONE writer of
``per_input_scales`` and the seam where the root scale is anchored -- pinned the
DEVICE and not the DTYPE, one function after ``perceptron_source_out_scale``
deliberately stopped inheriting the ambient default dtype for exactly that reason.
A root scale is born in whatever ``torch.get_default_dtype()`` happens to be, so
the stamp could disagree with the float32 currency every derived scale carries.
"""

import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.scale_propagation import assign_per_input_scales
from mimarsinan.mapping.mappers.structural import ConcatMapper
from mimarsinan.mapping.support.residual_merge import _ResidualConcatMapper
from mimarsinan.mapping.support.scale_broadcast import concat_source_scales
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron


class _Node:
    """A dep key; the walk keys ``out_scales`` by node identity."""


def _deps_and_scales(*scales):
    nodes = [_Node() for _ in scales]
    return nodes, {n: s for n, s in zip(nodes, scales)}


class TestConcatJoinsSurviveAMixedDeviceGraph:
    """A CPU root scale meeting a device theta is the documented precondition."""

    def test_structural_concat_lifts_the_root_scale(self):
        deps, out_scales = _deps_and_scales(
            torch.ones(4), torch.full((4,), 2.0, device="meta")
        )
        joined = ConcatMapper([]).propagate_source_scale(deps, out_scales)
        assert joined is not None
        assert joined.device.type == "meta"
        assert joined.shape == (8,)

    def test_residual_concat_lifts_the_root_scale(self):
        deps, out_scales = _deps_and_scales(
            torch.ones(4), torch.full((4,), 2.0, device="meta")
        )
        joined = _ResidualConcatMapper([]).propagate_source_scale(deps, out_scales)
        assert joined is not None
        assert joined.device.type == "meta"
        assert joined.shape == (8,)

    def test_the_anchored_device_wins_regardless_of_source_order(self):
        deps, out_scales = _deps_and_scales(
            torch.ones(2, device="meta"), torch.ones(3)
        )
        joined = ConcatMapper([]).propagate_source_scale(deps, out_scales)
        assert joined is not None and joined.device.type == "meta"

    def test_a_sourceless_concat_still_records_nothing(self):
        assert ConcatMapper([]).propagate_source_scale([], {}) is None
        assert _ResidualConcatMapper([]).propagate_source_scale([], {}) is None


class TestConcatJoinsAreValueIdenticalOnCpu:
    """Byte-identical default: an all-CPU walk is the untouched ``torch.cat``."""

    def test_structural_concat_values_are_unchanged(self):
        deps, out_scales = _deps_and_scales(
            torch.tensor([1.0, 2.0]), torch.tensor([3.0])
        )
        joined = ConcatMapper([]).propagate_source_scale(deps, out_scales)
        torch.testing.assert_close(joined, torch.tensor([1.0, 2.0, 3.0]))

    def test_residual_concat_values_are_unchanged(self):
        deps, out_scales = _deps_and_scales(
            torch.tensor([1.0, 2.0]), torch.tensor([3.0])
        )
        joined = _ResidualConcatMapper([]).propagate_source_scale(deps, out_scales)
        torch.testing.assert_close(joined, torch.tensor([1.0, 2.0, 3.0]))

    def test_the_shared_join_returns_the_inputs_untouched_when_all_cpu(self):
        parts = [torch.ones(2), torch.ones(3)]
        torch.testing.assert_close(concat_source_scales(parts), torch.ones(5))


class TestTheStampAnchorsDtypeAsWellAsDevice:
    """``per_input_scales`` is the weight-fold currency; float32 is that currency."""

    def _root_scale(self, n):
        """A ROOT scale, born the only way a parameterless node can make one."""
        return torch.ones(n)

    def test_a_float64_root_scale_is_stamped_in_the_walk_currency(self):
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            p = Perceptron(4, 4, normalization=nn.Identity())
            assign_per_input_scales(p, self._root_scale(4))
        finally:
            torch.set_default_dtype(previous)
        assert p.per_input_scales is not None
        assert p.per_input_scales.dtype is torch.float32

    def test_the_repeat_interleave_branch_is_anchored_too(self):
        p = Perceptron(4, 8, normalization=nn.Identity())
        assign_per_input_scales(p, torch.ones(4, dtype=torch.float64))
        assert p.per_input_scales is not None
        assert p.per_input_scales.dtype is torch.float32
        assert p.per_input_scales.shape == (8,)

    def test_the_mean_fold_branch_is_anchored_too(self):
        p = Perceptron(4, 6, normalization=nn.Identity())
        assign_per_input_scales(p, torch.full((4,), 0.5, dtype=torch.float64))
        assert p.per_input_scales is not None
        assert p.per_input_scales.dtype is torch.float32

    def test_device_anchoring_is_unaffected(self):
        p = Perceptron(4, 4, normalization=nn.Identity()).to("meta")
        assign_per_input_scales(p, torch.ones(4, dtype=torch.float64))
        assert p.per_input_scales is not None
        assert p.per_input_scales.device == p.layer.weight.device

    def test_float32_stamps_are_value_identical(self):
        p = Perceptron(4, 8, normalization=nn.Identity())
        assign_per_input_scales(p, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        torch.testing.assert_close(
            p.per_input_scales,
            torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]),
        )
