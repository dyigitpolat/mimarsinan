"""W0.8 finding 2 — the scale-propagation device invariant, made real.

W0.7 claimed "all effective-parameter inputs are materialized from the perceptron's
own parameters, so the device/dtype invariant holds by construction". Two bare-CPU
creates of exactly the class W0.7 fixed survived that claim:

  * ``scale_propagation.apply_compute_op_scale_policy`` mean-filled a mismatched
    fan-in with ``torch.full(..., t.mean().item(), dtype=t.dtype)`` -- it carried
    the dtype and dropped the DEVICE;
  * its direct callee ``scale_broadcast.broadcast_scale_pair`` mean-filled with
    ``torch.full((n,), short.mean().item())`` -- neither device nor dtype.

The invariant is now enforced, not asserted, at three enumerated points, one per
kind of tensor the walk can produce; each has a test below.

  (1) DERIVED scales follow their input: every fold goes through
      ``spread_scalar``, which broadcasts the reduction tensor instead of
      round-tripping it through a host float.
  (2) SEEDED scales state their device: the only ``torch.full`` left in the walk
      takes ``device=`` from the perceptron's weight and pins ``dtype`` explicitly.
  (3) ROOT scales are anchored by their consumer: ``InputMapper`` is parameterless
      and can only emit CPU, so ``assign_per_input_scales`` lands the stamp on the
      perceptron's device and ``align_scale_devices`` lifts it before any join.

The CPU suite reproduces the mixed-device precondition with the ``meta`` device,
exactly as the W0.7 test does.
"""

import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.scale_propagation import (
    apply_compute_op_scale_policy,
    assign_per_input_scales,
    normalize_fan_in_scales,
    perceptron_source_out_scale,
)
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.support.scale_broadcast import (
    align_scale_devices,
    broadcast_scale_pair,
    broadcast_scale_to_dim,
    spread_scalar,
)
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron


class _Op:
    """The ComputeOp slots ``apply_compute_op_scale_policy`` writes."""

    is_wire_value_op = False

    def __init__(self):
        self.per_source_scales = None
        self.output_scale = None

    def combine_source_scales(self, scales):
        return torch.stack([s.to(dtype=torch.float32) for s in scales]).mean(dim=0)


class TestDerivedScalesFollowTheirInput:
    def test_spread_scalar_keeps_device(self):
        assert spread_scalar(torch.tensor(0.5, device="meta"), 3).device.type == "meta"

    def test_spread_scalar_keeps_dtype(self):
        assert spread_scalar(torch.tensor(0.5, dtype=torch.float64), 3).dtype is torch.float64

    def test_broadcast_scale_pair_mean_fill_stays_on_device(self):
        """Was a bare ``torch.full``: a meta/CUDA fan-in came back on CPU."""
        short = torch.ones(2, device="meta")
        long_ = torch.ones(5, device="meta")
        a, b = broadcast_scale_pair(short, long_)
        assert a.device == short.device and b.device == long_.device

    def test_broadcast_scale_pair_mean_fill_keeps_dtype(self):
        short = torch.ones(2, dtype=torch.float64)
        long_ = torch.ones(5, dtype=torch.float64)
        a, _ = broadcast_scale_pair(short, long_)
        assert a.dtype is torch.float64

    def test_broadcast_scale_to_dim_mean_fill_stays_on_device(self):
        scale = torch.ones(2, device="meta")
        assert broadcast_scale_to_dim(scale, 5).device == scale.device

    def test_compute_op_mismatched_fan_in_stays_on_device(self):
        """The second surviving site: the mean-fold inside the op's fan-in policy.

        Pre-fix this raised ``Tensor.item() cannot be called on meta tensors`` --
        the host round-trip that relocated the result to CPU is exactly what the
        meta device refuses to perform."""
        normalized = normalize_fan_in_scales(
            [torch.ones(4, device="meta"), torch.full((3,), 2.0, device="meta")]
        )
        assert all(s.shape == (4,) for s in normalized)
        assert all(s.device.type == "meta" for s in normalized)

    def test_mean_fill_values_are_unchanged_on_cpu(self):
        """Byte-identical default: on CPU float32 the rewrite is a no-op."""
        short = torch.tensor([1.0, 3.0])
        _, expanded = broadcast_scale_pair(torch.ones(5), short)
        torch.testing.assert_close(expanded, torch.full((5,), 2.0))
        torch.testing.assert_close(
            broadcast_scale_to_dim(torch.tensor([1.0, 3.0]), 5), torch.full((5,), 2.0)
        )


class TestSeededScalesStateTheirDevice:
    def test_scalar_theta_seed_follows_the_perceptron_device(self):
        p = Perceptron(4, 6, normalization=nn.Identity()).to("meta")
        assert perceptron_source_out_scale(p).device == p.layer.weight.device

    def test_scalar_theta_seed_pins_float32_like_the_tensor_branch(self):
        """``torch.full`` with a python float takes the AMBIENT default dtype, so
        the scalar branch could disagree with the float32 the tensor branch pins."""
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            p = Perceptron(4, 6, normalization=nn.Identity())
            p.activation_scale = nn.Parameter(torch.tensor(2.0), requires_grad=False)
            scalar_branch = perceptron_source_out_scale(
                _ScalarThetaPerceptron(p)
            )
            assert scalar_branch.dtype is torch.float32
        finally:
            torch.set_default_dtype(previous)


class _ScalarThetaPerceptron:
    """A perceptron view whose ``activation_scale`` is a plain python float,
    which is the branch that seeds rather than derives."""

    def __init__(self, perceptron):
        self.layer = perceptron.layer
        self.output_channels = perceptron.output_channels
        self.activation_scale = 2.0


class TestRootScalesAreAnchoredByTheirConsumer:
    def test_input_mapper_emits_a_parameterless_cpu_unit_scale(self):
        """Stated, not hidden: the root has nothing to anchor on."""
        scale = InputMapper((3, 8)).propagate_source_scale([], {})
        assert scale.device.type == "cpu"
        torch.testing.assert_close(scale, torch.ones(3))

    def test_the_stamp_lands_a_cpu_root_scale_on_the_perceptron_device(self):
        """Without this, a graph rooted in InputMapper seeds a mixed-device
        perceptron on every CUDA model -- the W0.7 defect, one seam over."""
        p = Perceptron(4, 4, normalization=nn.Identity()).to("meta")
        assign_per_input_scales(p, torch.ones(4))  # bare CPU root scale
        assert p.per_input_scales is not None
        assert p.per_input_scales.device == p.layer.weight.device

    def test_the_stamp_anchors_the_repeat_interleave_branch_too(self):
        p = Perceptron(4, 8, normalization=nn.Identity()).to("meta")
        assign_per_input_scales(p, torch.ones(4))
        assert p.per_input_scales is not None
        assert p.per_input_scales.device == p.layer.weight.device
        assert p.per_input_scales.shape == (8,)

    def test_the_stamp_anchors_the_mean_fold_branch_too(self):
        p = Perceptron(4, 6, normalization=nn.Identity()).to("meta")
        assign_per_input_scales(p, torch.full((4,), 0.5))
        assert p.per_input_scales is not None
        assert p.per_input_scales.device == p.layer.weight.device

    def test_a_join_lifts_the_root_scale_before_anything_mixes_it(self):
        """A ComputeOp joining the input wire to a perceptron output used to
        ``torch.stack`` a CPU tensor with a device one."""
        aligned = align_scale_devices([torch.ones(4), torch.ones(4, device="meta")])
        assert all(s.device.type == "meta" for s in aligned)

    def test_an_all_cpu_walk_is_returned_unchanged(self):
        scales = [torch.ones(4), torch.ones(4)]
        aligned = align_scale_devices(scales)
        assert [s.device.type for s in aligned] == ["cpu", "cpu"]
        assert all(a is b for a, b in zip(aligned, scales))

    def test_compute_op_fan_in_survives_a_mixed_device_join(self):
        """A CPU root scale meeting a device theta used to reach
        ``combine_source_scales``' ``torch.stack`` as-is."""
        normalized = normalize_fan_in_scales(
            [torch.ones(4), torch.full((4,), 2.0, device="meta")]
        )
        assert all(s.device.type == "meta" for s in normalized)

    def test_compute_op_policy_is_value_identical_on_cpu(self):
        """Byte-identical default: the CPU float32 walk is untouched."""
        op = _Op()
        out = apply_compute_op_scale_policy(
            op, [torch.tensor([1.0, 1.0]), torch.tensor([3.0, 3.0])]
        )
        assert out is not None
        torch.testing.assert_close(out, torch.full((2,), 2.0))
        assert op.per_source_scales is not None

    def test_stamped_values_are_unchanged_on_the_cpu_default_path(self):
        p = Perceptron(4, 8, normalization=nn.Identity())
        assign_per_input_scales(p, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        torch.testing.assert_close(
            p.per_input_scales,
            torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]),
        )
