"""Fan-in wire-scale contract (§6b contract-1): observed traffic, never mean-of-θ.

At a multi-source boundary join (a residual add is the canonical case) the
consumer's wire scale must cover the traffic that actually crosses — a sum's
range is up to the SUM of its source ranges, so the mean-of-producer-θ rule
systematically under-covers and saturates the re-encode clamp (0.5% → 25%
down the t0_19 residual chain, T4 §2a). The calibration lifts a durable
per-consumer floor from a calibration batch through the mode's forward;
equal-θ fan-ins whose traffic stays inside the mean remain bit-identical.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from cascade_fixtures import install_ttfs_nodes

from mimarsinan.mapping.support.residual_merge import _ResidualConcatMapper
from mimarsinan.spiking.distribution_matching import calibrate_fanin_boundary_scales
from mimarsinan.spiking.scale_aware_boundaries import propagate_boundary_input_scales
from mimarsinan.torch_mapping.converter import convert_torch_model

_S = 4
_IN = 8


class _ResidualJoinMLP(nn.Module):
    """Two perceptron branches summed at a host residual-add join (fan-in 2).

    The bounded ``z * 0.7 + 0.05`` host op after the encoder stops the subsume
    cascade, so f1/f2/tail stay perceptrons (the cascade_fixtures pattern).
    """

    def __init__(self, gain: float = 0.4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(_IN, _IN), nn.ReLU())
        self.f1 = nn.Sequential(nn.Linear(_IN, _IN), nn.ReLU())
        self.f2 = nn.Sequential(nn.Linear(_IN, _IN), nn.ReLU())
        self.tail = nn.Sequential(nn.Linear(_IN, 4), nn.ReLU())
        # Positive weights: at high gain BOTH branches saturate their θ, so the
        # sum crossing the join provably exceeds any single producer's range.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, 0.0, gain)
                nn.init.uniform_(m.bias, 0.0, 0.05)

    def forward(self, x):
        z = self.enc(x)
        z = z * 0.7 + 0.05
        y = self.f1(z) + self.f2(z)
        return self.tail(y)


class _ChainMLP(nn.Module):
    """No joins anywhere (single-source boundaries only)."""

    def __init__(self):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(_IN, _IN), nn.ReLU())
        self.b = nn.Sequential(nn.Linear(_IN, 4), nn.ReLU())

    def forward(self, x):
        z = self.a(x)
        z = z * 0.7 + 0.05
        return self.b(z)


def _build(module_cls, *, seed=0, **kwargs):
    torch.manual_seed(seed)
    base = module_cls(**kwargs)
    flow = convert_torch_model(base, (_IN,), 4, device="cpu")
    cal_x = torch.rand(32, _IN, dtype=torch.float64)
    return flow, cal_x


def _prepare_ttfs(flow, *, theta_by_name=None):
    for p in flow.get_perceptrons():
        theta = (theta_by_name or {}).get(p.name, 1.0)
        p.set_activation_scale(float(theta))
    propagate_boundary_input_scales(flow)
    install_ttfs_nodes(flow, _S)
    return flow


def _consumer(flow, name_part):
    for p in flow.get_perceptrons():
        if name_part in str(p.name):
            return p
    raise AssertionError(f"no perceptron matching {name_part!r}")


class TestEqualThetaBitIdentical:
    def test_non_saturating_equal_theta_join_is_bit_identical(self):
        flow, cal_x = _build(_ResidualJoinMLP, gain=0.05)
        _prepare_ttfs(flow, theta_by_name=None)  # all θ equal (1.0)
        reference = copy.deepcopy(flow)
        stats = calibrate_fanin_boundary_scales(flow, cal_x, _S)
        assert stats["n_joins"] == 1
        for p_new, p_ref in zip(flow.get_perceptrons(), reference.get_perceptrons()):
            assert float(p_new.input_activation_scale) == float(
                p_ref.input_activation_scale
            ), "equal-θ fan-in with in-range traffic must stay bit-identical"
            assert float(p_new.activation_scale) == float(p_ref.activation_scale)

    def test_no_join_graph_is_a_no_op(self):
        flow, cal_x = _build(_ChainMLP)
        _prepare_ttfs(flow)
        reference = copy.deepcopy(flow)
        stats = calibrate_fanin_boundary_scales(flow, cal_x, _S)
        assert stats["n_joins"] == 0 and stats["n_lifted"] == 0
        for p_new, p_ref in zip(flow.get_perceptrons(), reference.get_perceptrons()):
            assert float(p_new.input_activation_scale) == float(
                p_ref.input_activation_scale
            )


class TestObservedTrafficLift:
    def _saturating_flow(self):
        # Strong positive weights: z + branch(z) traffic exceeds every producer θ.
        flow, cal_x = _build(_ResidualJoinMLP, gain=0.9)
        _prepare_ttfs(flow)
        return flow, cal_x

    def test_saturating_join_lifts_consumer_scale(self):
        flow, cal_x = self._saturating_flow()
        tail = _consumer(flow, "tail")
        before = float(tail.input_activation_scale)
        stats = calibrate_fanin_boundary_scales(flow, cal_x, _S)
        after = float(tail.input_activation_scale)
        assert stats["n_lifted"] == 1
        assert after > before, "under-covered join traffic must lift the wire scale"
        assert float(tail.boundary_scale_floor) == pytest.approx(after)

    def test_floor_survives_repropagation(self):
        flow, cal_x = self._saturating_flow()
        tail = _consumer(flow, "tail")
        calibrate_fanin_boundary_scales(flow, cal_x, _S)
        lifted = float(tail.input_activation_scale)
        propagate_boundary_input_scales(flow)  # the SCM-step re-propagation
        assert float(tail.input_activation_scale) == pytest.approx(lifted), (
            "the observed-traffic floor must survive downstream re-propagations"
        )

    def test_non_join_consumers_untouched(self):
        flow, cal_x = self._saturating_flow()
        branch = _consumer(flow, "f1")
        before = float(branch.input_activation_scale)
        calibrate_fanin_boundary_scales(flow, cal_x, _S)
        assert float(branch.input_activation_scale) == pytest.approx(before)


class TestDistmatchWiresFaninCalibration:
    def test_match_activation_distributions_applies_floors(self):
        from mimarsinan.spiking.distribution_matching import (
            match_activation_distributions,
        )

        flow, cal_x = _build(_ResidualJoinMLP, gain=0.9)
        _prepare_ttfs(flow)
        teacher = copy.deepcopy(flow)
        stats = match_activation_distributions(
            flow, teacher, cal_x, _S, bias_iters=1,
        )
        assert stats["fanin_joins"] == 1
        tail = _consumer(flow, "tail")
        assert getattr(tail, "boundary_scale_floor", None) is not None


class TestResidualConcatBoundaryScale:
    """The on-chip merge concat carries lane-parallel traffic: a scalar wire scale
    must cover the widest lane (max), never the mean of heterogeneous branch θs."""

    def _scale(self, out_scales_values):
        mapper = _ResidualConcatMapper([])
        deps = list(range(len(out_scales_values)))
        out_scales = dict(enumerate(out_scales_values))
        return mapper.propagate_boundary_scale(deps, out_scales, 1.0)

    def test_equal_theta_unchanged(self):
        assert self._scale([1.36, 1.36]) == pytest.approx(1.36)

    def test_heterogeneous_takes_max(self):
        assert self._scale([0.8, 1.9]) == pytest.approx(1.9)

    def test_no_sources_falls_back_to_default(self):
        assert self._scale([]) == pytest.approx(1.0)
