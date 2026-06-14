"""P5a: in-place RateBuffer for the AdaptationManager family (flag tuning_inplace_rate).

The buffer path mutates one shared ``alpha`` buffer in place (O(1)) instead of
rebuilding every perceptron's decorator stack. The conformance contract is two-
fold: (1) outputs bit-match the rebuild path at a rate grid that includes exact
0.0 and 1.0, and (2) the global torch RNG state is identical after
``set_rate(0.0) + forward`` via the buffer path vs the rebuild path — i.e. the
``rate == 0.0`` short-circuit's ``torch.rand`` skip is preserved bit-exact.
"""

from __future__ import annotations

import copy

import pytest
import torch

from conftest import make_tiny_supermodel, default_config
from mimarsinan.models.nn.decorators.rate_buffer import RateBuffer
from mimarsinan.models.nn.decorators.adjustment import (
    RateAdjustedDecorator,
    RandomMaskAdjustmentStrategy,
    MixAdjustmentStrategy,
    NestedAdjustmentStrategy,
)
from mimarsinan.models.nn.layers import (
    NestedDecoration,
    ShiftDecorator,
    QuantizeDecorator,
)
from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
    create_adaptation_manager_for_model,
)
from mimarsinan.tuning.perceptron_rate import apply_manager_rate
from mimarsinan.tuning.axes import ClampAxis, ActQuantAxis, ActivationAdaptationAxis


RATE_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def _fwd(model, x, seed=0):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        return model(x).clone()


# ---------------------------------------------------------------------------
# RateBuffer unit behavior
# ---------------------------------------------------------------------------

class TestRateBuffer:
    def test_registered_buffer_zero_init(self):
        rb = RateBuffer()
        assert "alpha" in dict(rb.named_buffers())
        assert rb.alpha.shape == ()
        assert float(rb.alpha) == 0.0

    def test_set_fills_in_place_same_storage(self):
        rb = RateBuffer()
        storage_before = rb.alpha.data_ptr()
        rb.set(0.6)
        # In-place: no rebuild, same underlying tensor storage.
        assert rb.alpha.data_ptr() == storage_before
        # float32 carrier — exact within float32 round-trip.
        assert float(rb.alpha) == pytest.approx(0.6, abs=1e-6)

    def test_float_protocol(self):
        rb = RateBuffer()
        rb.set(0.3)
        assert float(rb) == pytest.approx(0.3, abs=1e-6)

    def test_set_accepts_int_and_float(self):
        rb = RateBuffer()
        rb.set(1)
        assert float(rb.alpha) == 1.0
        rb.set(0.0)
        assert float(rb.alpha) == 0.0


# ---------------------------------------------------------------------------
# RateAdjustedDecorator: buffer rate == float rate, bit-for-bit + RNG parity
# ---------------------------------------------------------------------------

def _quant_decorator(rate):
    """The stochastic quant decorator shape (carries the RandomMask RNG-skip)."""
    return RateAdjustedDecorator(
        rate,
        NestedDecoration([
            ShiftDecorator(torch.tensor(0.1)),
            QuantizeDecorator(torch.tensor(4.0), torch.tensor(2.0)),
        ]),
        NestedAdjustmentStrategy([
            RandomMaskAdjustmentStrategy(), MixAdjustmentStrategy(),
        ]),
    )


class TestRateAdjustedDecoratorBuffer:
    def test_output_bit_matches_float_across_grid(self):
        x = torch.randn(8, 16)
        for r in RATE_GRID:
            torch.manual_seed(123)
            float_out = _quant_decorator(r).input_transform(x)

            rb = RateBuffer()
            rb.set(r)
            torch.manual_seed(123)
            buf_out = _quant_decorator(rb).input_transform(x)

            assert torch.equal(float_out, buf_out), f"mismatch at rate {r}"

    def test_rng_state_identical_after_zero_rate(self):
        """rate==0.0 must skip the RandomMask torch.rand on BOTH paths."""
        x = torch.randn(8, 16)

        torch.manual_seed(7)
        _quant_decorator(0.0).input_transform(x)
        float_state = torch.get_rng_state()

        rb = RateBuffer()
        rb.set(0.0)
        torch.manual_seed(7)
        _quant_decorator(rb).input_transform(x)
        buf_state = torch.get_rng_state()

        assert torch.equal(float_state, buf_state)

    def test_rng_state_identical_after_nonzero_rate(self):
        """rate in (0,1): both paths must consume the SAME torch.rand draw."""
        x = torch.randn(8, 16)

        torch.manual_seed(11)
        _quant_decorator(0.5).input_transform(x)
        float_state = torch.get_rng_state()

        rb = RateBuffer()
        rb.set(0.5)
        torch.manual_seed(11)
        _quant_decorator(rb).input_transform(x)
        buf_state = torch.get_rng_state()

        assert torch.equal(float_state, buf_state)

    def test_one_rate_short_circuits_to_target(self):
        x = torch.randn(4, 16)
        rb = RateBuffer()
        rb.set(1.0)
        out_buf = _quant_decorator(rb).input_transform(x)
        out_float = _quant_decorator(1.0).input_transform(x)
        assert torch.equal(out_buf, out_float)


# ---------------------------------------------------------------------------
# Axis conformance: buffer-path vs rebuild-path (the critical contract)
# ---------------------------------------------------------------------------

def _buffer_config():
    cfg = default_config()
    cfg["tuning_inplace_rate"] = True
    return cfg


def _make_axis(axis_cls, cfg, model, manager):
    axis = axis_cls()
    axis.attach(model, manager, cfg)
    return axis


class TestAxisConformance:
    def _conformance(self, axis_cls):
        x = torch.randn(3, 1, 8, 8)

        # Buffer path (flag on).
        cfg_buf = _buffer_config()
        model_buf = make_tiny_supermodel()
        mgr_buf = create_adaptation_manager_for_model(cfg_buf, model_buf)
        axis_buf = _make_axis(axis_cls, cfg_buf, model_buf, mgr_buf)

        # Rebuild path (flag off — legacy).
        cfg_reb = default_config()
        model_reb = make_tiny_supermodel()
        mgr_reb = create_adaptation_manager_for_model(cfg_reb, model_reb)
        axis_reb = _make_axis(axis_cls, cfg_reb, model_reb, mgr_reb)
        # Share weights so only the rate-application path differs.
        model_reb.load_state_dict(model_buf.state_dict(), strict=False)

        for r in RATE_GRID:
            torch.manual_seed(999)
            axis_buf.set_rate(r)
            buf_out = _fwd(model_buf, x)

            torch.manual_seed(999)
            axis_reb.set_rate(r)
            reb_out = _fwd(model_reb, x)

            assert torch.allclose(buf_out, reb_out, atol=0.0, rtol=0.0), (
                f"{axis_cls.__name__}: output mismatch at rate {r}"
            )

    def test_clamp_axis_conformance(self):
        self._conformance(ClampAxis)

    def test_act_quant_axis_conformance(self):
        self._conformance(ActQuantAxis)

    def test_activation_adaptation_axis_conformance(self):
        self._conformance(ActivationAdaptationAxis)

    def test_rng_state_identical_after_zero_rate_forward(self):
        """Critical: global torch RNG identical after set_rate(0.0)+forward,
        buffer path vs rebuild path (pins the rate==0 RNG-skip end to end)."""
        x = torch.randn(3, 1, 8, 8)

        cfg_buf = _buffer_config()
        model_buf = make_tiny_supermodel()
        mgr_buf = create_adaptation_manager_for_model(cfg_buf, model_buf)
        axis_buf = _make_axis(ActQuantAxis, cfg_buf, model_buf, mgr_buf)
        # Prime the buffer install at a nonzero rate first (realistic ramp), then
        # drop to 0.0; the forward must still skip the RandomMask draw.
        axis_buf.set_rate(0.5)

        cfg_reb = default_config()
        model_reb = make_tiny_supermodel()
        mgr_reb = create_adaptation_manager_for_model(cfg_reb, model_reb)
        axis_reb = _make_axis(ActQuantAxis, cfg_reb, model_reb, mgr_reb)
        model_reb.load_state_dict(model_buf.state_dict(), strict=False)
        axis_reb.set_rate(0.5)

        torch.manual_seed(31337)
        axis_buf.set_rate(0.0)
        _fwd(model_buf, x, seed=None)
        buf_state = torch.get_rng_state()

        torch.manual_seed(31337)
        axis_reb.set_rate(0.0)
        _fwd(model_reb, x, seed=None)
        reb_state = torch.get_rng_state()

        assert torch.equal(buf_state, reb_state)


class TestInPlaceSemantics:
    def test_no_rebuild_after_install(self):
        """After the one-time install, set_rate mutates alpha in place — the
        decorator objects keep their identity (no per-perceptron rebuild)."""
        cfg = _buffer_config()
        model = make_tiny_supermodel()
        mgr = create_adaptation_manager_for_model(cfg, model)
        axis = ActQuantAxis()
        axis.attach(model, mgr, cfg)

        axis.set_rate(0.3)
        first_decorators = [
            tuple(p.activation.decorators) for p in model.get_perceptrons()
        ]
        buffer = mgr._rate_buffer("quantization_rate")
        alpha_storage = buffer.alpha.data_ptr()

        axis.set_rate(0.8)
        second_decorators = [
            tuple(p.activation.decorators) for p in model.get_perceptrons()
        ]

        # Same decorator objects (no rebuild) and same buffer storage (in place).
        assert first_decorators == second_decorators
        assert buffer.alpha.data_ptr() == alpha_storage
        assert float(buffer.alpha) == pytest.approx(0.8, abs=1e-6)

    def test_reattach_reinstalls_buffer_stack(self):
        """A fresh attach (new model/manager) re-runs the one-time install."""
        cfg = _buffer_config()
        axis = ActQuantAxis()

        model_a = make_tiny_supermodel()
        mgr_a = create_adaptation_manager_for_model(cfg, model_a)
        axis.attach(model_a, mgr_a, cfg)
        axis.set_rate(0.5)
        assert mgr_a._rate_buffer("quantization_rate") is not None

        model_b = make_tiny_supermodel()
        mgr_b = create_adaptation_manager_for_model(cfg, model_b)
        axis.attach(model_b, mgr_b, cfg)
        axis.set_rate(0.5)
        buffer_b = mgr_b._rate_buffer("quantization_rate")
        assert buffer_b is not None
        for p in model_b.get_perceptrons():
            assert p.activation.decorators[-1].rate is buffer_b

    def test_buffer_shared_across_perceptrons(self):
        """One buffer drives every perceptron's decorator (a single write ramps all)."""
        cfg = _buffer_config()
        model = make_tiny_supermodel()
        mgr = create_adaptation_manager_for_model(cfg, model)
        axis = ActQuantAxis()
        axis.attach(model, mgr, cfg)
        axis.set_rate(0.5)

        buffer = mgr._rate_buffer("quantization_rate")
        for p in model.get_perceptrons():
            quant = p.activation.decorators[-1]
            assert quant.rate is buffer


class TestFlagOffUnchanged:
    def test_flag_off_uses_rebuild_path(self):
        """Flag-off set_rate is byte-identical to apply_manager_rate (no buffer)."""
        cfg = default_config()  # flag absent → off
        x = torch.randn(2, 1, 8, 8)

        model_axis = make_tiny_supermodel()
        model_direct = copy.deepcopy(model_axis)
        mgr_axis = create_adaptation_manager_for_model(cfg, model_axis)
        mgr_direct = create_adaptation_manager_for_model(cfg, model_direct)

        axis = ActQuantAxis()
        axis.attach(model_axis, mgr_axis, cfg)
        axis.set_rate(0.5)
        apply_manager_rate(model_direct, mgr_direct, cfg, "quantization_rate", 0.5)

        assert mgr_axis.quantization_rate == mgr_direct.quantization_rate == 0.5
        # Flag-off path installs no RateBuffer on the manager.
        assert not getattr(mgr_axis, "_rate_buffers", {})
        assert torch.allclose(_fwd(model_axis, x), _fwd(model_direct, x))

    def test_buffer_path_state_carriage_roundtrip(self):
        """get/set_extra_state still round-trips under the buffer path."""
        cfg = _buffer_config()
        model = make_tiny_supermodel()
        mgr = create_adaptation_manager_for_model(cfg, model)
        axis = ActQuantAxis()
        axis.attach(model, mgr, cfg)

        axis.set_rate(0.4)
        extra = axis.get_extra_state()
        assert float(extra) == pytest.approx(0.4, abs=1e-6)

        axis.set_rate(0.9)
        assert float(axis.get_extra_state()) == pytest.approx(0.9, abs=1e-6)
        axis.set_extra_state(extra)
        assert float(axis.get_extra_state()) == pytest.approx(0.4, abs=1e-6)
