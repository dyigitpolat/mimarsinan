"""Wave-3 U3 — genuine annealed ramp for cascaded ttfs_cycle (default-OFF).

With ``ttfs_genuine_annealed_ramp=True`` (cascaded only) the GENUINE single-spike
cascade forward (``_SegmentSpikeForward``) is ``model.forward`` for the WHOLE
ramp; the committed rate drives the spike surrogate sharpness ``alpha``
smooth->sharp via ``TTFSGenuineAxis`` (rate 0 -> alpha_min, rate 1 -> alpha_max).

Because ``alpha`` is backward-only (exact ``pre>0`` Heaviside forward), the ramp
forward output is the exact deployed cascade at EVERY rate, so at rate 1 the ramp
forward is bit-identical to the finalize forward and the finalize cliff is ZERO by
construction. r=0 output is the genuine cascade at alpha_min (NOT the continuous
teacher), as intended — KD recovery adapts the weights.

These are MECHANISM tests; the empirical accuracy non-regression gate is a
separate full real-model run (out of scope here). Flag-OFF behavior must stay
byte-identical to the shipping value-domain proxy ramp.
"""

from __future__ import annotations

import copy

import pytest
import torch

from conftest import MockPipeline, default_config, make_tiny_supermodel

from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.tuning.axes.blend_axis import BlendAxis, TTFSGenuineAxis
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.perceptron_rate import rebuild_activations, set_blend_rate
from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import (
    TTFSCycleAdaptationTuner,
    _SegmentSpikeForward,
)


def _make_pipeline(tmp_path, *, schedule="cascaded", genuine=True):
    cfg = default_config()
    cfg["spiking_mode"] = "ttfs_cycle_based"
    cfg["ttfs_cycle_schedule"] = schedule
    cfg["activation_quantization"] = True
    cfg["tuning_budget_scale"] = 1.0
    cfg["simulation_steps"] = 16
    cfg["ttfs_genuine_annealed_ramp"] = genuine
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    return pipeline


def _make_tuner(tmp_path, *, schedule="cascaded", genuine=True):
    pipeline = _make_pipeline(tmp_path, schedule=schedule, genuine=genuine)
    model = make_tiny_supermodel()
    am = AdaptationManager()
    tuner = TTFSCycleAdaptationTuner(
        pipeline, model=model, target_accuracy=0.5,
        lr=pipeline.config["lr"], adaptation_manager=am,
    )
    return tuner, model, am


def _ttfs_nodes(model):
    return [m for m in model.modules() if isinstance(m, TTFSActivation)]


def _x(pipeline, n=3):
    return torch.randn(n, *pipeline.config["input_shape"])


# ── Flag ON: axis + alpha mechanism ───────────────────────────────────────────


class TestGenuineAxisIsActive:
    def test_axis_is_ttfs_genuine_under_flag(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, genuine=True)
        assert isinstance(tuner._axis, TTFSGenuineAxis)
        assert tuner._axis.name == "ttfs_genuine"

    def test_set_rate_anneals_alpha_endpoints(self, tmp_path):
        tuner, model, _ = _make_tuner(tmp_path, genuine=True)
        nodes = _ttfs_nodes(model)
        assert nodes, "genuine ramp must expose bare TTFSActivation nodes"

        tuner._set_rate(0.0)
        assert all(n.surrogate_alpha == pytest.approx(0.5) for n in nodes)
        tuner._set_rate(1.0)
        assert all(n.surrogate_alpha == pytest.approx(2.0) for n in nodes)
        # blend rate tracks too (the same per-perceptron .rate carriage).
        assert all(p.base_activation.rate == pytest.approx(1.0)
                   for p in model.get_perceptrons())

    def test_flag_off_axis_is_plain_blend(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, genuine=False)
        assert type(tuner._axis) is BlendAxis


# ── Flag ON: the genuine ramp forward is installed + bare TTFS nodes ──────────


class TestGenuineRampForwardInstalled:
    def test_ramp_forward_is_segment_spike_bound_to_model(self, tmp_path):
        tuner, model, _ = _make_tuner(tmp_path, genuine=True)
        fwd = tuner._ramp_forward()
        assert isinstance(fwd, _SegmentSpikeForward)
        assert fwd.model is model
        assert fwd.T == tuner._T

    def test_forward_installed_on_model_during_ramp(self, tmp_path):
        tuner, model, _ = _make_tuner(tmp_path, genuine=True)
        installed = model.__dict__.get("forward")
        assert isinstance(installed, _SegmentSpikeForward), (
            "the genuine annealed ramp must install the single-spike cascade as "
            "model.forward for the WHOLE ramp"
        )

    def test_base_activation_is_bare_ttfs_node(self, tmp_path):
        """Genuine mode bypasses the value blend: base_activation IS the bare
        TTFSActivation so the segment policy drives genuine nodes (no ReLU side of
        a blend corrupting the spike cascade)."""
        tuner, model, am = _make_tuner(tmp_path, genuine=True)
        assert am.ttfs_active is True
        for p in model.get_perceptrons():
            assert isinstance(p.base_activation, TTFSActivation)


# ── Flag ON: alpha is backward-only (forward alpha-invariant) ─────────────────


class TestAlphaIsBackwardOnly:
    def test_node_forward_is_alpha_invariant(self, tmp_path):
        """Focused node-level assert: the TTFSActivation forward output is
        identical across surrogate_alpha values (alpha shapes only the backward)."""
        tuner, _, _ = _make_tuner(tmp_path, genuine=True)
        node = TTFSActivation(T=tuner._T, activation_scale=1.0)
        node.set_cycle_accurate(True)
        v = torch.linspace(-0.3, 1.3, 50, dtype=torch.float32)

        outs = []
        for a in (0.25, 0.5, 1.0, 2.0, 8.0):
            node.set_surrogate_alpha(a)
            node.reset_state()
            outs.append(node(v).detach().clone())
        for o in outs[1:]:
            torch.testing.assert_close(o, outs[0], rtol=0, atol=0)

    def test_genuine_ramp_forward_is_alpha_invariant(self, tmp_path):
        """The whole genuine ramp forward output is invariant to the annealed
        alpha (the deployed cascade at every rate is the same VALUE)."""
        torch.manual_seed(3)
        tuner, model, _ = _make_tuner(tmp_path, genuine=True)
        ramp = model.__dict__.get("forward")
        x = _x(tuner.pipeline)

        tuner._set_rate(0.0)  # alpha_min
        with torch.no_grad():
            out_min = ramp(x).clone()
        tuner._set_rate(1.0)  # alpha_max
        with torch.no_grad():
            out_max = ramp(x).clone()
        torch.testing.assert_close(out_min, out_max, rtol=0, atol=0)

    def test_anneal_conditions_gradients_not_values(self, tmp_path):
        """The 'gradual' lives in gradient conditioning: forward is alpha-invariant
        (asserted above) while the parameter gradient through the genuine cascade
        DOES change with the annealed alpha (smooth->sharp surrogate)."""
        torch.manual_seed(4)
        tuner, model, _ = _make_tuner(tmp_path, genuine=True)
        ramp = model.__dict__.get("forward")
        x = _x(tuner.pipeline)
        param = next(p for p in model.parameters() if p.requires_grad)

        def grad_at(rate):
            tuner._set_rate(rate)
            model.zero_grad(set_to_none=True)
            ramp(x).sum().backward()
            return param.grad.detach().clone()

        g_smooth = grad_at(0.0)   # alpha_min
        g_sharp = grad_at(1.0)    # alpha_max
        assert not torch.allclose(g_smooth, g_sharp), (
            "annealing alpha must change gradient conditioning (else the ramp is "
            "not gradual in any meaningful sense)"
        )


# ── Flag ON: rate=1 ramp forward ≡ finalize forward (bit-exact) ───────────────


def _independent_finalize_forward(tuner, model):
    """Build the genuine forward exactly as finalize does, on an isolated clone:
    deepcopy -> blend rate 1.0 -> finalize-rebuild activations -> _SegmentSpikeForward."""
    device = tuner.pipeline.config["device"]
    clone = copy.deepcopy(model).to(device)
    set_blend_rate(clone, 1.0)
    rebuild_activations(clone, tuner.adaptation_manager, tuner.pipeline.config)
    return clone, _SegmentSpikeForward(clone, tuner._T)


class TestRateOneRampEqualsFinalize:
    def test_model_output_bit_identical_at_rate_one(self, tmp_path):
        torch.manual_seed(5)
        tuner, model, _ = _make_tuner(tmp_path, genuine=True)
        tuner._set_rate(1.0)  # alpha_max — the deployed dynamics
        ramp = model.__dict__.get("forward")
        x = _x(tuner.pipeline)

        clone, indep = _independent_finalize_forward(tuner, model)
        with torch.no_grad():
            got = ramp(x)
            expected = indep(x)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_node_values_bit_identical_at_rate_one(self, tmp_path):
        """forward_with_node_values is bit-identical too — the NF per-neuron parity
        side (gate #3): ramp-end node values == finalize node values."""
        torch.manual_seed(6)
        tuner, model, _ = _make_tuner(tmp_path, genuine=True)
        tuner._set_rate(1.0)
        x = _x(tuner.pipeline)

        ramp = model.__dict__.get("forward")
        ramp_exec = ramp._ensure_executor(ramp._build_executor)
        clone, indep = _independent_finalize_forward(tuner, model)
        indep_exec = indep._ensure_executor(indep._build_executor)

        with torch.no_grad():
            out_a, nodes_a = ramp_exec.forward_with_node_values(x)
            out_b, nodes_b = indep_exec.forward_with_node_values(x)

        torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)
        # node-value dicts are keyed by mapper-node identity (different objects on
        # the clone); compare the bit-exact per-node value tensors order-free.
        assert nodes_a and len(nodes_a) == len(nodes_b), "no per-neuron values recorded"
        stacked_a = sorted(tuple(v.flatten().tolist()) for v in nodes_a.values())
        stacked_b = sorted(tuple(v.flatten().tolist()) for v in nodes_b.values())
        assert stacked_a == stacked_b


# ── Flag ON: finalize is effectively a no-op (cliff ≈ 0) ──────────────────────


def _run_step(pipeline):
    from mimarsinan.pipelining.pipeline_steps.adaptation.ttfs_cycle_adaptation_step import (
        TTFSCycleAdaptationStep,
    )

    model = make_tiny_supermodel()
    am = AdaptationManager()
    pipeline.seed("model", model, step_name="Activation Quantization")
    pipeline.seed("adaptation_manager", am, step_name="Activation Quantization")
    step = TTFSCycleAdaptationStep(pipeline)
    step.name = "TTFS Cycle Fine-Tuning"
    pipeline.prepare_step(step)
    step.run()
    return step, model, am


class TestFinalizeIsNoOp:
    def test_finalize_cliff_is_near_zero(self, tmp_path):
        """End-to-end: under the genuine annealed ramp both the ramp-end metric
        (ramp forward @ alpha_max) and the post-finalize metric run the SAME
        deployed cascade, so the finalize cliff is ~0 (within a few accuracy_se on
        the tiny model)."""
        torch.manual_seed(9)
        pipeline = _make_pipeline(tmp_path, genuine=True)
        step, _, _ = _run_step(pipeline)
        cliff = step.tuner._finalize_cliff
        assert cliff is not None
        assert abs(float(cliff)) <= 0.2, (
            f"genuine annealed ramp finalize_cliff should be ~0, got {cliff}"
        )

    def test_deployed_forward_persists_genuine_cascade(self, tmp_path):
        """Finalize leaves the genuine single-spike cascade installed and
        bit-identical to a freshly built cascade on the deployed weights."""
        torch.manual_seed(10)
        pipeline = _make_pipeline(tmp_path, genuine=True)
        step, model, _ = _run_step(pipeline)
        installed = model.__dict__.get("forward")
        assert isinstance(installed, _SegmentSpikeForward)

        T = int(pipeline.config["simulation_steps"])
        x = _x(pipeline)
        fresh = _SegmentSpikeForward(model, T)
        with torch.no_grad():
            torch.testing.assert_close(model(x), fresh(x), rtol=0, atol=0)


# ── Flag OFF (default): value-domain proxy ramp byte-identical to today ────────


class TestFlagOffUnchanged:
    def test_ramp_forward_is_none(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, genuine=False)
        assert tuner._ramp_forward() is None

    def test_finalize_forward_is_segment_spike(self, tmp_path):
        tuner, model, _ = _make_tuner(tmp_path, genuine=False)
        fwd = tuner._finalize_forward()
        assert isinstance(fwd, _SegmentSpikeForward)
        assert fwd.model is model

    def test_base_activation_is_blend(self, tmp_path):
        """Default path keeps the value-domain BlendActivation (rate carriage),
        not the bare TTFS node."""
        tuner, model, _ = _make_tuner(tmp_path, genuine=False)
        for p in model.get_perceptrons():
            assert not isinstance(p.base_activation, TTFSActivation)
            assert hasattr(p.base_activation, "rate")
            assert isinstance(p.base_activation.target_activation, TTFSActivation)

    def test_no_forward_installed_during_ramp(self, tmp_path):
        tuner, model, _ = _make_tuner(tmp_path, genuine=False)
        assert "forward" not in model.__dict__


# ── Synchronized + flag ON: flag ignored ──────────────────────────────────────


class TestSynchronizedIgnoresFlag:
    def test_synchronized_ramp_forward_none_even_with_flag(self, tmp_path):
        tuner, model, _ = _make_tuner(tmp_path, schedule="synchronized", genuine=True)
        assert tuner._ramp_forward() is None
        assert "forward" not in model.__dict__

    def test_synchronized_axis_is_plain_blend_even_with_flag(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, schedule="synchronized", genuine=True)
        assert type(tuner._axis) is BlendAxis

    def test_synchronized_base_activation_is_blend_even_with_flag(self, tmp_path):
        tuner, model, _ = _make_tuner(tmp_path, schedule="synchronized", genuine=True)
        for p in model.get_perceptrons():
            assert not isinstance(p.base_activation, TTFSActivation)
            assert hasattr(p.base_activation, "rate")
