"""Scale-aware boundary calibration for the genuine TTFS single-spike cascade.

A fixed-window TTFS spike train only encodes [0,1], so each perceptron block
needs ``theta_out`` (activation_scale) normalizing its output to [0,1] and
``input_scale`` (input_activation_scale = upstream theta_out) un-normalizing the
[0,1]-encoded spike input back to the value domain before the linear op.
"""

import pytest
import torch

from conftest import (
    MockPipeline,
    make_tiny_supermodel,
)

from mimarsinan.spiking.scale_aware_boundaries import (
    calibrate_scale_aware_boundaries,
    propagate_boundary_input_scales,
)
from mimarsinan.spiking.segment_partition import perceptron_of
from mimarsinan.pipelining.pipeline_steps.adaptation.ttfs_cycle_adaptation_step import (
    TTFSCycleAdaptationStep,
)
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager


def _upstream_perceptron_source(model_repr, node):
    """Walk deps back through transparent nodes to the first perceptron source."""
    model_repr._ensure_exec_graph()
    stack = list(model_repr._deps.get(node, []))
    seen = set()
    while stack:
        dep = stack.pop(0)
        if dep is None or id(dep) in seen:
            continue
        seen.add(id(dep))
        if perceptron_of(dep) is not None:
            return perceptron_of(dep)
        stack.extend(model_repr._deps.get(dep, []))
    return None


def _perceptron_nodes_in_order(model_repr):
    model_repr._ensure_exec_graph()
    return [n for n in model_repr._exec_order if perceptron_of(n) is not None]


class TestCalibrateActivationScale:
    def test_activation_scale_set_to_provided_theta_out(self):
        model = make_tiny_supermodel()
        scales = [2.5, 3.0]
        calibrate_scale_aware_boundaries(model, scales)
        for p, s in zip(model.get_perceptrons(), scales):
            assert float(p.activation_scale) == pytest.approx(s)


class TestPropagateInputScale:
    def test_input_scale_equals_upstream_theta_out(self):
        model = make_tiny_supermodel()
        scales = [2.5, 3.0]
        calibrate_scale_aware_boundaries(model, scales)

        repr_ = model.get_mapper_repr()
        for node in _perceptron_nodes_in_order(repr_):
            p = perceptron_of(node)
            upstream = _upstream_perceptron_source(repr_, node)
            if upstream is None:
                continue
            assert float(p.input_activation_scale) == pytest.approx(
                float(upstream.activation_scale)
            ), "non-input perceptron input_scale must equal upstream theta_out"

    def test_input_boundary_uses_input_data_scale(self):
        model = make_tiny_supermodel()
        scales = [2.5, 3.0]
        input_data_scale = 0.75
        calibrate_scale_aware_boundaries(
            model, scales, input_data_scale=input_data_scale
        )

        repr_ = model.get_mapper_repr()
        for node in _perceptron_nodes_in_order(repr_):
            p = perceptron_of(node)
            if _upstream_perceptron_source(repr_, node) is None:
                assert float(p.input_activation_scale) == pytest.approx(
                    input_data_scale
                ), "input-boundary perceptron input_scale must be the data scale"

    def test_propagate_defaults_input_scale_to_one(self):
        model = make_tiny_supermodel()
        for p in model.get_perceptrons():
            p.set_activation_scale(1.5)
        propagate_boundary_input_scales(model)

        repr_ = model.get_mapper_repr()
        first = _perceptron_nodes_in_order(repr_)[0]
        assert float(perceptron_of(first).input_activation_scale) == pytest.approx(1.0)

    def test_accepts_model_repr_directly(self):
        model = make_tiny_supermodel()
        for p in model.get_perceptrons():
            p.set_activation_scale(2.0)
        propagate_boundary_input_scales(model.get_mapper_repr())

        repr_ = model.get_mapper_repr()
        nodes = _perceptron_nodes_in_order(repr_)
        downstream = perceptron_of(nodes[1])
        upstream = perceptron_of(nodes[0])
        assert float(downstream.input_activation_scale) == pytest.approx(
            float(upstream.activation_scale)
        )


class TestBoundaryEncodableProperty:
    def test_block_output_over_theta_out_in_unit_interval(self):
        """Each block's decoded output / theta_out lies within [0, 1] (+eps)."""
        model = make_tiny_supermodel()
        x = torch.randn(8, *(1, 8, 8))

        captured = {}

        def hook(perceptron, _inp, out):
            captured[id(perceptron)] = out.detach()

        handles = [p.register_forward_hook(hook) for p in model.get_perceptrons()]
        model.eval()
        with torch.no_grad():
            model(x)
        for h in handles:
            h.remove()

        # theta_out grounded in the activation distribution: the block's own max
        # (the percentile/max scale Activation Analysis would compute) normalizes
        # the output into [0,1].
        scales = [
            max(float(captured[id(p)].abs().max()), 1e-6)
            for p in model.get_perceptrons()
        ]
        calibrate_scale_aware_boundaries(model, scales)

        eps = 1e-5
        for p in model.get_perceptrons():
            out = captured[id(p)]
            theta_out = float(p.activation_scale)
            normalized = out / theta_out
            assert normalized.min().item() >= -eps
            assert normalized.max().item() <= 1.0 + eps


class TestFlagDefaultOff:
    def _seed(self, mock_pipeline, *, schedule="cascaded"):
        model = make_tiny_supermodel()
        am = AdaptationManager()
        mock_pipeline.config["spiking_mode"] = "ttfs_cycle_based"
        mock_pipeline.config["ttfs_cycle_schedule"] = schedule
        mock_pipeline.config["activation_quantization"] = True
        mock_pipeline.config["tuning_budget_scale"] = 1.0
        mock_pipeline.config.setdefault("simulation_steps", 16)
        mock_pipeline._target_metric = 0.5
        mock_pipeline.seed("model", model, step_name="Activation Quantization")
        mock_pipeline.seed(
            "adaptation_manager", am, step_name="Activation Quantization"
        )
        return model, am

    def test_flag_defaults_false(self, mock_pipeline):
        assert mock_pipeline.config.get("ttfs_scale_aware_boundaries", False) is False

    def test_flag_off_leaves_scales_unchanged(self, mock_pipeline):
        """Flag off: the ttfs_cycle step applies no calibration — activation_scale
        and input_activation_scale keep their pre-step values."""
        model, _ = self._seed(mock_pipeline)
        pre_act = [float(p.activation_scale) for p in model.get_perceptrons()]
        pre_in = [float(p.input_activation_scale) for p in model.get_perceptrons()]

        step = TTFSCycleAdaptationStep(mock_pipeline)
        step.name = "TTFS Cycle Fine-Tuning"
        mock_pipeline.prepare_step(step)
        step.run()

        post_act = [float(p.activation_scale) for p in model.get_perceptrons()]
        post_in = [float(p.input_activation_scale) for p in model.get_perceptrons()]
        assert post_act == pre_act
        assert post_in == pre_in
        assert all(v == 1.0 for v in post_in), (
            "flag-off must leave input_scale degenerate at 1.0 (byte-identical)"
        )

    def test_flag_off_requires_excludes_activation_scales(self, mock_pipeline):
        self._seed(mock_pipeline)
        step = TTFSCycleAdaptationStep(mock_pipeline)
        assert "activation_scales" not in step.requires


class TestFlagOnIntegration:
    def _seed(self, mock_pipeline, scales, *, schedule="cascaded"):
        model = make_tiny_supermodel()
        am = AdaptationManager()
        mock_pipeline.config["spiking_mode"] = "ttfs_cycle_based"
        mock_pipeline.config["ttfs_cycle_schedule"] = schedule
        mock_pipeline.config["activation_quantization"] = True
        mock_pipeline.config["tuning_budget_scale"] = 1.0
        mock_pipeline.config["ttfs_scale_aware_boundaries"] = True
        mock_pipeline.config.setdefault("simulation_steps", 16)
        mock_pipeline._target_metric = 0.5
        mock_pipeline.seed("model", model, step_name="Activation Quantization")
        mock_pipeline.seed(
            "adaptation_manager", am, step_name="Activation Quantization"
        )
        mock_pipeline.seed(
            "activation_scales", scales, step_name="Activation Analysis"
        )
        return model, am

    def test_flag_on_adds_activation_scales_to_requires(self, mock_pipeline):
        self._seed(mock_pipeline, [2.0, 2.0])
        step = TTFSCycleAdaptationStep(mock_pipeline)
        assert "activation_scales" in step.requires

    def test_flag_on_calibrates_before_tuner(self, mock_pipeline):
        scales = [2.5, 3.0]
        model, _ = self._seed(mock_pipeline, scales)
        step = TTFSCycleAdaptationStep(mock_pipeline)
        step.name = "TTFS Cycle Fine-Tuning"
        mock_pipeline.prepare_step(step)
        step.run()

        perceptrons = model.get_perceptrons()
        for p, s in zip(perceptrons, scales):
            assert float(p.activation_scale) == pytest.approx(s)
        # Downstream perceptron's input_scale equals upstream theta_out.
        assert float(perceptrons[1].input_activation_scale) == pytest.approx(
            scales[0]
        )
        # Input-boundary perceptron's input_scale defaults to 1.0 ([0,1] inputs).
        assert float(perceptrons[0].input_activation_scale) == pytest.approx(1.0)
