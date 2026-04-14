"""
Diagnostic test for JointArchHwProblem evaluation with TorchMLPMixer.

Verifies that _evaluate_inner does not silently fail and return penalty
objectives for valid model configs and platform constraints.
"""

import pytest
import torch

from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory
import mimarsinan.data_handling.data_providers.mnist_data_provider  # noqa: F401 (register)
from mimarsinan.models.builders.torch_mlp_mixer_core_builder import (
    TorchMLPMixerCoreBuilder,
)
from mimarsinan.search.problems.joint_arch_hw_problem import JointArchHwProblem
from mimarsinan.search.results import resolve_active_objectives


def _make_data_provider_factory():
    return BasicDataProviderFactory(
        "MNIST_DataProvider", "./datasets", seed=0,
    )


def _make_model_config():
    return {
        "base_activation": "LeakyReLU",
        "patch_n_1": 4,
        "patch_m_1": 4,
        "patch_c_1": 32,
        "fc_w_1": 64,
        "fc_w_2": 64,
    }


def _make_platform_constraints():
    return {
        "cores": [
            {"max_axons": 256, "max_neurons": 256, "count": 200},
        ],
        "target_tq": 32,
        "weight_bits": 8,
    }


def _make_problem(
    search_mode="joint",
    objective_names=None,
):
    if objective_names is None:
        objective_names = [
            "estimated_accuracy",
            "total_params",
            "total_param_capacity",
            "total_sync_barriers",
            "param_utilization_pct",
            "neuron_wastage_pct",
            "axon_wastage_pct",
            "fragmentation_pct",
        ]

    input_shape = (1, 28, 28)
    return JointArchHwProblem(
        data_provider_factory=_make_data_provider_factory(),
        device=torch.device("cpu"),
        input_shape=input_shape,
        num_classes=10,
        target_tq=32,
        lr=0.001,
        search_mode=search_mode,
        builder_factory=TorchMLPMixerCoreBuilder,
        arch_options=(),
        model_config_assembler=lambda raw: raw,
        validate_fn=TorchMLPMixerCoreBuilder.validate_config,
        active_objective_names=objective_names,
        accuracy_seed=0,
        warmup_fraction=0.10,
        accuracy_evaluator="extrapolating",
        extrapolation_num_train_epochs=1,
        extrapolation_num_checkpoints=2,
        extrapolation_target_epochs=10,
    )


class TestEvaluateInnerDoesNotReturnPenalties:
    """The core regression: evaluate must return real metrics, not penalties."""

    def test_build_model_succeeds(self):
        """_build_model should succeed for a valid mlp_mixer config."""
        problem = _make_problem()
        mc = _make_model_config()
        pcfg = _make_platform_constraints()
        model, total_params = problem._build_model(mc, pcfg)
        assert total_params > 0
        assert hasattr(model, "get_mapper_repr"), (
            "Model should be converted to torch-mapped flow (has get_mapper_repr)"
        )

    def test_collect_softcores_succeeds(self):
        """_collect_softcores should produce a non-empty list."""
        problem = _make_problem()
        mc = _make_model_config()
        pcfg = _make_platform_constraints()
        model, _ = problem._build_model(mc, pcfg)
        softcores, host_segments = problem._collect_softcores(model, pcfg)
        assert len(softcores) > 0, "Softcores list should not be empty"

    def test_hw_objectives_not_none(self):
        """_compute_hw_objectives should return a non-None dict (feasible packing)."""
        problem = _make_problem()
        mc = _make_model_config()
        pcfg = _make_platform_constraints()
        model, total_params = problem._build_model(mc, pcfg)
        softcores, host_segments = problem._collect_softcores(model, pcfg)
        hw_obj, error = problem._compute_hw_objectives(
            softcores, pcfg, total_params, host_segments,
        )
        assert error is None, f"Packing should be feasible, got: {error}"
        assert hw_obj is not None, "Packing should be feasible"
        assert hw_obj["total_params"] == total_params

    def test_evaluate_inner_returns_real_objectives(self):
        """_evaluate_inner must not raise; returned objectives must not be penalties."""
        problem = _make_problem()
        mc = _make_model_config()
        pcfg = _make_platform_constraints()
        obj = problem._evaluate_inner(mc, pcfg)
        assert obj["total_params"] < 1e17, (
            f"total_params looks like a penalty: {obj['total_params']}"
        )
        assert obj.get("estimated_accuracy", -1) >= 0.0

    def test_evaluate_returns_real_objectives(self):
        """Full evaluate() must not return penalty values."""
        problem = _make_problem()
        config = {
            "model_config": _make_model_config(),
            "platform_constraints": _make_platform_constraints(),
        }
        obj = problem.evaluate(config)
        penalty = problem._penalty_objectives()
        assert obj != penalty, (
            f"evaluate() returned penalty objectives: {obj}"
        )

    def test_hw_only_objectives_no_accuracy(self):
        """HW objectives alone (no accuracy) should return real values."""
        problem = _make_problem(
            objective_names=[
                "total_params",
                "total_param_capacity",
                "total_sync_barriers",
                "param_utilization_pct",
                "neuron_wastage_pct",
                "axon_wastage_pct",
            ],
        )
        mc = _make_model_config()
        pcfg = _make_platform_constraints()
        obj = problem._evaluate_inner(mc, pcfg)
        assert obj["total_params"] < 1e17
        assert "estimated_accuracy" not in obj


class TestCoalescingFlagAlignment:
    """Canonical ``allow_coalescing`` controls layout width vs packing feasibility."""

    _WIDE_BASE = {
        # Enough cores for coalescing-expanded fragments (layout can be wide).
        "cores": [{"max_axons": 16, "max_neurons": 64, "count": 2000}],
        "target_tq": 32,
        "weight_bits": 8,
    }

    def test_allow_coalescing_false_blocks_wide_packing(self):
        """With coalescing off, the wide layout scenario should fail HW packing."""
        from mimarsinan.mapping.coalescing import normalize_coalescing_config

        problem = _make_problem(
            objective_names=[
                "total_params",
                "param_utilization_pct",
                "neuron_wastage_pct",
                "axon_wastage_pct",
                "fragmentation_pct",
            ],
        )
        mc = _make_model_config()
        pcfg = {**self._WIDE_BASE, "allow_coalescing": False}
        normalize_coalescing_config(pcfg)

        model, total_params = problem._build_model(mc, pcfg)
        softcores, host_segments = problem._collect_softcores(model, pcfg)
        hw_obj, error = problem._compute_hw_objectives(
            softcores, pcfg, total_params, host_segments,
        )
        assert hw_obj is None
        assert error is not None


class TestAgentEvolveLikeConfigs:
    """Reproduce agent-evolved configs that all returned penalties in production."""

    KEDI_CONFIGS = [
        (
            {"base_activation": "ReLU", "fc_w_1": 32, "fc_w_2": 32,
             "patch_c_1": 24, "patch_m_1": 4, "patch_n_1": 4},
            {"cores": [
                {"count": 80, "max_axons": 128, "max_neurons": 128},
                {"count": 80, "max_axons": 192, "max_neurons": 192},
                {"count": 80, "max_axons": 128, "max_neurons": 128},
            ]},
        ),
        (
            {"base_activation": "GELU", "fc_w_1": 96, "fc_w_2": 64,
             "patch_c_1": 64, "patch_m_1": 7, "patch_n_1": 7},
            {"cores": [
                {"count": 150, "max_axons": 256, "max_neurons": 256},
                {"count": 150, "max_axons": 384, "max_neurons": 384},
                {"count": 150, "max_axons": 256, "max_neurons": 256},
            ]},
        ),
        (
            {"base_activation": "LeakyReLU", "fc_w_1": 192, "fc_w_2": 256,
             "patch_c_1": 128, "patch_m_1": 14, "patch_n_1": 14},
            {"cores": [
                {"count": 800, "max_axons": 384, "max_neurons": 384},
                {"count": 800, "max_axons": 512, "max_neurons": 512},
                {"count": 800, "max_axons": 384, "max_neurons": 384},
            ]},
        ),
    ]

    @pytest.mark.parametrize("mc,pcfg", KEDI_CONFIGS,
                             ids=["small_relu", "medium_gelu", "large_leaky"])
    def test_evaluate_inner_no_penalty(self, mc, pcfg):
        """_evaluate_inner must not raise for configs matching agent-evolved output."""
        problem = _make_problem(
            objective_names=[
                "total_params", "total_param_capacity", "total_sync_barriers",
                "param_utilization_pct", "neuron_wastage_pct", "axon_wastage_pct",
            ],
        )
        obj = problem._evaluate_inner(mc, pcfg)
        assert obj["total_params"] < 1e17, f"Got penalty: {obj}"

    @pytest.mark.parametrize("mc,pcfg", KEDI_CONFIGS,
                             ids=["small_relu", "medium_gelu", "large_leaky"])
    def test_full_evaluate_with_accuracy(self, mc, pcfg):
        """Full evaluate() with accuracy must return real values."""
        problem = _make_problem()
        config = {"model_config": mc, "platform_constraints": pcfg}
        obj = problem.evaluate(config)
        penalty = problem._penalty_objectives()
        assert obj != penalty, f"evaluate() returned penalty: {obj}"


class TestPerPhaseErrorHandling:
    """Accuracy failure should not discard HW objectives and vice versa."""

    def test_accuracy_failure_preserves_hw_objectives(self):
        """If accuracy evaluation raises, HW objectives should still be real."""
        problem = _make_problem()
        mc = _make_model_config()
        pcfg = _make_platform_constraints()

        original_evaluate_accuracy = problem._evaluate_accuracy
        def _failing_accuracy(model):
            raise RuntimeError("Simulated accuracy failure")
        problem._evaluate_accuracy = _failing_accuracy

        obj = problem._evaluate_inner(mc, pcfg)
        problem._evaluate_accuracy = original_evaluate_accuracy

        assert obj["total_params"] < 1e17, "HW objectives should be real"
        assert obj["estimated_accuracy"] == 0.0, "Accuracy should be 0.0 on failure"

    def test_hw_failure_rejected_at_validate(self):
        """HW conversion failure should be caught by validate_detailed."""
        problem = _make_problem()
        config = {
            "model_config": _make_model_config(),
            "platform_constraints": _make_platform_constraints(),
        }

        original_ensure = problem._ensure_mapper_repr
        def _failing_convert(model):
            raise RuntimeError("Simulated conversion failure")
        problem._ensure_mapper_repr = _failing_convert

        vr = problem.validate_detailed(config)
        problem._ensure_mapper_repr = original_ensure

        assert not vr.is_valid
        assert vr.failure_phase == "hw_conversion"
        assert "Simulated conversion failure" in vr.error_message

    def test_packing_infeasible_rejected_at_validate(self):
        """Infeasible packing should be caught by validate_detailed with diagnostics."""
        tiny_pcfg = {
            "cores": [
                {"max_axons": 256, "max_neurons": 256, "count": 1},
            ],
            "target_tq": 32,
            "weight_bits": 8,
        }
        problem = _make_problem()
        config = {
            "model_config": _make_model_config(),
            "platform_constraints": tiny_pcfg,
        }

        vr = problem.validate_detailed(config)

        assert not vr.is_valid
        assert vr.failure_phase == "hw_packing"
        assert "softcores=" in vr.error_message
        assert "total_hw_capacity=" in vr.error_message


class TestValidationCacheIntegration:
    """Validation cache must avoid redundant model builds."""

    def test_validate_caches_model_for_evaluate(self):
        """Model should be built once across validate_detailed + evaluate."""
        problem = _make_problem()
        config = {
            "model_config": _make_model_config(),
            "platform_constraints": _make_platform_constraints(),
        }

        build_count = 0
        original_build = problem._build_raw_model

        def _counting_build(mc, pcfg):
            nonlocal build_count
            build_count += 1
            return original_build(mc, pcfg)

        problem._build_raw_model = _counting_build

        vr = problem.validate_detailed(config)
        assert vr.is_valid

        obj = problem.evaluate(config)
        problem._build_raw_model = original_build

        assert obj["total_params"] < 1e17, "Should return real objectives"
        assert build_count == 1, (
            f"_build_raw_model called {build_count} times, expected 1"
        )

    def test_evaluate_standalone_works(self):
        """evaluate() without prior validate_detailed() must work end-to-end."""
        problem = _make_problem()
        config = {
            "model_config": _make_model_config(),
            "platform_constraints": _make_platform_constraints(),
        }

        obj = problem.evaluate(config)
        penalty = problem._penalty_objectives()
        assert obj != penalty, f"evaluate() returned penalty: {obj}"
        assert obj["total_params"] < 1e17
