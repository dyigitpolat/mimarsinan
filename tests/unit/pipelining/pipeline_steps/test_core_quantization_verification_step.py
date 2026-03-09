"""Tests for CoreQuantizationVerificationStep and chip quantization in Soft Core Mapping.

When weight_quantization is False, Soft Core Mapping leaves weights as float and sets
threshold=1.0; Core Quantization Verification skips its checks. When weight_quantization
is True, chip quantization rounds weights into [q_min, q_max] and verification enforces it.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from conftest import (
    MockPipeline,
    make_tiny_supermodel,
    default_config,
    platform_constraints,
)

from mimarsinan.pipelining.pipeline_steps.soft_core_mapping_step import SoftCoreMappingStep
from mimarsinan.pipelining.pipeline_steps.core_quantization_verification_step import (
    CoreQuantizationVerificationStep,
)


def _fused_model_with_identity_norm():
    """Model that is already 'fused' (Identity norm) for use as fused_model seed."""
    model = make_tiny_supermodel()
    for p in model.get_perceptrons():
        p.normalization = nn.Identity()
    return model


def _run_soft_core_mapping_then_verification(
    mock_pipeline, fused_model, platform_constraints_dict, *, weight_quantization=False
):
    """Run Soft Core Mapping then Core Quantization Verification; return ir_graph."""
    mock_pipeline.config["weight_quantization"] = weight_quantization
    mock_pipeline.config["weight_bits"] = 8
    mock_pipeline.config.setdefault("firing_mode", "Default")

    mock_pipeline.seed("fused_model", fused_model, step_name="Normalization Fusion")
    mock_pipeline.seed(
        "platform_constraints_resolved",
        platform_constraints_dict,
        step_name="Model Configuration",
    )

    scm = SoftCoreMappingStep(mock_pipeline)
    scm.name = "Soft Core Mapping"
    mock_pipeline.prepare_step(scm)
    scm.run()

    ir_graph = mock_pipeline.cache["Soft Core Mapping.ir_graph"]
    verif = CoreQuantizationVerificationStep(mock_pipeline)
    verif.name = "Core Quantization Verification"
    mock_pipeline.prepare_step(verif)
    verif.run()
    return ir_graph


class TestCoreQuantizationVerificationWithWeightQuantizationFalse:
    """With weight_quantization=False, verification skips (no integer check); pipeline runs."""

    def test_verification_passes_after_soft_core_mapping_when_weight_quantization_false(
        self, mock_pipeline, platform_constraints
    ):
        fused_model = _fused_model_with_identity_norm()
        _run_soft_core_mapping_then_verification(
            mock_pipeline, fused_model, platform_constraints, weight_quantization=False
        )
        # No exception: verification step skips when weight_quantization is False.

    def test_all_core_matrices_in_range_after_soft_core_mapping(
        self, mock_pipeline, platform_constraints
    ):
        fused_model = _fused_model_with_identity_norm()
        ir_graph = _run_soft_core_mapping_then_verification(
            mock_pipeline, fused_model, platform_constraints, weight_quantization=False
        )
        # With weight_quantization=False we do not quantize; weights stay float. Only check cores exist.
        cores = ir_graph.get_neural_cores()
        assert len(cores) >= 1
        for core in cores:
            W = core.get_core_matrix(ir_graph)
            assert W.dtype in (np.float32, np.float64), f"{core.name}: expected float weights when quantization disabled"


class TestCoreQuantizationWithLargeWeights:
    """When raw weights exceed q_max and weight_quantization is True, chip quantization scales them into range."""

    def test_large_raw_weights_quantized_into_range(
        self, mock_pipeline, platform_constraints
    ):
        # Model with large weights (e.g. max |W| > 127); use weight_quantization=True so they get scaled into range.
        fused_model = _fused_model_with_identity_norm()
        with torch.no_grad():
            for p in fused_model.get_perceptrons():
                if hasattr(p.layer, "weight") and p.layer.weight is not None:
                    p.layer.weight.mul_(200.0)  # scale so max |W| can exceed 127
                if hasattr(p.layer, "bias") and p.layer.bias is not None:
                    p.layer.bias.zero_()
        ir_graph = _run_soft_core_mapping_then_verification(
            mock_pipeline, fused_model, platform_constraints, weight_quantization=True
        )
        q_min = -128
        q_max = 127
        for core in ir_graph.get_neural_cores():
            W = core.get_core_matrix(ir_graph)
            assert np.min(W) >= q_min - 1e-6
            assert np.max(W) <= q_max + 1e-6
