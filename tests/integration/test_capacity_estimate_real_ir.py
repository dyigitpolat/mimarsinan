"""E4 capacity estimate on REAL IR graphs — reproduce the E3 scale-probe regimes.

E3 found HCM placement fails LATE ("No more hard cores available") at ImageNet
conv scale. ``estimate_cores_needed`` must reproduce E3's regime split from the
IR alone, EARLY: a small feasible net (lenet5) passes its budget; an
ImageNet-scale conv segment overflows a 1000-core budget by a wide margin. The
estimate is a SOUND lower bound, so its verdict never rejects a config the greedy
packer could actually place.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mimarsinan.config_schema.defaults import (
    get_default_deployment_parameters,
    get_default_platform_constraints,
)
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.platform.platform_constraints import (
    resolve_platform_mapping_params,
)
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.mapping.verification.capacity import (
    CapacityExceededError,
    estimate_cores_needed,
)
from mimarsinan.models.builders import BUILDERS_REGISTRY
from mimarsinan.torch_mapping import convert_torch_model
from mimarsinan.transformations.quantization_bounds import quantization_bounds


def _build_ir(model_type, input_shape, cores, allow_coalescing, num_classes=10):
    device = "cpu"
    cfg = {}
    cfg.update(get_default_deployment_parameters())
    cfg.update(get_default_platform_constraints())
    cfg.update({
        "model_type": model_type, "input_shape": list(input_shape),
        "input_size": int(np.prod(input_shape)), "num_classes": num_classes,
        "device": device, "allow_coalescing": allow_coalescing, "cores": cores,
        "firing_mode": "Default", "spike_generation_mode": "Uniform",
        "thresholding_mode": "<=", "cycle_accurate_lif_forward": True,
    })
    params = resolve_platform_mapping_params(cores, allow_coalescing=allow_coalescing)
    builder = BUILDERS_REGISTRY[model_type](
        device=device, input_shape=input_shape, num_classes=num_classes,
        pipeline_config=cfg,
    )
    flow = convert_torch_model(
        builder.build({}), input_shape, num_classes, device=device, strict=True,
    ).to(device).eval()
    _, q_max = quantization_bounds(cfg["weight_bits"])
    mr = flow.get_mapper_repr()
    mr.assign_perceptron_indices()
    compute_per_source_scales(mr)
    irm = IRMapping(
        q_max=q_max, firing_mode=cfg["firing_mode"],
        max_axons=params.effective_max_axons,
        max_neurons=params.effective_max_neurons,
        allow_coalescing=allow_coalescing, hardware_bias=params.hardware_bias,
    )
    return irm.map(mr)


_LENET_CORES = [
    {"max_axons": 784, "max_neurons": 512, "count": 60, "has_bias": True},
    {"max_axons": 512, "max_neurons": 256, "count": 60, "has_bias": True},
]


@pytest.mark.integration
def test_lenet5_is_feasible_on_its_budget():
    """The shipped lenet5 config (120-core budget, coalescing on) is feasible, and
    the SOUND lower bound stays at or below the budget (E3 measured 57 placed)."""
    ir = _build_ir("lenet5", (1, 28, 28), _LENET_CORES, allow_coalescing=True)
    est = estimate_cores_needed(
        ir, {"cores": _LENET_CORES, "allow_coalescing": True},
    )
    assert est.cores_available == 120
    assert est.feasible is True
    assert est.overflowing_segment is None
    # lower bound must never exceed what the packer actually placed (57)
    assert est.cores_needed <= 57
    est.raise_if_infeasible()  # no raise


@pytest.mark.integration
def test_deep_cnn_d8_feasible_under_its_suggested_hw_config():
    """No behavior change for a real small config: deep_cnn d8 maps feasible under
    the hardware config the project's OWN suggester derives, and the SOUND lower
    bound stays at or below that budget. The gate agrees with existing tooling —
    it only rejects configs that provably cannot fit."""
    from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
    from mimarsinan.mapping.verification.suggester.hw_config_suggester import (
        suggest_hardware_config,
    )
    from mimarsinan.mapping.verification.verifier import verify_soft_core_mapping
    from mimarsinan.models.builders.deep_cnn_builder import DeepCNNBuilder

    builder = DeepCNNBuilder("cpu", (1, 28, 28), 10, {})
    model = builder.build({"depth": 8, "width": 16, "base_activation": "ReLU"})
    flow = convert_torch_model(model, (1, 28, 28), 10, device="cpu", strict=True).eval()
    mr = flow.get_mapper_repr()
    mr.assign_perceptron_indices()
    compute_per_source_scales(mr)

    result = verify_soft_core_mapping(mr, max_axons=1024, max_neurons=1024)
    suggestion = suggest_hardware_config(result.softcores)
    cores = [dict(ct, has_bias=True) for ct in suggestion.core_types]
    params = resolve_platform_mapping_params(cores, allow_coalescing=False)
    ir = IRMapping(
        q_max=127.0, firing_mode="Default",
        max_axons=params.effective_max_axons,
        max_neurons=params.effective_max_neurons,
    ).map(mr)
    est = estimate_cores_needed(ir, {"cores": cores, "allow_coalescing": False})
    assert est.feasible is True
    assert est.cores_needed <= est.cores_available
    est.raise_if_infeasible()  # no raise


@pytest.mark.integration
@pytest.mark.slow
def test_vgg16_imagenet_is_infeasible_on_1000_core_budget():
    """VGG16@224 needs hundreds of thousands of cores on the realistic 1000-core
    ImageNet budget — the E3 headline overflow, surfaced EARLY as a verdict.

    Marked slow: the VGG16@224 IR build is the heavy ~40s mapping path.
    """
    cores = [{"max_axons": 256, "max_neurons": 256, "count": 1000, "has_bias": True}]
    ir = _build_ir("torch_vgg16", (3, 224, 224), cores, allow_coalescing=True)
    est = estimate_cores_needed(ir, {"cores": cores, "allow_coalescing": True})
    assert est.cores_available == 1000
    assert est.feasible is False
    assert est.cores_needed > 100_000  # hundreds of thousands (E3: ~316k lower bound)
    assert est.overflowing_segment is not None
    with pytest.raises(CapacityExceededError) as exc:
        est.raise_if_infeasible()
    assert exc.value.cores_needed == est.cores_needed
    assert exc.value.cores_available == 1000


def test_synthetic_imagenet_conv_segment_overflows_1000_budget():
    """Fast stand-in for the E3 headline: a synthetic early-conv segment of 50,176
    softcores of (576 axons, 64 neurons) — the exact ``features_6`` shape E3 named —
    overflows a 256x256x1000 budget. No model build, so this runs in CI."""
    n_softcores = 50_176
    in_count = 576  # 64 channels * 3*3 kernel
    out_count = 64
    nodes = []
    for i in range(n_softcores):
        nodes.append(
            NeuralCore(
                id=i, name=f"conv_{i}",
                input_sources=np.array(
                    [IRSource(node_id=-2, index=j % 16) for j in range(in_count)],
                    dtype=object,
                ),
                core_matrix=np.ones((in_count, out_count), dtype=np.float32),
                threshold=1.0, latency=0,
            )
        )
    graph = IRGraph(nodes=nodes, output_sources=np.array(
        [IRSource(node_id=n_softcores - 1, index=0)], dtype=object))
    cores = [{"max_axons": 256, "max_neurons": 256, "count": 1000, "has_bias": True}]
    est = estimate_cores_needed(graph, {"cores": cores, "allow_coalescing": True})
    # effective max_axons = 256 (has_bias) → frags = ceil(576/256) = 3 per core
    # neuron bound = ceil(50176*64 / 256) = 12544; axon bound = ceil(50176*576/256)
    # = 112896 dominates → far over the 1000 budget.
    assert est.feasible is False
    assert est.cores_needed > 100_000
    assert est.overflowing_segment is not None
