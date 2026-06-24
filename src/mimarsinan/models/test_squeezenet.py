"""Unit tests for the SqueezeNet conv vehicle and its honest region descriptor.

The descriptor is MEASURED with the framework's own instruments
(``classify_validity`` + ``estimate_cores_needed``) on the converted IR graph —
no hand-rolled core counting, no training, no GPU.
"""

import torch
import torch.nn as nn
import pytest

from mimarsinan.models.squeezenet import SqueezeNet, FireModule


_MNIST_SHAPE = (1, 28, 28)
_SVHN_SHAPE = (3, 32, 32)
_NUM_CLASSES = 10


def _build(input_shape=_MNIST_SHAPE, num_classes=_NUM_CLASSES, width=None):
    if width is None:
        return SqueezeNet(input_shape, num_classes)
    return SqueezeNet(input_shape, num_classes, width=width)


def _param_count(model):
    return int(sum(p.numel() for p in model.parameters()))


# ── Structural / build tests ─────────────────────────────────────────────────


def test_builds_for_in_channels_num_classes_input_size():
    """Builds for both the MNIST (1ch) and SVHN (3ch) input geometries."""
    m_mnist = _build(_MNIST_SHAPE, 10)
    m_svhn = _build(_SVHN_SHAPE, 10)
    assert isinstance(m_mnist, nn.Module)
    assert isinstance(m_svhn, nn.Module)


def test_forward_logits_shape():
    """Forward produces (batch, num_classes) logits for several configurations."""
    for shape in (_MNIST_SHAPE, _SVHN_SHAPE):
        for num_classes in (10, 7):
            model = _build(shape, num_classes).eval()
            x = torch.zeros(4, *shape)
            with torch.no_grad():
                y = model(x)
            assert y.shape == (4, num_classes), (shape, num_classes, y.shape)


def test_fire_module_squeeze_then_expand_concat():
    """A Fire module squeezes 1x1 then expands 1x1 + 3x3 and concatenates them."""
    fire = FireModule(in_channels=32, squeeze=8, expand1x1=16, expand3x3=16).eval()
    x = torch.zeros(2, 32, 7, 7)
    with torch.no_grad():
        y = fire(x)
    # Output channels = expand1x1 + expand3x3; spatial preserved (SAME padding).
    assert y.shape == (2, 32, 7, 7)
    assert fire.expand3x3.padding == (1, 1)  # SAME padding keeps it on the mappable path
    assert fire.squeeze.kernel_size == (1, 1)
    assert fire.expand1x1.kernel_size == (1, 1)
    assert fire.expand3x3.kernel_size == (3, 3)


def test_param_count_in_sane_squeezenet_range():
    """Scaled-down SqueezeNet param count sits in a sane sub-classic range."""
    n = _param_count(_build(_MNIST_SHAPE, 10))
    # Classic v1.1 ~ 1.2M; this scaled vehicle is smaller but recognisably a CNN.
    assert 100_000 <= n <= 1_200_000, n


def test_param_count_scales_with_width():
    """Widening the base channel budget grows the parameter count monotonically."""
    n_small = _param_count(_build(_MNIST_SHAPE, 10, width=16))
    n_big = _param_count(_build(_MNIST_SHAPE, 10, width=32))
    assert n_big > n_small


def test_composed_only_of_mappable_ops():
    """No attention / LayerNorm / normalization-norm ops — conv/relu/pool/linear only."""
    forbidden = (nn.MultiheadAttention, nn.LayerNorm, nn.GroupNorm)
    allowed = (
        nn.Conv2d,
        nn.ReLU,
        nn.MaxPool2d,
        nn.AdaptiveAvgPool2d,
        nn.AvgPool2d,
        nn.Linear,
        nn.Flatten,
        nn.Dropout,
    )
    for module in _build().modules():
        if isinstance(module, (SqueezeNet, FireModule, nn.Sequential, nn.ModuleList)):
            continue
        assert not isinstance(module, forbidden), type(module).__name__
        assert isinstance(module, allowed), type(module).__name__
    # No grouped/depthwise convolution (un-mappable in the soft-core mapper).
    for module in _build().modules():
        if isinstance(module, nn.Conv2d):
            assert module.groups == 1, module


# ── MEASURED region descriptor (framework instruments) ───────────────────────


def _build_ir_graph(model, input_shape, num_classes, platform_constraints):
    """Convert a native model to an IR graph the same way the pipeline does."""
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.platform.platform_constraints import (
        resolve_platform_mapping_params,
    )

    flow = convert_torch_model(
        model, tuple(input_shape), int(num_classes),
        encoding_layer_placement="offload",
    )
    mapper_repr = flow.get_mapper_repr()
    if hasattr(mapper_repr, "assign_perceptron_indices"):
        mapper_repr.assign_perceptron_indices()
    cores = platform_constraints["cores"]
    params = resolve_platform_mapping_params(
        cores, allow_coalescing=bool(platform_constraints.get("allow_coalescing", False))
    )
    ir_mapping = IRMapping(
        q_max=255,
        firing_mode="Default",
        max_axons=params.effective_max_axons,
        max_neurons=params.effective_max_neurons,
        allow_coalescing=params.allow_coalescing,
        hardware_bias=params.hardware_bias,
    )
    return ir_mapping.map(mapper_repr)


def test_measured_validity_tier_is_valid():
    """The honest on-chip-fraction verdict: SqueezeNet maps as a VALID conv vehicle."""
    from mimarsinan.mapping.verification.onchip_fraction import (
        classify_validity,
        TIER_INVALID,
    )

    verdict = classify_validity(
        _build(), _MNIST_SHAPE, _NUM_CLASSES, encoding_placement="offload"
    )
    print(
        f"\n[SqueezeNet MEASURED] tier={verdict.tier} "
        f"param_frac={verdict.param_frac:.4f} mac_frac={verdict.mac_frac:.4f} "
        f"research_gap_ops={verdict.research_gap_ops} "
        f"placement_fixable_ops={verdict.placement_fixable_ops}"
    )
    # A purely conv/relu/pool/linear model must carry NO research-frontier ops.
    assert verdict.research_gap_ops == []
    assert verdict.tier != TIER_INVALID
    assert verdict.is_valid
    # MEASURED region descriptor (recorded): fully on-chip under offload placement.
    assert verdict.tier == "VALID"
    assert verdict.param_frac == pytest.approx(1.0, abs=1e-6)
    assert verdict.mac_frac == pytest.approx(1.0, abs=1e-6)


def test_measured_capacity_estimate():
    """The honest core-count verdict from the framework capacity instrument."""
    from mimarsinan.config_schema.defaults import get_default_platform_constraints
    from mimarsinan.mapping.verification.capacity import estimate_cores_needed

    platform_constraints = get_default_platform_constraints()
    ir_graph = _build_ir_graph(
        _build(), _MNIST_SHAPE, _NUM_CLASSES, platform_constraints
    )
    estimate = estimate_cores_needed(ir_graph, platform_constraints)
    print(
        f"\n[SqueezeNet MEASURED] cores_needed={estimate.cores_needed} "
        f"cores_available={estimate.cores_available} "
        f"feasible={estimate.feasible} scheduled={estimate.scheduled} "
        f"phase_count={estimate.phase_count} "
        f"peak_phase_cores={estimate.peak_phase_cores}"
    )
    # A static lower bound is always positive and well-defined.
    assert estimate.cores_needed > 0
    assert estimate.cores_available > 0
    assert estimate.phase_count >= 1
    # The scaled-down vehicle fits the default 1000-core budget.
    assert estimate.feasible
    # MEASURED region descriptor (recorded against the default 1000-core budget).
    assert estimate.cores_available == 1000
    assert not estimate.scheduled
    assert estimate.phase_count == 1
    assert estimate.peak_phase_cores == estimate.cores_needed
    # The static diagonal lower bound is well under the budget (sanity band).
    assert estimate.cores_needed < estimate.cores_available
