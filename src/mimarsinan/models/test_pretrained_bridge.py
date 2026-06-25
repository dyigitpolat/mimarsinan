"""Unit tests for the pretrained bridge and its honest region descriptor.

The ``regime=pretrained`` region is MEASURED with the framework's own instruments
(``classify_validity`` + ``estimate_cores_needed``) on the converted IR graph of a
stock ImageNet-pretrained ResNet-18 -- no hand-rolled core counting, no training,
no GPU. The validity/capacity verdict is purely STRUCTURAL (shape-driven), so the
heavy descriptor tests run with random-weight construction (offline-safe); a
separate, network-gated test asserts the REAL pretrained weights load -- the
capability's actual payload.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import torch.nn as nn

from mimarsinan.models.pretrained_bridge import load_pretrained_resnet18


_SHAPE = (3, 32, 32)  # native 3-channel stem; small spatial keeps the core count sane
_NUM_CLASSES = 10


# ── Structural / build tests (offline-safe: random weights) ──────────────────


def test_builds_and_resizes_head():
    """The bridge builds a ResNet-18 and re-sizes only the fc head to num_classes."""
    model = load_pretrained_resnet18(_NUM_CLASSES, pretrained=False)
    assert isinstance(model, nn.Module)
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.out_features == _NUM_CLASSES
    # The convolutional trunk is kept verbatim: the native 3-channel stem stays 3ch.
    assert model.conv1.in_channels == 3


def test_forward_logits_shape():
    """Forward produces (batch, num_classes) logits for several class counts."""
    for num_classes in (10, 7):
        model = load_pretrained_resnet18(num_classes, pretrained=False).eval()
        x = torch.zeros(4, *_SHAPE)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (4, num_classes), (num_classes, y.shape)


def test_rejects_nonpositive_num_classes():
    """num_classes must be a positive class count."""
    with pytest.raises(ValueError):
        load_pretrained_resnet18(0, pretrained=False)


def test_composed_only_of_mappable_ops():
    """No attention / LayerNorm -- conv/bn/relu/pool/linear + residual add only."""
    forbidden = (nn.MultiheadAttention, nn.LayerNorm, nn.GroupNorm)
    model = load_pretrained_resnet18(_NUM_CLASSES, pretrained=False)
    for module in model.modules():
        assert not isinstance(module, forbidden), type(module).__name__
    # No grouped/depthwise convolution (un-mappable in the soft-core mapper).
    for module in model.modules():
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


def test_measured_validity_tier_is_valid_flagged():
    """Honest on-chip-fraction verdict: stock ResNet-18 maps as VALID_FLAGGED.

    A pure conv/bn/relu/pool/linear+residual net carries NO research-frontier op,
    but the residual ``add`` segment boundaries push the segment-start conv encoders
    host-side, so the on-chip PARAM majority is lost (param-minority) while the
    forward MACs stay overwhelmingly on-chip. That MAC-majority / param-minority
    split is exactly the located pretrained-residual frontier.
    """
    from mimarsinan.mapping.verification.onchip_fraction import (
        classify_validity,
        TIER_INVALID,
        TIER_VALID_FLAGGED,
    )

    verdict = classify_validity(
        load_pretrained_resnet18(_NUM_CLASSES, pretrained=False),
        _SHAPE,
        _NUM_CLASSES,
        encoding_placement="offload",
    )
    print(
        f"\n[ResNet18 MEASURED] tier={verdict.tier} "
        f"param_frac={verdict.param_frac:.6f} mac_frac={verdict.mac_frac:.6f} "
        f"research_gap_ops={verdict.research_gap_ops} "
        f"placement_fixable_ops={verdict.placement_fixable_ops}"
    )
    # Residual adds + BN absorb into conv -- NO unsupported research-frontier op.
    assert verdict.research_gap_ops == []
    assert verdict.tier != TIER_INVALID
    assert verdict.is_valid
    # MEASURED region descriptor (recorded) for the pretrained residual regime.
    assert verdict.tier == TIER_VALID_FLAGGED
    assert verdict.is_flagged
    assert verdict.param_frac == pytest.approx(0.423193, abs=1e-4)
    assert verdict.mac_frac == pytest.approx(0.998918, abs=1e-4)
    # MAC-majority on-chip but param-minority: the residual-boundary host cost.
    assert verdict.mac_frac >= 0.5
    assert verdict.param_frac < 0.5


def test_measured_capacity_estimate():
    """The honest core-count verdict from the framework capacity instrument."""
    from mimarsinan.config_schema.defaults import get_default_platform_constraints
    from mimarsinan.mapping.verification.capacity import estimate_cores_needed

    platform_constraints = get_default_platform_constraints()
    ir_graph = _build_ir_graph(
        load_pretrained_resnet18(_NUM_CLASSES, pretrained=False),
        _SHAPE,
        _NUM_CLASSES,
        platform_constraints,
    )
    estimate = estimate_cores_needed(ir_graph, platform_constraints)
    print(
        f"\n[ResNet18 MEASURED] cores_needed={estimate.cores_needed} "
        f"cores_available={estimate.cores_available} "
        f"feasible={estimate.feasible} scheduled={estimate.scheduled} "
        f"phase_count={estimate.phase_count} "
        f"peak_phase_cores={estimate.peak_phase_cores}"
    )
    # A static lower bound is always positive and well-defined.
    assert estimate.cores_needed > 0
    assert estimate.cores_available > 0
    assert estimate.phase_count >= 1
    # The 3x32x32 pretrained ResNet-18 fits the default 1000-core budget.
    assert estimate.feasible
    # MEASURED region descriptor (recorded against the default 1000-core budget).
    assert estimate.cores_available == 1000
    assert estimate.cores_needed == 651
    assert not estimate.scheduled
    assert estimate.phase_count == 1
    assert estimate.peak_phase_cores == estimate.cores_needed
    assert estimate.cores_needed < estimate.cores_available


# ── The capability payload: REAL pretrained weights load (network-gated) ──────


def test_loads_real_imagenet_weights():
    """The bridge imports GENUINE ImageNet1K pretrained weights (not random init).

    Skipped (not failed) when the weights are neither cached nor downloadable here,
    so the deterministic structural descriptor above never depends on the network.
    """
    try:
        model = load_pretrained_resnet18(_NUM_CLASSES, pretrained=True)
    except Exception as exc:  # noqa: BLE001 - any download/transport failure is a skip
        pytest.skip(f"pretrained weights unavailable in this environment: {exc}")

    # Pretrained conv weights are non-trivial (not a zero/constant random sentinel):
    # the first stem conv carries a learned, non-degenerate filter bank.
    stem_w = model.conv1.weight.detach()
    assert stem_w.shape == (64, 3, 7, 7)
    assert torch.isfinite(stem_w).all()
    assert stem_w.std().item() > 1e-3
    # Forward still produces task-sized logits with the re-sized head.
    with torch.no_grad():
        y = model(torch.zeros(1, *_SHAPE))
    assert y.shape == (1, _NUM_CLASSES)
