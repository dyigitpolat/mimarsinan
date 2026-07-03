"""Unit tests for the pretrained bridge and its MEASURED region descriptor."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import torch.nn as nn

from mimarsinan.models.pretrained_bridge import (
    DeployedEval,
    deploy_and_eval,
    load_pretrained_resnet18,
    load_pretrained_resnet50,
)


_SHAPE = (3, 32, 32)
_NUM_CLASSES = 10


def test_builds_and_resizes_head():
    """The bridge builds a ResNet-18 and re-sizes only the fc head to num_classes."""
    model = load_pretrained_resnet18(_NUM_CLASSES, pretrained=False)
    assert isinstance(model, nn.Module)
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.out_features == _NUM_CLASSES
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
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            assert module.groups == 1, module


def _build_ir_graph(
    model, input_shape, num_classes, platform_constraints, *, encoding_placement="offload"
):
    """Convert a native model to an IR graph the same way the pipeline does."""
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.platform.platform_constraints import (
        resolve_platform_mapping_params,
    )

    flow = convert_torch_model(
        model, tuple(input_shape), int(num_classes),
        encoding_layer_placement=encoding_placement,
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
    """Honest on-chip-fraction verdict: stock ResNet-18 maps as VALID_FLAGGED."""
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
    assert verdict.research_gap_ops == []
    assert verdict.tier != TIER_INVALID
    assert verdict.is_valid
    assert verdict.tier == TIER_VALID_FLAGGED
    assert verdict.is_flagged
    assert verdict.param_frac == pytest.approx(0.423193, abs=1e-4)
    assert verdict.mac_frac == pytest.approx(0.998918, abs=1e-4)
    assert verdict.mac_frac >= 0.5
    assert verdict.param_frac < 0.5


def _classify(loader, *, placement):
    from mimarsinan.mapping.verification.onchip_fraction import classify_validity

    return classify_validity(
        loader(_NUM_CLASSES, pretrained=False),
        _SHAPE,
        _NUM_CLASSES,
        encoding_placement=placement,
    )


def test_offload_does_not_lift_resnet18_to_valid():
    """RESOLVED-STILL-FLAGGED: ``offload`` does NOT lift ResNet-18 to VALID."""
    from mimarsinan.mapping.verification.onchip_fraction import TIER_VALID_FLAGGED

    subsume = _classify(load_pretrained_resnet18, placement="subsume")
    offload = _classify(load_pretrained_resnet18, placement="offload")
    print(
        f"\n[ResNet18 SWEEP] subsume: tier={subsume.tier} "
        f"param_frac={subsume.param_frac:.6f} mac_frac={subsume.mac_frac:.6f} "
        f"fixable={subsume.placement_fixable_ops}"
        f"\n[ResNet18 SWEEP] offload: tier={offload.tier} "
        f"param_frac={offload.param_frac:.6f} mac_frac={offload.mac_frac:.6f} "
        f"fixable={offload.placement_fixable_ops}"
    )
    assert subsume.research_gap_ops == []
    assert offload.research_gap_ops == []
    assert subsume.placement_fixable_ops == ["Linear"]
    assert offload.placement_fixable_ops == []
    assert subsume.param_frac == pytest.approx(0.422340, abs=1e-4)
    assert subsume.mac_frac == pytest.approx(0.996931, abs=1e-4)
    assert offload.param_frac == pytest.approx(0.423193, abs=1e-4)
    assert offload.mac_frac == pytest.approx(0.998918, abs=1e-4)
    assert offload.param_frac - subsume.param_frac < 0.01
    assert subsume.param_frac < 0.5
    assert offload.param_frac < 0.5
    assert subsume.tier == TIER_VALID_FLAGGED
    assert offload.tier == TIER_VALID_FLAGGED


def test_resnet50_structural_op_set_is_mappable():
    """ResNet-50 carries the same mappable op set (no grouped/depthwise conv)."""
    forbidden = (nn.MultiheadAttention, nn.LayerNorm, nn.GroupNorm)
    model = load_pretrained_resnet50(_NUM_CLASSES, pretrained=False)
    for module in model.modules():
        assert not isinstance(module, forbidden), type(module).__name__
        if isinstance(module, nn.Conv2d):
            assert module.groups == 1, module
    assert model.fc.out_features == _NUM_CLASSES
    assert model.conv1.in_channels == 3


def test_resnet50_measured_validity_is_valid_under_both_placements():
    """MEASURED: ResNet-50 is VALID (param-MAJORITY on-chip) under subsume AND offload."""
    from mimarsinan.mapping.verification.onchip_fraction import TIER_VALID

    subsume = _classify(load_pretrained_resnet50, placement="subsume")
    offload = _classify(load_pretrained_resnet50, placement="offload")
    print(
        f"\n[ResNet50 SWEEP] subsume: tier={subsume.tier} "
        f"param_frac={subsume.param_frac:.6f} mac_frac={subsume.mac_frac:.6f}"
        f"\n[ResNet50 SWEEP] offload: tier={offload.tier} "
        f"param_frac={offload.param_frac:.6f} mac_frac={offload.mac_frac:.6f}"
    )
    assert subsume.research_gap_ops == []
    assert offload.research_gap_ops == []
    assert subsume.param_frac == pytest.approx(0.665655, abs=1e-4)
    assert subsume.mac_frac == pytest.approx(0.998093, abs=1e-4)
    assert offload.param_frac == pytest.approx(0.666060, abs=1e-4)
    assert offload.mac_frac == pytest.approx(0.998694, abs=1e-4)
    assert subsume.param_frac >= 0.5
    assert offload.param_frac >= 0.5
    assert subsume.tier == TIER_VALID
    assert offload.tier == TIER_VALID
    assert subsume.is_valid and not subsume.is_flagged
    assert offload.is_valid and not offload.is_flagged


def test_resnet50_measured_capacity_estimate():
    """MEASURED capacity: ResNet-50 exceeds the 1000-core SUM budget; SCHEDULED fits."""
    from mimarsinan.config_schema.defaults import get_default_platform_constraints
    from mimarsinan.mapping.verification.capacity import estimate_cores_needed

    platform_constraints = get_default_platform_constraints()
    for placement, sum_cores, sched_phases in (
        ("subsume", 1460, 16),
        ("offload", 1607, 17),
    ):
        ir_graph = _build_ir_graph(
            load_pretrained_resnet50(_NUM_CLASSES, pretrained=False),
            _SHAPE,
            _NUM_CLASSES,
            platform_constraints,
            encoding_placement=placement,
        )
        summed = estimate_cores_needed(ir_graph, platform_constraints)
        scheduled = estimate_cores_needed(
            ir_graph, platform_constraints, allow_scheduling=True
        )
        print(
            f"\n[ResNet50 {placement} MEASURED] sum.cores_needed={summed.cores_needed} "
            f"sum.feasible={summed.feasible} | sched.feasible={scheduled.feasible} "
            f"sched.phases={scheduled.phase_count} sched.peak={scheduled.peak_phase_cores}"
        )
        assert summed.cores_available == 1000
        assert summed.cores_needed == sum_cores
        assert not summed.feasible
        assert not summed.scheduled
        assert scheduled.scheduled
        assert scheduled.feasible
        assert scheduled.phase_count == sched_phases
        assert scheduled.peak_phase_cores == 208
        assert scheduled.peak_phase_cores < scheduled.cores_available


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
    assert estimate.cores_needed > 0
    assert estimate.cores_available > 0
    assert estimate.phase_count >= 1
    assert estimate.feasible
    assert estimate.cores_available == 1000
    assert estimate.cores_needed == 651
    assert not estimate.scheduled
    assert estimate.phase_count == 1
    assert estimate.peak_phase_cores == estimate.cores_needed
    assert estimate.cores_needed < estimate.cores_available


def test_loads_real_imagenet_weights():
    """The bridge imports GENUINE ImageNet1K pretrained weights (network-gated skip)."""
    try:
        model = load_pretrained_resnet18(_NUM_CLASSES, pretrained=True)
    except Exception as exc:  # noqa: BLE001 - any download/transport failure is a skip
        pytest.skip(f"pretrained weights unavailable in this environment: {exc}")

    stem_w = model.conv1.weight.detach()
    assert stem_w.shape == (64, 3, 7, 7)
    assert torch.isfinite(stem_w).all()
    assert stem_w.std().item() > 1e-3
    with torch.no_grad():
        y = model(torch.zeros(1, *_SHAPE))
    assert y.shape == (1, _NUM_CLASSES)


def test_loads_real_imagenet_weights_resnet50():
    """The ResNet-50 bridge imports GENUINE ImageNet1K weights (network-gated skip)."""
    try:
        model = load_pretrained_resnet50(_NUM_CLASSES, pretrained=True)
    except Exception as exc:  # noqa: BLE001 - any download/transport failure is a skip
        pytest.skip(f"pretrained weights unavailable in this environment: {exc}")

    stem_w = model.conv1.weight.detach()
    assert stem_w.shape == (64, 3, 7, 7)
    assert torch.isfinite(stem_w).all()
    assert stem_w.std().item() > 1e-3
    with torch.no_grad():
        y = model(torch.zeros(1, *_SHAPE))
    assert y.shape == (1, _NUM_CLASSES)


def test_resnet50_rejects_nonpositive_num_classes():
    """num_classes must be a positive class count for the ResNet-50 bridge too."""
    with pytest.raises(ValueError):
        load_pretrained_resnet50(0, pretrained=False)


class _TinyConvNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(4, 8)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.pool(self.relu1(self.bn1(self.conv1(x))))
        x = self.flatten(self.gap(x))
        return self.fc(self.relu2(self.hidden(x)))


_DEPLOY_SHAPE = (3, 8, 8)
_DEPLOY_T = 4
_DEPLOY_CLASSES = 4
_DEPLOY_N = 6


def _deploy_eval_batch(seed: int = 0):
    torch.manual_seed(seed)
    x = torch.rand(_DEPLOY_N, *_DEPLOY_SHAPE)
    y = torch.randint(0, _DEPLOY_CLASSES, (_DEPLOY_N,))
    return x, y


def test_deploy_and_eval_returns_deployed_accuracy_from_real_sim():
    """A small model DEPLOYS end-to-end and returns a deployed accuracy number off the real sim."""
    x, y = _deploy_eval_batch()
    result = deploy_and_eval(
        _TinyConvNet(_DEPLOY_CLASSES).eval(),
        _DEPLOY_SHAPE,
        _DEPLOY_CLASSES,
        x,
        y,
        simulation_length=_DEPLOY_T,
    )
    print(
        f"\n[TinyConvNet DEPLOYED] acc={result.accuracy:.4f} "
        f"n={result.num_samples} T={result.simulation_length} "
        f"segments={result.neural_segments} hard_cores={result.hard_cores}"
    )
    assert isinstance(result, DeployedEval)
    assert 0.0 <= result.accuracy <= 1.0
    assert result.num_samples == _DEPLOY_N
    assert result.num_classes == _DEPLOY_CLASSES
    assert result.simulation_length == _DEPLOY_T
    assert result.spiking_mode == "lif"
    assert result.neural_segments >= 1
    assert result.hard_cores >= 1
    assert result.logits.shape == (_DEPLOY_N, _DEPLOY_CLASSES)
    assert torch.isfinite(result.logits).all()
    predicted = result.logits.argmax(dim=1)
    expected_acc = float((predicted == y).double().mean())
    assert result.accuracy == pytest.approx(expected_acc, abs=1e-12)


def test_deploy_and_eval_bridge_resnet18_deploys_on_small_input():
    """A REAL bridge ResNet-18 deploys end-to-end through the SNN pipeline (subset)."""
    _, y = _deploy_eval_batch(seed=1)
    torch.manual_seed(1)
    x = torch.rand(_DEPLOY_N, 3, 16, 16)
    result = deploy_and_eval(
        load_pretrained_resnet18(_DEPLOY_CLASSES, pretrained=False),
        (3, 16, 16),
        _DEPLOY_CLASSES,
        x,
        y,
        simulation_length=_DEPLOY_T,
    )
    print(
        f"\n[ResNet18 DEPLOYED] acc={result.accuracy:.4f} "
        f"n={result.num_samples} T={result.simulation_length} "
        f"segments={result.neural_segments} hard_cores={result.hard_cores}"
    )
    assert 0.0 <= result.accuracy <= 1.0
    assert result.logits.shape == (_DEPLOY_N, _DEPLOY_CLASSES)
    assert torch.isfinite(result.logits).all()
    assert result.neural_segments >= 2
    assert result.hard_cores >= result.neural_segments


def test_deploy_and_eval_rejects_mismatched_eval_shapes():
    """Inconsistent eval batch shapes are an honest, precise ValueError (not a crash)."""
    x, y = _deploy_eval_batch()
    with pytest.raises(ValueError):
        deploy_and_eval(
            _TinyConvNet(_DEPLOY_CLASSES), _DEPLOY_SHAPE, _DEPLOY_CLASSES,
            x, y[:-1], simulation_length=_DEPLOY_T,
        )
    with pytest.raises(ValueError):
        deploy_and_eval(
            _TinyConvNet(_DEPLOY_CLASSES), _DEPLOY_SHAPE, _DEPLOY_CLASSES,
            torch.rand(_DEPLOY_N, 3, 9, 9), y, simulation_length=_DEPLOY_T,
        )


def test_deploy_and_eval_rejects_unsupported_spiking_mode():
    """Only the lossless-capable LIF deploy is wired; TTFS is a precise NotYet ValueError."""
    x, y = _deploy_eval_batch()
    with pytest.raises(ValueError):
        deploy_and_eval(
            _TinyConvNet(_DEPLOY_CLASSES), _DEPLOY_SHAPE, _DEPLOY_CLASSES,
            x, y, simulation_length=_DEPLOY_T, spiking_mode="ttfs_quantized",
        )
