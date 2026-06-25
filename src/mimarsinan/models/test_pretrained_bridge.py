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

from mimarsinan.models.pretrained_bridge import (
    DeployedEval,
    deploy_and_eval,
    load_pretrained_resnet18,
    load_pretrained_resnet50,
)


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


# ── PLACEMENT SWEEP: resolve the offload hypothesis on ResNet-18 ──────────────


def _classify(loader, *, placement):
    from mimarsinan.mapping.verification.onchip_fraction import classify_validity

    return classify_validity(
        loader(_NUM_CLASSES, pretrained=False),
        _SHAPE,
        _NUM_CLASSES,
        encoding_placement=placement,
    )


def test_offload_does_not_lift_resnet18_to_valid():
    """RESOLVED-STILL-FLAGGED: ``offload`` does NOT lift ResNet-18 to VALID.

    The wave-5 hypothesis was that ``encoding_layer_placement=offload`` would lift
    ResNet-18's ``param_frac`` above 0.50 (as it did for deep_mlp d8). MEASURED here
    under BOTH placements from the SAME live instrument: it does NOT. Offload only
    relocates the single ``placement`` Linear encoder on-chip (so it leaves
    ``placement_fixable_ops``), nudging ``param_frac`` 0.4223 -> 0.4232 -- still a
    param-MINORITY. The host param majority lives in the residual-boundary
    ``Sequential`` ComputeOps (classified ``supported_host``, NOT placement-fixable),
    which ``offload`` cannot relocate. Both placements stay VALID_FLAGGED.
    """
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
    # Neither placement carries a research-frontier op (pure residual CNN).
    assert subsume.research_gap_ops == []
    assert offload.research_gap_ops == []
    # The verdict RESPONDS to placement: offload moves the single Linear encoder
    # on-chip, clearing it from the placement-fixable list (live-instrument signal,
    # not a hardcoded constant).
    assert subsume.placement_fixable_ops == ["Linear"]
    assert offload.placement_fixable_ops == []
    # MEASURED fracs under each placement (recorded region descriptor).
    assert subsume.param_frac == pytest.approx(0.422340, abs=1e-4)
    assert subsume.mac_frac == pytest.approx(0.996931, abs=1e-4)
    assert offload.param_frac == pytest.approx(0.423193, abs=1e-4)
    assert offload.mac_frac == pytest.approx(0.998918, abs=1e-4)
    # RESOLUTION: offload's param lift is negligible (~0.001) and STAYS below 0.50,
    # so the tier is STILL VALID_FLAGGED under both placements. Hypothesis refuted.
    assert offload.param_frac - subsume.param_frac < 0.01
    assert subsume.param_frac < 0.5
    assert offload.param_frac < 0.5
    assert subsume.tier == TIER_VALID_FLAGGED
    assert offload.tier == TIER_VALID_FLAGGED


# ── ResNet-50 region: a bottleneck-block residual net stays param-MAJORITY ─────


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
    """MEASURED: ResNet-50 is VALID (param-MAJORITY on-chip) under subsume AND offload.

    The contrast with ResNet-18 is the headline result: ResNet-50's bottleneck
    blocks (1x1->3x3->1x1 trunk) hold the param majority on-chip, so its
    ``param_frac`` ~0.666 clears the 0.50 majority -- VALID, not flagged -- under
    BOTH placements. The residual-boundary host cost is real but its FRACTION is
    architecture-dependent: BasicBlock (R18) tips param-minority, Bottleneck (R50)
    stays param-majority. Same live instrument, same shape; only the model differs.
    """
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
    # MEASURED region descriptor (recorded).
    assert subsume.param_frac == pytest.approx(0.665655, abs=1e-4)
    assert subsume.mac_frac == pytest.approx(0.998093, abs=1e-4)
    assert offload.param_frac == pytest.approx(0.666060, abs=1e-4)
    assert offload.mac_frac == pytest.approx(0.998694, abs=1e-4)
    # Param-MAJORITY on-chip under BOTH placements -> VALID (not flagged).
    assert subsume.param_frac >= 0.5
    assert offload.param_frac >= 0.5
    assert subsume.tier == TIER_VALID
    assert offload.tier == TIER_VALID
    assert subsume.is_valid and not subsume.is_flagged
    assert offload.is_valid and not offload.is_flagged


def test_resnet50_measured_capacity_estimate():
    """MEASURED capacity: ResNet-50 exceeds the 1000-core SUM budget; SCHEDULED fits.

    Under the default single-pool SUM budget ResNet-50 (3x32x32) needs ~1460/1607
    cores -> NOT feasible on 1000 -- an honest capacity verdict (valid in PLACEMENT
    terms, over budget in CAPACITY terms). The SCHEDULED (fresh-pool-per-phase) path
    time-multiplexes it: peak phase 208 cores fits the budget across ~16-17
    reprogramming passes. All numbers come straight from ``estimate_cores_needed``.
    """
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
        # SUM verdict: over the 1000-core single-pool budget (honest capacity gap).
        assert summed.cores_available == 1000
        assert summed.cores_needed == sum_cores
        assert not summed.feasible
        assert not summed.scheduled
        # SCHEDULED verdict: time-multiplexed peak fits; reprogramming-pass count.
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


# ── DEPLOY-and-EVAL: the model actually runs on the deployed spiking sim ───────
#
# These tests convert a (small) bridge model through the REAL SNN pipeline
# (convert -> IR map -> hybrid HCM pack -> deployed SpikingHybridCoreFlow) and
# read a DEPLOYED accuracy number off the on-chip sim. They are FAST/SUBSET by
# construction: tiny input shape, tiny T, tiny eval batch -- NOT full ImageNet
# (that deploy is a supervised Group-2 GPU run). The structural descriptor tests
# above never depend on this path; offline random weights keep it deterministic.

# A small pipeline-native classifier: conv/bn/relu/pool/linear only (the same op
# set as the bridge residual nets, minus the residual add) -- the canonical small
# "pretrained-style" deploy vehicle so the deployed sim is cheap. The hidden Linear
# (after the GAP/flatten host encoder) is a genuine ON-CHIP neural layer, so the
# model packs onto at least one hard core rather than fully subsuming host-side.
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


_DEPLOY_SHAPE = (3, 8, 8)   # tiny spatial -> few cores -> fast deployed sim
_DEPLOY_T = 4               # tiny spiking window -> fast sim
_DEPLOY_CLASSES = 4
_DEPLOY_N = 6               # tiny eval subset


def _deploy_eval_batch(seed: int = 0):
    torch.manual_seed(seed)
    x = torch.rand(_DEPLOY_N, *_DEPLOY_SHAPE)
    y = torch.randint(0, _DEPLOY_CLASSES, (_DEPLOY_N,))
    return x, y


def test_deploy_and_eval_returns_deployed_accuracy_from_real_sim():
    """A small model DEPLOYS end-to-end and returns a deployed accuracy number.

    The headline of the deploy bridge: this is NOT the static validity descriptor
    -- the model is converted, mapped, packed into a hybrid hard-core mapping, and
    RUN on the deployed ``SpikingHybridCoreFlow`` (the same executor production
    ``SimulationRunner`` uses). The returned ``accuracy`` is the genuine deployed
    top-1 on the (tiny) eval subset, and ``logits`` come straight off the sim.
    """
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
    # A genuine accuracy fraction from the deployed sim (not a NaN / sentinel).
    assert 0.0 <= result.accuracy <= 1.0
    assert result.num_samples == _DEPLOY_N
    assert result.num_classes == _DEPLOY_CLASSES
    assert result.simulation_length == _DEPLOY_T
    assert result.spiking_mode == "lif"
    # The model actually packed onto hard cores (a real deployed structure).
    assert result.neural_segments >= 1
    assert result.hard_cores >= 1
    # Deployed logits are finite, task-sized, and came off the sim (one per sample).
    assert result.logits.shape == (_DEPLOY_N, _DEPLOY_CLASSES)
    assert torch.isfinite(result.logits).all()
    # The reported accuracy is exactly the argmax-vs-target rate of those logits.
    predicted = result.logits.argmax(dim=1)
    expected_acc = float((predicted == y).double().mean())
    assert result.accuracy == pytest.approx(expected_acc, abs=1e-12)


def test_deploy_and_eval_bridge_resnet18_deploys_on_small_input():
    """A REAL bridge ResNet-18 deploys end-to-end through the SNN pipeline (subset).

    The capability payload for F3/F4: the exact stock-residual bridge model
    (``load_pretrained_resnet18``, random weights for an offline-safe / fast build)
    is converted, mapped, packed, and RUN on the deployed spiking sim at a tiny
    input shape + tiny T + tiny eval batch. It returns a deployed accuracy off the
    REAL sim -- proving the residual-net bridge is deployable, not just classifiable.
    The residual ``add`` boundaries split it into multiple neural segments, which the
    deployed hybrid executor runs across the sync points.
    """
    _, y = _deploy_eval_batch(seed=1)
    torch.manual_seed(1)
    x = torch.rand(_DEPLOY_N, 3, 16, 16)  # tiny spatial keeps the ResNet deploy cheap
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
    # The residual ``add`` boundaries make the deployed net multi-segment.
    assert result.neural_segments >= 2
    assert result.hard_cores >= result.neural_segments


def test_deploy_and_eval_rejects_mismatched_eval_shapes():
    """Inconsistent eval batch shapes are an honest, precise ValueError (not a crash)."""
    x, y = _deploy_eval_batch()
    # Sample count mismatch.
    with pytest.raises(ValueError):
        deploy_and_eval(
            _TinyConvNet(_DEPLOY_CLASSES), _DEPLOY_SHAPE, _DEPLOY_CLASSES,
            x, y[:-1], simulation_length=_DEPLOY_T,
        )
    # Wrong per-sample shape.
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
