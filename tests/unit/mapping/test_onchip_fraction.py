"""Static on-chip-fraction resolver: reproduce the authoritative gate WITHOUT a run.

The deployment validity gate (``onchip_majority.py``) needs a fully MAPPED
``IRGraph`` and a pipeline run. This static resolver classifies host vs. on-chip
parameters (and MACs) from the MODEL SPEC ALONE — FX-tracing/segmenting the model
the same way the converter would — so the campaign scheduler can reject a
host-majority job BEFORE it claims a GPU. Every PARAMS estimate here is asserted
to reproduce the authoritative ``count_host_params`` decomposition exactly.
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.verification.onchip_fraction import (
    OnchipFractionEstimate,
    ValidityVerdict,
    assert_onchip_majority_estimate_or_raise,
    classify_validity,
    estimate_onchip_fraction,
)
from mimarsinan.mapping.verification.onchip_majority import (
    OnchipMajorityError,
    count_host_params,
)


def _build(model_type, model_config, input_shape, num_classes):
    from mimarsinan.models.builders import BUILDERS_REGISTRY

    builder = BUILDERS_REGISTRY[model_type]("cpu", input_shape, num_classes, {})
    return builder.build(model_config)


def _authoritative_breakdown(model, input_shape, num_classes, placement):
    """The REAL gate: convert + map to an IRGraph, then count_host_params."""
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
    from mimarsinan.torch_mapping.converter import convert_torch_model

    flow = convert_torch_model(
        model, input_shape, num_classes, encoding_layer_placement=placement
    )
    total = int(sum(p.numel() for p in flow.parameters()))
    mapper_repr = flow.get_mapper_repr()
    if hasattr(mapper_repr, "assign_perceptron_indices"):
        mapper_repr.assign_perceptron_indices()
    compute_per_source_scales(mapper_repr)
    irm = IRMapping(
        q_max=127,
        firing_mode="Default",
        max_axons=None,
        max_neurons=None,
        allow_coalescing=False,
        hardware_bias=True,
    )
    ir_graph = irm.map(mapper_repr)
    host = int(count_host_params(ir_graph))
    return host, total


# (model_type, model_config, input_shape, num_classes, placement, expected_fraction, tol)
_PARAM_CASES = [
    ("deep_mlp", {"depth": 8, "width": 64}, (1, 28, 28), 10, "subsume", 0.36, 0.03),
    ("deep_mlp", {"depth": 4, "width": 64}, (1, 28, 28), 10, "subsume", 0.20, 0.03),
    ("deep_mlp", {"depth": 8, "width": 64}, (1, 28, 28), 10, "offload", 0.99, 0.01),
    ("deep_cnn", {"depth": 8, "width": 16}, (1, 28, 28), 10, "subsume", 0.985, 0.02),
    ("lenet5", {"variant": "lenet5"}, (1, 28, 28), 10, "subsume", 0.99, 0.01),
    (
        "mlp_mixer_core",
        {
            "base_activation": "LeakyReLU",
            "patch_n_1": 4,
            "patch_m_1": 4,
            "patch_c_1": 32,
            "fc_w_1": 64,
            "fc_w_2": 64,
            "num_blocks": 2,
        },
        (1, 28, 28),
        10,
        "subsume",
        0.90,
        0.05,
    ),
]


@pytest.mark.parametrize(
    "model_type,model_config,input_shape,num_classes,placement,expected,tol",
    _PARAM_CASES,
)
def test_param_estimate_matches_model_spec(
    model_type, model_config, input_shape, num_classes, placement, expected, tol
):
    model = _build(model_type, model_config, input_shape, num_classes)
    est = estimate_onchip_fraction(
        model,
        input_shape,
        num_classes,
        encoding_placement=placement,
        metric="params",
    )
    assert isinstance(est, OnchipFractionEstimate)
    assert est.metric == "params"
    assert est.onchip + est.host == est.total
    assert 0.0 <= est.fraction <= 1.0
    assert est.fraction == pytest.approx(expected, abs=tol)


@pytest.mark.parametrize(
    "model_type,model_config,input_shape,num_classes,placement,expected,tol",
    _PARAM_CASES,
)
def test_param_estimate_reproduces_authoritative_gate(
    model_type, model_config, input_shape, num_classes, placement, expected, tol
):
    """The static host/on-chip params must equal the REAL IRGraph decomposition."""
    model = _build(model_type, model_config, input_shape, num_classes)
    host, total = _authoritative_breakdown(model, input_shape, num_classes, placement)

    # rebuild fresh — conversion mutates the model in place
    model = _build(model_type, model_config, input_shape, num_classes)
    est = estimate_onchip_fraction(
        model, input_shape, num_classes, encoding_placement=placement, metric="params"
    )
    assert est.host == host
    assert est.total == total
    assert est.onchip == total - host


def test_offload_strictly_exceeds_subsume_for_deep_mlp():
    input_shape, num_classes = (1, 28, 28), 10
    cfg = {"depth": 8, "width": 64}
    sub = estimate_onchip_fraction(
        _build("deep_mlp", cfg, input_shape, num_classes),
        input_shape,
        num_classes,
        encoding_placement="subsume",
    )
    off = estimate_onchip_fraction(
        _build("deep_mlp", cfg, input_shape, num_classes),
        input_shape,
        num_classes,
        encoding_placement="offload",
    )
    assert off.fraction > sub.fraction


def test_macs_metric_returns_sane_fraction():
    input_shape, num_classes = (1, 28, 28), 10
    est = estimate_onchip_fraction(
        _build("deep_mlp", {"depth": 8, "width": 64}, input_shape, num_classes),
        input_shape,
        num_classes,
        metric="macs",
    )
    assert est.metric == "macs"
    assert est.total > 0
    assert est.onchip + est.host == est.total
    assert 0.0 <= est.fraction <= 1.0


def test_macs_and_params_differ_for_conv_model():
    """A CNN spends MACs on conv but holds few params there: MAC frac != param frac."""
    input_shape, num_classes = (1, 28, 28), 10
    cfg = {"depth": 8, "width": 16}
    p = estimate_onchip_fraction(
        _build("deep_cnn", cfg, input_shape, num_classes),
        input_shape,
        num_classes,
        metric="params",
    )
    m = estimate_onchip_fraction(
        _build("deep_cnn", cfg, input_shape, num_classes),
        input_shape,
        num_classes,
        metric="macs",
    )
    assert 0.0 <= m.fraction <= 1.0
    assert m.total != p.total  # MAC budget and param budget are different totals


def test_unknown_metric_raises():
    input_shape, num_classes = (1, 28, 28), 10
    with pytest.raises(ValueError):
        estimate_onchip_fraction(
            _build("deep_mlp", {"depth": 4, "width": 64}, input_shape, num_classes),
            input_shape,
            num_classes,
            metric="flops_or_something",
        )


def test_invalid_placement_raises():
    input_shape, num_classes = (1, 28, 28), 10
    with pytest.raises(ValueError):
        estimate_onchip_fraction(
            _build("deep_mlp", {"depth": 4, "width": 64}, input_shape, num_classes),
            input_shape,
            num_classes,
            encoding_placement="nowhere",
        )


class TestAssertHelper:
    def test_host_majority_model_raises(self):
        # deep_mlp d4 subsume is host-majority (~0.20 on chip) -> must RAISE.
        input_shape, num_classes = (1, 28, 28), 10
        model = _build("deep_mlp", {"depth": 4, "width": 64}, input_shape, num_classes)
        with pytest.raises(OnchipMajorityError) as exc:
            assert_onchip_majority_estimate_or_raise(
                model, input_shape, num_classes, encoding_placement="subsume"
            )
        assert "on-chip" in str(exc.value).lower()

    def test_onchip_majority_model_passes(self):
        # deep_mlp d8 offload is on-chip-majority -> returns the estimate.
        input_shape, num_classes = (1, 28, 28), 10
        model = _build("deep_mlp", {"depth": 8, "width": 64}, input_shape, num_classes)
        est = assert_onchip_majority_estimate_or_raise(
            model, input_shape, num_classes, encoding_placement="offload"
        )
        assert est.fraction >= 0.5

    def test_custom_min_fraction_floor(self):
        # deep_mlp d8 subsume ~0.36; a 0.30 floor passes, a 0.50 floor rejects.
        input_shape, num_classes = (1, 28, 28), 10
        model = _build("deep_mlp", {"depth": 8, "width": 64}, input_shape, num_classes)
        est = assert_onchip_majority_estimate_or_raise(
            model, input_shape, num_classes, min_fraction=0.30
        )
        assert est.fraction >= 0.30
        model = _build("deep_mlp", {"depth": 8, "width": 64}, input_shape, num_classes)
        with pytest.raises(OnchipMajorityError):
            assert_onchip_majority_estimate_or_raise(
                model, input_shape, num_classes, min_fraction=0.50
            )


class _HostMajorityModel(nn.Module):
    """A model whose parameters live almost entirely in a host-side classifier head.

    A tiny on-chip-mappable Linear stem feeds a huge readout Linear (the segment
    output → host ComputeOp), so the on-chip param/MAC fraction sits well below the
    20% floor: a synthetic INVALID case for the tiered gate.
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(28 * 28, 8)
        self.act = nn.ReLU()
        self.head = nn.Linear(8, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.head(self.act(self.stem(x)))


class TestClassifyValidity:
    """Tiered validity (floor=0.20, majority=0.50) on BOTH params and MACs."""

    def _verdict(self, model_type, model_config, input_shape, num_classes, placement):
        model = _build(model_type, model_config, input_shape, num_classes)
        return classify_validity(
            model, input_shape, num_classes, encoding_placement=placement
        )

    def test_deep_cnn_d8_is_valid(self):
        v = self._verdict("deep_cnn", {"depth": 8, "width": 16}, (1, 28, 28), 10, "subsume")
        assert isinstance(v, ValidityVerdict)
        assert v.tier == "VALID"
        assert min(v.param_frac, v.mac_frac) >= 0.50
        assert v.research_gap_ops == []

    def test_lenet5_is_valid(self):
        v = self._verdict("lenet5", {"variant": "lenet5"}, (1, 28, 28), 10, "subsume")
        assert v.tier == "VALID"
        assert v.research_gap_ops == []

    def test_mlp_mixer_core_is_valid(self):
        cfg = {
            "base_activation": "LeakyReLU",
            "patch_n_1": 4,
            "patch_m_1": 4,
            "patch_c_1": 32,
            "fc_w_1": 64,
            "fc_w_2": 64,
            "num_blocks": 2,
        }
        v = self._verdict("mlp_mixer_core", cfg, (1, 28, 28), 10, "subsume")
        assert v.tier == "VALID"
        assert v.research_gap_ops == []

    def test_deep_mlp_w64_subsume_is_flagged_placement_only(self):
        v = self._verdict("deep_mlp", {"depth": 8, "width": 64}, (1, 28, 28), 10, "subsume")
        assert v.tier == "VALID_FLAGGED"
        assert min(v.param_frac, v.mac_frac) >= 0.20
        assert not (v.param_frac >= 0.50 and v.mac_frac >= 0.50)
        # the flag is PLACEMENT (the host encoder Linear), NOT a research gap
        assert v.placement_fixable_ops  # non-empty: the host encoder Linear
        assert "Linear" in v.placement_fixable_ops
        assert v.research_gap_ops == []

    def test_deep_mlp_w64_offload_is_valid(self):
        v = self._verdict("deep_mlp", {"depth": 8, "width": 64}, (1, 28, 28), 10, "offload")
        assert v.tier == "VALID"
        # offloading the encoder removes the placement flag entirely
        assert v.placement_fixable_ops == []

    def test_torch_vit_is_flagged_with_attention_and_layernorm_research_gap(self):
        v = self._verdict("torch_vit", {}, (3, 32, 32), 10, "subsume")
        assert v.tier == "VALID_FLAGGED"
        assert v.tier != "INVALID"
        # the on-chip-attention / LayerNorm frontier is the research gap
        assert "MultiheadAttention" in v.research_gap_ops
        assert "LayerNorm" in v.research_gap_ops

    def test_torch_vgg16_is_valid(self):
        v = self._verdict("torch_vgg16", {}, (3, 32, 32), 10, "subsume")
        assert v.tier == "VALID"
        assert min(v.param_frac, v.mac_frac) >= 0.50
        assert v.research_gap_ops == []

    def test_synthetic_host_majority_is_invalid(self):
        model = _HostMajorityModel()
        v = classify_validity(model, (1, 28, 28), 10, encoding_placement="subsume")
        assert v.tier == "INVALID"
        assert min(v.param_frac, v.mac_frac) < 0.20

    def test_tier_boundaries_are_inclusive_on_both_metrics(self):
        # both metrics exactly at majority -> VALID; one just below -> FLAGGED;
        # one below floor -> INVALID. Drives the min()/>= semantics directly.
        from mimarsinan.mapping.verification.onchip_fraction import _tier_for

        assert _tier_for(0.50, 0.50, 0.20, 0.50) == "VALID"
        assert _tier_for(0.50, 0.49, 0.20, 0.50) == "VALID_FLAGGED"
        assert _tier_for(0.20, 0.90, 0.20, 0.50) == "VALID_FLAGGED"
        assert _tier_for(0.19, 0.90, 0.20, 0.50) == "INVALID"
        assert _tier_for(0.90, 0.19, 0.20, 0.50) == "INVALID"

    def test_classify_validity_rejects_unknown_placement(self):
        model = _build("deep_mlp", {"depth": 4, "width": 64}, (1, 28, 28), 10)
        with pytest.raises(ValueError):
            classify_validity(model, (1, 28, 28), 10, encoding_placement="nowhere")


class TestLazyModelGuard:
    """A model reaching the static gate with unmaterialized Lazy modules must
    fail with a named, actionable error — not torch's bare numel ValueError
    (the t0_20 crash: a device-mismatched warmup silently skipped
    materialization)."""

    def test_unmaterialized_lazy_model_raises_named_error(self):
        model = _build(
            "simple_mlp", {"mlp_width_1": 256, "mlp_width_2": 128}, (1, 28, 28), 10,
        )
        with pytest.raises(ValueError, match="warmup forward"):
            assert_onchip_majority_estimate_or_raise(
                model, (1, 28, 28), 10, encoding_placement="subsume",
            )


def _tier0_model_specs():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "test_configs" / "tier0"
    specs = []
    for path in sorted(root.glob("t0_*.json")):
        cfg = json.loads(path.read_text())
        dp = cfg["deployment_parameters"]
        specs.append(pytest.param(
            dp["model_type"],
            dp["model_config"],
            dp.get("encoding_layer_placement", "subsume"),
            id=path.stem,
        ))
    return specs


class TestTier0MatrixClearsTheStaticFloor:
    """The Model Building fail-fast twin must not fire on ANY current tier-0
    cell (the mandate for landing it in the pipeline; W2 Q3)."""

    @pytest.mark.parametrize(
        "model_type,model_config,placement", _tier0_model_specs()
    )
    def test_cell_clears_the_20pct_floor(self, model_type, model_config, placement):
        input_shape, num_classes = (1, 28, 28), 10
        model = _build(model_type, model_config, input_shape, num_classes)
        # ModelBuildingStep warms up before the gate (materializes Lazy modules).
        with torch.no_grad():
            model(torch.randn(2, *input_shape))
        est = assert_onchip_majority_estimate_or_raise(
            model, input_shape, num_classes,
            encoding_placement=placement, min_fraction=0.2,
        )
        assert est.fraction >= 0.2


def _map_to_ir_graph(model, input_shape, num_classes, placement):
    """Convert + map a model to a fully-mapped IRGraph (the authoritative gate input)."""
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
    from mimarsinan.torch_mapping.converter import convert_torch_model

    flow = convert_torch_model(
        model, input_shape, num_classes, encoding_layer_placement=placement
    )
    mapper_repr = flow.get_mapper_repr()
    if hasattr(mapper_repr, "assign_perceptron_indices"):
        mapper_repr.assign_perceptron_indices()
    compute_per_source_scales(mapper_repr)
    irm = IRMapping(
        q_max=127, firing_mode="Default", max_axons=None, max_neurons=None,
        allow_coalescing=False, hardware_bias=True,
    )
    return flow, irm.map(mapper_repr)


# Feed-forward corpus (no attention/sequence gather): the IRGraph host-op count
# reproduces the forward-based estimator's MAC decomposition exactly.
_OPS_REPRO_CASES = [
    ("deep_mlp", {"depth": 8, "width": 64}, (1, 28, 28), 10, "subsume"),
    ("deep_mlp", {"depth": 8, "width": 64}, (1, 28, 28), 10, "offload"),
    ("deep_mlp", {"depth": 4, "width": 64}, (1, 28, 28), 10, "subsume"),
    ("deep_cnn", {"depth": 8, "width": 16}, (1, 28, 28), 10, "subsume"),
    ("lenet5", {"variant": "lenet5"}, (1, 28, 28), 10, "subsume"),
    ("lenet5", {"variant": "lenet5"}, (1, 28, 28), 10, "offload"),
    (
        "mlp_mixer_core",
        {
            "base_activation": "LeakyReLU",
            "patch_n_1": 4, "patch_m_1": 4, "patch_c_1": 32,
            "fc_w_1": 64, "fc_w_2": 64, "num_blocks": 2,
        },
        (1, 28, 28), 10, "subsume",
    ),
]


class TestCountHostOpsReproducesEstimator:
    """The mandate: IRGraph host-op MACs reproduce the static estimator's MAC
    decomposition, exactly as count_host_params reproduces its param decomposition."""

    @pytest.mark.parametrize(
        "model_type,model_config,input_shape,num_classes,placement", _OPS_REPRO_CASES,
    )
    def test_host_ops_match_the_forward_estimator(
        self, model_type, model_config, input_shape, num_classes, placement
    ):
        from mimarsinan.mapping.verification.onchip_majority import (
            compute_onchip_ops_fraction,
            count_host_ops,
        )

        _flow, ir_graph = _map_to_ir_graph(
            _build(model_type, model_config, input_shape, num_classes),
            input_shape, num_classes, placement,
        )
        # rebuild fresh — conversion mutates the model in place
        est = estimate_onchip_fraction(
            _build(model_type, model_config, input_shape, num_classes),
            input_shape, num_classes, encoding_placement=placement, metric="macs",
        )
        assert count_host_ops(ir_graph) == est.host
        ops = compute_onchip_ops_fraction(ir_graph, total_ops=est.total)
        assert ops.host_ops == est.host
        assert ops.onchip_ops == est.total - est.host
        assert ops.fraction == pytest.approx(est.fraction)


class _MappableHostMajorityModel(nn.Module):
    """A tiny on-chip middle feeding a huge host classifier readout: an on-chip
    MINORITY that still maps (>=1 neural core), for the runtime INVALID case."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 16)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(16, 16)
        self.a2 = nn.ReLU()
        self.head = nn.Linear(16, 4000)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.a1(self.l1(x))
        x = self.a2(self.l2(x))
        return self.head(x)


class TestAssertOnchipValidity:
    """The authoritative tiered gate over BOTH metrics on a mapped IR graph."""

    def _report(self, model_type, model_config, num_classes, placement):
        from mimarsinan.mapping.verification.onchip_fraction import (
            assert_onchip_validity_or_raise,
        )

        input_shape = (1, 28, 28)
        flow, ir_graph = _map_to_ir_graph(
            _build(model_type, model_config, input_shape, num_classes),
            input_shape, num_classes, placement,
        )
        return assert_onchip_validity_or_raise(
            ir_graph, flow, input_shape, num_classes, encoding_placement=placement
        )

    def test_onchip_majority_model_is_valid(self):
        r = self._report("deep_cnn", {"depth": 8, "width": 16}, 10, "subsume")
        assert r.tier == "VALID"
        assert min(r.param_frac, r.mac_frac) >= 0.50

    def test_flagged_model_deploys_without_raising(self):
        r = self._report("deep_mlp", {"depth": 8, "width": 64}, 10, "subsume")
        assert r.tier == "VALID_FLAGGED"
        assert min(r.param_frac, r.mac_frac) >= 0.20
        assert not (r.param_frac >= 0.50 and r.mac_frac >= 0.50)

    def test_host_majority_mapping_raises_naming_both_metrics(self):
        from mimarsinan.mapping.verification.onchip_fraction import (
            assert_onchip_validity_or_raise,
        )

        input_shape, num_classes = (1, 28, 28), 4000
        flow, ir_graph = _map_to_ir_graph(
            _MappableHostMajorityModel(), input_shape, num_classes, "subsume"
        )
        with pytest.raises(OnchipMajorityError) as exc:
            assert_onchip_validity_or_raise(
                ir_graph, flow, input_shape, num_classes, encoding_placement="subsume"
            )
        msg = str(exc.value)
        assert "ops" in msg and "params" in msg
        assert "floor" in msg
        assert "%" in msg  # the actual fractions are named

    def test_zero_floor_is_the_documented_offload_escape(self):
        from mimarsinan.mapping.verification.onchip_fraction import (
            assert_onchip_validity_or_raise,
        )

        input_shape, num_classes = (1, 28, 28), 4000
        flow, ir_graph = _map_to_ir_graph(
            _MappableHostMajorityModel(), input_shape, num_classes, "subsume"
        )
        # thresholds=0 disables enforcement: an intentionally host-offloaded run
        # deploys instead of raising.
        r = assert_onchip_validity_or_raise(
            ir_graph, flow, input_shape, num_classes,
            encoding_placement="subsume", floor=0.0, majority=0.0,
        )
        assert r.tier == "VALID"

    def test_default_thresholds_are_the_framework_ssot(self):
        from mimarsinan.mapping.verification.onchip_majority import (
            DEFAULT_ONCHIP_FLOOR,
            DEFAULT_ONCHIP_MAJORITY,
        )

        assert DEFAULT_ONCHIP_FLOOR == 0.20
        assert DEFAULT_ONCHIP_MAJORITY == 0.50
