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
