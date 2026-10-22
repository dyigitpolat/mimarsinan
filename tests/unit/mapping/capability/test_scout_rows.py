"""The census on the scout's converting members equals the pinned oracle rows, up to reviewed deltas with causes.

The rows are pinned from ``RA/10_wave1/oracles/req_census.py`` (sha256 296ff421dcc7088b) run on the Wave 0 base
(``base_rows.json``, sha256 18cd59f98582604c); ``reviewed_deltas.json`` lists every field where the toolchain's
own IR census differs, with its cause. A difference outside it fails, and so does a delta that no longer differs.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.capability.census import ladder_census
from mimarsinan.mapping.capability.ladder import LADDER, ON_CHIP_WRITE_BACK
from mimarsinan.mapping.capability.mac_split import ESTIMATOR_BELOW_CORE_FLOOR, mac_statement
from mimarsinan.mapping.capability.member import (
    AGREEMENT,
    REFUSED,
    VERDICT_COLUMN,
    c0_verdict,
    convert_for_census,
    member_census,
)
from mimarsinan.mapping.capability.pricing_inputs import level_quantities
from mimarsinan.mapping.layout.segmentation import HostSegment, NeuralSegment, partition_ir_graph
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.mapping.verification.onchip_majority import count_host_ops
from mimarsinan.torch_mapping.converter import probe_stages

from fixtures.requirement_census.scout_members import MEMBERS, NC

from .ir_fixtures import independent_boundary_elements

FIXTURES = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "requirement_census"
BITS = 8

#: Converting on the Wave 0 base (stage-2 certified); the grouped members join at CL4 after G1's host placement.
CONVERTING = (
    "resnet18_cifar_w1/4", "mobilenetv2_small_relu6_dense_twin", "efficientnet_mbconv_se_dense_twin",
    "vit_lite_7_4_softmax_ln_gelu", "unet_level_convtranspose",
)
#: Refused on the Wave 0 base; asserted generically (a ToolchainRefusal subclass), re-pinned at CL4 (§13 rule 6).
REFUSED_MEMBERS = ("mobilenetv2_small_relu6", "efficientnet_mbconv_se", "vit_lite_7_4_bn_relu_reluattn")
#: Rows whose IR holds a multi-input ComputeOp: both MAC counters printed, equality asserted elsewhere (CL4: all).
MULTI_INPUT_ROWS = frozenset({
    "resnet18_cifar_w1/4", "mobilenetv2_small_relu6_dense_twin", "efficientnet_mbconv_se_dense_twin",
    "vit_lite_7_4_softmax_ln_gelu",
})


def _factory(key):
    return next((factory, shape) for name, factory, shape in MEMBERS if name == key)


def _model(key):
    factory, shape = _factory(key)
    torch.manual_seed(0)
    return factory().eval(), shape


@lru_cache(maxsize=None)
def _converted(key):
    model, shape = _model(key)
    return convert_for_census(model, shape, NC, packaging=MVM_PACKAGING)


def _comparable(row, total_macs):
    onchip = 1.0 - row.host_macs / max(1, total_macs)
    return {
        "host_ops": row.host_ops, "host_kinds": row.host_kinds, "neural_segments": row.neural_segments,
        "host_segments": row.host_segments, "boundary_bytes_8bit": row.boundary_bytes,
        "dynamic_payload_bytes_8bit": row.dynamic_payload_bytes, "onchip_mac_fraction": round(onchip, 4),
        "macs_total": total_macs,
    }


def _pinned(key):
    rows = json.loads((FIXTURES / "base_rows.json").read_text())["members"][key]["rows"]
    ordered = [rows[level.title] for level in LADDER]
    return [{**{k: r[k] for k in ("host_ops", "host_kinds", "neural_segments", "host_segments",
                                  "boundary_bytes_8bit", "dynamic_payload_bytes_8bit", "onchip_mac_fraction")},
             "macs_total": r["macs_static"] + r["macs_dynamic"]} for r in ordered]


def _deltas():
    return json.loads((FIXTURES / "reviewed_deltas.json").read_text())


def test_the_pinned_rows_are_the_oracle_output():
    doc = json.loads((FIXTURES / "base_rows.json").read_text())
    assert doc["script_sha256_16"] == "296ff421dcc7088b"
    assert set(CONVERTING) | set(REFUSED_MEMBERS) <= set(doc["members"])


def test_every_reviewed_delta_names_a_cause():
    doc = _deltas()
    for member, levels in doc["members"].items():
        for level, fields in levels.items():
            for field, delta in fields.items():
                assert delta["cause"] in doc["causes"], (member, level, field)


@pytest.mark.parametrize("key", CONVERTING)
def test_c0_equals_two_independent_counts_on_the_same_ir(key):
    converted = _converted(key)
    c0 = ladder_census(converted.ir, value_bits=BITS, levels=LADDER[:1])[0]
    segments = partition_ir_graph(converted.ir)
    assert c0.neural_segments == sum(isinstance(s, NeuralSegment) for s in segments)
    assert c0.host_segments == sum(isinstance(s, HostSegment) for s in segments)
    assert c0.boundary_bytes == independent_boundary_elements(converted.ir)


@pytest.mark.parametrize("key", CONVERTING)
def test_rows_equal_the_pinned_rows_up_to_the_reviewed_deltas(key):
    converted = _converted(key)
    rows = [row for row in ladder_census(converted.ir, value_bits=BITS)
            if row.payload_write in (None, ON_CHIP_WRITE_BACK)]
    deltas = _deltas()["members"].get(key, {})
    mismatches, stale = [], []
    pinned_rows = _pinned(key)
    assert len(rows) == len(pinned_rows) == len(LADDER), "one compared row per level, none dropped"
    for level, row, pinned in zip(LADDER, rows, pinned_rows):
        got = _comparable(row, converted.mac_counters.estimator_total_macs)
        listed = deltas.get(level.name, {})
        for field, want in pinned.items():
            delta = listed.get(field)
            if delta is not None and delta["oracle"] != want:
                stale.append((level.name, field, "pinned value moved"))
            expected = delta["toolchain"] if delta is not None else want
            if got[field] != expected:
                mismatches.append((level.name, field, want, got[field]))
            if delta is not None and got[field] == want:
                stale.append((level.name, field, "no longer differs"))
    assert not mismatches, f"(level, field, pinned, toolchain) outside the reviewed deltas: {mismatches}"
    assert not stale, f"stale reviewed deltas: {stale}"


@pytest.mark.parametrize("key", CONVERTING)
def test_the_mac_counters_agree_on_rows_without_a_multi_input_op(key):
    converted = _converted(key)
    counters = converted.mac_counters
    assert counters.ir_host_macs == count_host_ops(converted.ir)
    assert counters.has_multi_input_op == (key in MULTI_INPUT_ROWS)
    if not counters.has_multi_input_op:
        assert counters.agree and counters.reason is None
    else:
        assert counters.reason == (
            "multi-input op counted by the forward estimator only until G1's input_shapes rule")


@pytest.mark.parametrize("key", ("unet_level_convtranspose",))
def test_a_converting_member_carries_its_stage_two_agreement(key):
    model, shape = _model(key)
    census = member_census(model, shape, NC, packaging=MVM_PACKAGING, value_bits=BITS)
    assert census.verdict.status == AGREEMENT
    assert census.verdict.relative_error is not None and census.verdict.relative_error <= 1e-12
    assert census.verdict.column == VERDICT_COLUMN == "fp64 agreement (max relative error)"
    assert tuple(stage for stage, _ in census.verdict.stages) == probe_stages(MVM_PACKAGING) == (
        "converted model", "identity value program"), "the agreement carries the stage-2 identity value program"
    assert census.verdict.missing_stages == ()
    assert len(census.rows) == len(LADDER) + 4, "C3-C6 carry both payload writes"


@pytest.mark.parametrize("key", REFUSED_MEMBERS)
def test_a_refused_member_carries_the_refusal_type_by_name(key):
    model, shape = _model(key)
    verdict = c0_verdict(model, shape, packaging=MVM_PACKAGING)
    assert verdict.status == REFUSED and verdict.relative_error is None and not verdict.value_mismatch
    assert verdict.refusal_type is not None and issubclass(verdict.refusal_type, ToolchainRefusal)
    assert verdict.refusal == verdict.refusal_type.__name__
    assert verdict.root_refusal_type is not None and issubclass(verdict.root_refusal_type, ToolchainRefusal)
    assert verdict.root_refusal == verdict.root_refusal_type.__name__


@pytest.mark.parametrize("key", CONVERTING)
def test_the_field_map_never_states_on_chip_macs_below_the_cores_mac_floor(key):
    converted = _converted(key)
    for row in ladder_census(converted.ir, value_bits=BITS):
        quantities = level_quantities(row, counters=converted.mac_counters)
        statement = mac_statement(row, converted.mac_counters)
        if "onchip_macs" in quantities:
            assert quantities["onchip_macs"] >= row.core_mac_floor, (row.level, row.payload_write)
        else:
            assert statement.withheld["onchip_macs"], "absent with a named reason"


def test_resnet18_withholds_its_on_chip_macs_with_the_named_reason():
    converted = _converted("resnet18_cifar_w1/4")
    counters = converted.mac_counters
    assert counters.estimator_total_macs == 710_576, "every on-chip convolution counted at one output position"
    for row in ladder_census(converted.ir, value_bits=BITS):
        quantities = level_quantities(row, counters=counters)
        assert "onchip_macs" not in quantities, "absent, never below its own macs"
        assert quantities["macs"] == row.cells_committed == 35_211_876
        assert row.core_mac_floor == 30_446_784, "cells fed by a value: no bias row, no padding axon"
        statement = mac_statement(row, counters)
        assert statement.total_withheld == ESTIMATOR_BELOW_CORE_FLOOR == "estimator_conv_one_position"
        assert statement.withheld["onchip_macs"] == ESTIMATOR_BELOW_CORE_FLOOR
        assert quantities["host_macs"] == statement.host_macs == 0, "both host counters agree on 0"


def test_vit_lite_dependency_segmentation_is_the_oracles_30_where_in_order_emission_gives_37():
    converted = _converted("vit_lite_7_4_softmax_ln_gelu")
    rows = ladder_census(converted.ir, value_bits=BITS, levels=LADDER[:3])
    assert [row.dependency_neural_segments for row in rows] == [30, 30, 30]
    assert [row.neural_segments for row in rows] == [37, 37, 37], "the ir_emission_order reviewed delta"


def test_an_operator_outside_the_table_keeps_the_verdict_and_refuses_the_census_by_name():
    class SoftsignNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.s, self.b = torch.nn.Linear(8, 8), torch.nn.Softsign(), torch.nn.Linear(8, 4)

        def forward(self, x):
            return self.b(self.s(self.a(x)))

    torch.manual_seed(0)
    census = member_census(SoftsignNet().eval(), (8,), 4, packaging=MVM_PACKAGING, value_bits=BITS)
    assert census.verdict.status == AGREEMENT
    assert census.rows == () and census.census_refusal is not None
    assert census.census_refusal.startswith("UnknownOpKindError") and "Softsign" in census.census_refusal
