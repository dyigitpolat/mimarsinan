"""The introspection channel: typed, versioned payloads over the record.

Two views at two completenesses (schema doc §3): a search candidate answers what
its shape-only layout holds, a sealed record answers what it measured. A payload
is available exactly when its backing datum is populated — never a silently
empty answer that reads like a real one.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from mimarsinan.deployment_record.introspection import (
    INTROSPECTION_REGISTRY,
    BankCompositionPayload,
    CandidateLayoutView,
    CapabilitiesPayload,
    IntrospectionRegistry,
    IntrospectionSpec,
    LayerRollupPayload,
    LayoutStatsPayload,
    PlacementPayload,
    RecordIntrospectionView,
    SchedulePayload,
    SoftcoresPayload,
    build_registry,
)
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities

from .record_fixtures import make_full_record

PLATFORM = {
    "cores": [{"max_axons": 64, "max_neurons": 64, "count": 8}],
    "weight_bits": 8,
}


def _conv_then_fc():
    """Three bank-backed conv positions (one bank, one layer) plus an owned fc core."""
    conv = [
        LayoutSoftCoreSpec(
            input_count=9, output_count=4, residency_class_id=0, latency_tag=0,
            segment_id=0, bank_id=0, perceptron_index=0, name=f"conv1_pos{i}_g0",
        )
        for i in range(3)
    ]
    fc = LayoutSoftCoreSpec(
        input_count=36, output_count=10, residency_class_id=1, latency_tag=1,
        segment_id=0, bank_id=None, perceptron_index=1, name="fc1_tile_0_10",
    )
    return conv + [fc]


def _candidate(**overrides):
    return CandidateLayoutView.from_platform(
        _conv_then_fc(), {**PLATFORM, **overrides},
    )


def _record_view():
    return RecordIntrospectionView(make_full_record())


class TestCatalogueShape:
    def test_the_seven_payloads_are_registered_once(self):
        assert INTROSPECTION_REGISTRY.names() == (
            "softcores", "layer_rollup", "bank_composition", "placement",
            "schedule", "capabilities", "layout_stats",
        )

    def test_a_duplicate_registration_raises(self):
        registry = build_registry()
        spec = registry.get("softcores")
        with pytest.raises(ValueError, match="already registered"):
            registry.register(spec)

    def test_a_spec_must_declare_a_view_it_can_answer(self):
        with pytest.raises(ValueError, match="no view"):
            IntrospectionSpec(
                name="x", payload_type=SoftcoresPayload, requires="r", doc="d",
                builders={},
            )

    def test_the_catalogue_declares_version_and_views(self):
        rows = {row["name"]: row for row in INTROSPECTION_REGISTRY.catalog()}
        assert rows["placement"]["views"] == ["deployment_record"]
        assert rows["softcores"]["views"] == ["candidate_layout"]
        assert rows["schedule"]["views"] == ["candidate_layout", "deployment_record"]
        assert all(row["version"] >= 1 for row in rows.values())

    def test_an_unknown_name_fails_loud(self):
        with pytest.raises(ValueError, match="unknown introspection payload"):
            INTROSPECTION_REGISTRY.get("nope")


class TestTwoCompletenesses:
    def test_a_candidate_answers_what_a_shape_only_layout_holds(self):
        available = INTROSPECTION_REGISTRY.available_for(_candidate())
        assert set(available) == {
            "softcores", "layer_rollup", "bank_composition", "schedule",
            "capabilities", "layout_stats",
        }
        assert "placement" not in available

    def test_a_sealed_record_answers_the_deployed_questions(self):
        available = INTROSPECTION_REGISTRY.available_for(_record_view())
        assert set(available) == {
            "layer_rollup", "bank_composition", "placement", "schedule",
            "capabilities", "layout_stats",
        }

    def test_serving_an_unavailable_payload_names_what_it_requires(self):
        with pytest.raises(ValueError, match="requires a sealed record"):
            INTROSPECTION_REGISTRY.serve("placement", _candidate())

    def test_an_empty_candidate_serves_no_layout_payloads(self):
        empty = CandidateLayoutView(softcores=(), capabilities=ChipCapabilities())
        assert "softcores" not in INTROSPECTION_REGISTRY.available_for(empty)


class TestSoftcoresPayload:
    def test_both_identities_reach_the_consumer(self):
        payload = INTROSPECTION_REGISTRY.serve("softcores", _candidate())
        assert isinstance(payload, SoftcoresPayload)
        assert payload.count == 4
        banked = [row for row in payload.softcores if row.bank_id is not None]
        assert len(banked) == 3
        assert {row.perceptron_index for row in payload.softcores} == {0, 1}


class TestLayerRollupKeysOnRealIdentity:
    def test_rows_are_keyed_by_perceptron_index(self):
        payload = INTROSPECTION_REGISTRY.serve("layer_rollup", _candidate())
        assert isinstance(payload, LayerRollupPayload)
        assert [row.perceptron_index for row in payload.layers] == [0, 1]
        conv = payload.layers[0]
        assert conv.softcore_count == 3
        assert conv.total_area == 3 * 9 * 4
        assert conv.bank_ids == (0,)

    def test_renaming_a_core_cannot_change_the_rollup(self):
        """The name-split heuristic is gone: identity is a fact, not a convention."""
        renamed = [
            replace(sc, name=f"totally_different_{i}")
            for i, sc in enumerate(_conv_then_fc())
        ]
        original = INTROSPECTION_REGISTRY.serve("layer_rollup", _candidate())
        after = INTROSPECTION_REGISTRY.serve(
            "layer_rollup",
            CandidateLayoutView.from_platform(renamed, PLATFORM),
        )
        assert [
            (r.perceptron_index, r.softcore_count, r.total_area) for r in after.layers
        ] == [
            (r.perceptron_index, r.softcore_count, r.total_area)
            for r in original.layers
        ]

    def test_a_core_with_no_layer_identity_falls_back_to_its_name(self):
        relay = LayoutSoftCoreSpec(
            input_count=4, output_count=4, segment_id=0, latency_tag=2,
            name="relay_0", perceptron_index=None,
        )
        payload = INTROSPECTION_REGISTRY.serve(
            "layer_rollup",
            CandidateLayoutView.from_platform(_conv_then_fc() + [relay], PLATFORM),
        )
        tail = payload.layers[-1]
        assert tail.perceptron_index is None
        assert tail.layer == "relay_0"

    def test_a_record_rolls_up_on_the_same_key(self):
        payload = INTROSPECTION_REGISTRY.serve("layer_rollup", _record_view())
        assert [row.perceptron_index for row in payload.layers] == [3]


class TestBankComposition:
    def test_sharing_degree_is_visible(self):
        payload = INTROSPECTION_REGISTRY.serve("bank_composition", _candidate())
        assert isinstance(payload, BankCompositionPayload)
        assert len(payload.banks) == 1
        bank = payload.banks[0]
        assert (bank.bank_id, bank.softcore_count) == (0, 3)
        assert bank.perceptron_index == 0
        assert payload.unbanked_softcore_count == 1

    def test_a_sealed_record_adds_the_real_bank_size(self):
        payload = INTROSPECTION_REGISTRY.serve("bank_composition", _record_view())
        bank = payload.banks[0]
        assert (bank.rows, bank.cols, bank.params) == (100, 50, 5000)


class TestSchedulePayload:
    def test_a_candidate_reports_the_declared_policy(self):
        payload = INTROSPECTION_REGISTRY.serve(
            "schedule",
            _candidate(allow_scheduling=True, schedule_policy="bank_clustered"),
        )
        assert isinstance(payload, SchedulePayload)
        assert payload.schedule_policy == "bank_clustered"
        assert payload.max_schedule_passes == 8
        assert [row.segment_index for row in payload.segments] == [0]

    def test_a_sealed_record_reports_the_deployed_programming_classes(self):
        payload = INTROSPECTION_REGISTRY.serve("schedule", _record_view())
        assert (payload.pass_count, payload.sync_count) == (2, 1)
        assert (payload.reprogram_passes, payload.reuse_passes) == (1, 1)
        assert payload.compute_op_count == 1
        assert [row.programming for row in payload.segments] == ["resident"]


class TestCapabilitiesPayload:
    def test_every_declared_bit_is_served(self):
        payload = INTROSPECTION_REGISTRY.serve("capabilities", _candidate())
        assert isinstance(payload, CapabilitiesPayload)
        assert set(payload.bits) == set(ChipCapabilities().capability_bits())
        assert "schedule_policy" in payload.bits

    def test_no_served_bit_is_an_unread_default(self):
        """This is a trust channel: a served value must be the platform's answer.

        ``max_axons``/``max_neurons``/``hardware_bias`` were served as
        ``None``/``None``/``False`` while the doc promised "the complete set" —
        for a platform that declares 64x64 cores.
        """
        bits = INTROSPECTION_REGISTRY.serve("capabilities", _candidate()).bits
        assert [name for name, value in bits.items() if value is None] == []
        assert bits["max_axons"] == 64
        assert bits["max_neurons"] == 64
        assert bits["hardware_bias"] is True

    def test_a_record_serves_them_from_its_resolved_platform(self):
        payload = INTROSPECTION_REGISTRY.serve("capabilities", _record_view())
        assert set(payload.bits) == set(ChipCapabilities().capability_bits())
        assert payload.bits["max_axons"] is not None

    def test_the_registered_doc_describes_the_geometry_it_serves(self):
        """The doc IS the agent-facing tool description; it must not overpromise."""
        doc = INTROSPECTION_REGISTRY.get("capabilities").doc
        assert "EFFECTIVE per-core limits" in doc


class TestLayoutStatsPayload:
    def test_it_types_through_the_records_own_mirror(self):
        payload = INTROSPECTION_REGISTRY.serve("layout_stats", _candidate())
        assert isinstance(payload, LayoutStatsPayload)
        assert payload.typed().feasible is True


class TestVersioningAndRoundTrip:
    @pytest.mark.parametrize(
        "name", ["softcores", "layer_rollup", "bank_composition", "schedule",
                 "capabilities", "layout_stats"],
    )
    def test_candidate_payloads_round_trip_through_json(self, name):
        payload = INTROSPECTION_REGISTRY.serve(name, _candidate())
        data = json.loads(json.dumps(payload.to_dict(), default=str))
        assert data["payload"] == name
        assert data["payload_version"] == payload.VERSION
        assert type(payload).from_dict(data) == payload

    def test_record_only_payload_round_trips(self):
        payload = INTROSPECTION_REGISTRY.serve("placement", _record_view())
        data = json.loads(json.dumps(payload.to_dict()))
        assert PlacementPayload.from_dict(data) == payload

    def test_a_wrong_version_is_refused_rather_than_tolerated(self):
        data = INTROSPECTION_REGISTRY.serve("softcores", _candidate()).to_dict()
        data["payload_version"] = 99
        with pytest.raises(ValueError, match="migrate explicitly"):
            SoftcoresPayload.from_dict(data)

    def test_a_payload_will_not_load_under_another_name(self):
        data = INTROSPECTION_REGISTRY.serve("softcores", _candidate()).to_dict()
        with pytest.raises(ValueError, match="cannot load payload"):
            LayerRollupPayload.from_dict(data)

    def test_an_unknown_field_is_refused(self):
        data = INTROSPECTION_REGISTRY.serve("schedule", _candidate()).to_dict()
        data["invented"] = 1
        with pytest.raises(ValueError, match="unknown fields"):
            SchedulePayload.from_dict(data)

    def test_serve_all_dicts_is_self_describing(self):
        served = INTROSPECTION_REGISTRY.serve_all_dicts(_candidate())
        assert all(
            envelope["payload"] == name for name, envelope in served.items()
        )
        json.dumps(served, default=str)


class TestRegistryIsolation:
    def test_build_registry_returns_independent_registries(self):
        first, second = build_registry(), build_registry()
        assert isinstance(first, IntrospectionRegistry)
        assert first is not second
        assert first.names() == second.names()
