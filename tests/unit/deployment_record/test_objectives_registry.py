"""Objectives registry v2: the legacy 8 preserved, the record axes added, loud resolution."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from mimarsinan.deployment_record.cost import DeploymentCostModel, find_term
from mimarsinan.deployment_record.platform_physics.probe import probe_physics
from mimarsinan.deployment_record.quantities import CandidateQuantityContext
from mimarsinan.deployment_record.schema import SegmentRecord
from mimarsinan.deployment_record.objectives import (
    CANDIDATE_FRAGMENTS,
    OBJECTIVES,
    SEARCH_MODES,
    CandidateStaticView,
    DeploymentRecordView,
    ObjectiveRegistry,
    ObjectiveSpecV2,
    candidate_capability_probe,
    candidate_probe_without,
    chip_param_capacity,
)

from unit.deployment_record.record_fixtures import (
    make_full_record,
    make_layout,
    make_timing,
)

# The legacy tuple, frozen HERE as literal data: keys, directions and ORDER.
# A change to this list is a change to every optimizer's objective vector.
LEGACY_EIGHT = (
    ("estimated_accuracy", "max"),
    ("total_params", "min"),
    ("total_param_capacity", "min"),
    ("total_sync_barriers", "min"),
    ("param_utilization_pct", "max"),
    ("neuron_wastage_pct", "min"),
    ("axon_wastage_pct", "min"),
    ("fragmentation_pct", "min"),
)

# Schema doc section 8's added axes, in the order the document lists them.
SPEC_ADDED_KEYS = (
    "deployed_accuracy",
    "mj_per_sample",
    "latency_steps",
    "host_op_wall_s",
    "total_spikes",
    "pass_count",
    "reprogram_passes",
    "reprogramming_bytes",
    "params_reloaded",
    "noc_inter_tile_packets",
    "noc_total_packets",
    "programming_energy_mj",
    "sync_barrier_energy_mj",
    "throughput_samples_per_s",
)

#: [C2] The vendor-priced axes, registered last so the legacy prefix never shifts.
PHYSICS_AXIS_KEYS = (
    "chip_area_mm2",
    "energy_per_inference_mj",
    "e2e_latency_s",
    "throughput_inferences_s",
)

#: [N3] The traffic axis, after the physics axes — the catalog only appends.
TRAFFIC_AXIS_KEYS = ("noc_total_hops",)

#: [B] The pass-buffer metrics, last of all — record-only mapping performance.
BUFFER_AXIS_KEYS = ("carry_peak_live_bytes", "carried_raster_bytes")

DISTINCT_LAYOUT = replace(
    make_layout(),
    mapped_params_pct=33.0,
    total_wasted_neurons_pct=22.0,
    total_wasted_axons_pct=11.0,
    fragmentation_pct=44.0,
    schedule_sync_count=5,
)


def record_view(**overrides) -> DeploymentRecordView:
    record = make_full_record()
    if overrides:
        record = replace(record, **overrides)
    return DeploymentRecordView(record=record)


def candidate_view(**overrides) -> CandidateStaticView:
    kwargs = dict(
        layout=DISTINCT_LAYOUT,
        chip_param_capacity=1024.0,
        total_params=512.0,
        host_side_segment_count=3,
        estimated_accuracy=0.91,
        physics=probe_physics(),
        quantity_context=CandidateQuantityContext(
            timesteps=8, latency_steps=8, activity_factor=0.1, weight_bits=8, tiles=1,
            cores_per_tile=1, tile_mesh_height=1,
            cores_physical=4, neurons_physical=64, axons_physical=64,
            host_macs=0, onchip_macs=512,
        ),
        noc_fragments=SimpleNamespace(
            pass_placements=(((0, 0),),),
            census=SimpleNamespace(
                pair_wires={}, input_wires=(2,), on_wires=(0,),
            ),
        ),
    )
    kwargs.update(overrides)
    return CandidateStaticView(**kwargs)


class TestLegacyEightPreserved:
    def test_search_catalog_opens_with_the_legacy_eight_byte_equal(self):
        """C2 added the vendor-priced axes; the legacy eight remain the byte-equal
        PREFIX, so no optimizer's existing objective vector shifts."""
        catalog = OBJECTIVES.search_catalog()
        head = tuple((s.key, s.direction) for s in catalog[: len(LEGACY_EIGHT)])
        assert head == LEGACY_EIGHT
        assert tuple(s.key for s in catalog[len(LEGACY_EIGHT):]) == (
            PHYSICS_AXIS_KEYS + TRAFFIC_AXIS_KEYS
        )

    def test_legacy_name_and_goal_properties_mirror_key_and_direction(self):
        for spec in OBJECTIVES.all():
            assert spec.name == spec.key
            assert spec.goal == spec.direction

    def test_registration_order_puts_the_legacy_eight_first(self):
        keys = OBJECTIVES.keys()
        assert keys[: len(LEGACY_EIGHT)] == tuple(k for k, _ in LEGACY_EIGHT)

    def test_hardware_mode_drops_only_the_training_proxy(self):
        hardware = tuple(s.key for s in OBJECTIVES.for_search_mode("hardware"))
        assert hardware == (
            tuple(k for k, _ in LEGACY_EIGHT if k != "estimated_accuracy")
            + PHYSICS_AXIS_KEYS
            + TRAFFIC_AXIS_KEYS
        )

    def test_every_other_search_mode_carries_all_eight(self):
        for mode in ("model", "joint"):
            keys = tuple(s.key for s in OBJECTIVES.for_search_mode(mode))
            assert keys == (tuple(k for k, _ in LEGACY_EIGHT) + PHYSICS_AXIS_KEYS
                            + TRAFFIC_AXIS_KEYS)

    def test_the_spec_documented_axes_are_all_registered(self):
        assert OBJECTIVES.keys() == (
            tuple(k for k, _ in LEGACY_EIGHT) + SPEC_ADDED_KEYS
            + PHYSICS_AXIS_KEYS + TRAFFIC_AXIS_KEYS + BUFFER_AXIS_KEYS
        )


class TestRegistration:
    def test_duplicate_key_raises(self):
        registry = ObjectiveRegistry()
        spec = OBJECTIVES.get("total_params")
        registry.register(spec)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(spec)

    def test_duplicate_key_raises_even_for_a_different_spec(self):
        registry = ObjectiveRegistry()
        registry.register(OBJECTIVES.get("total_params"))
        clashing = replace(OBJECTIVES.get("total_params"), direction="max")
        with pytest.raises(ValueError, match="already registered"):
            registry.register(clashing)

    def test_unknown_key_lookup_is_loud(self):
        with pytest.raises(ValueError, match="unknown objective 'no_such_axis'"):
            OBJECTIVES.get("no_such_axis")

    def test_every_spec_declares_unit_provenance_requirement_and_doc(self):
        for spec in OBJECTIVES.all():
            assert spec.unit
            assert spec.requires
            assert spec.doc
            assert spec.direction in ("min", "max")
            assert spec.provenance in (
                "measured", "modeled", "static", "training_proxy",
            )

    def test_spec_rejects_an_illegal_direction(self):
        spec = OBJECTIVES.get("total_params")
        with pytest.raises(ValueError, match="direction"):
            replace(spec, direction="minimise")

    def test_spec_rejects_an_illegal_provenance(self):
        spec = OBJECTIVES.get("total_params")
        with pytest.raises(ValueError, match="provenance"):
            replace(spec, provenance="vibes")

    def test_spec_is_frozen(self):
        spec = OBJECTIVES.get("total_params")
        with pytest.raises(AttributeError):
            spec.key = "other"


class TestAvailabilityOnTheCandidateView:
    def test_a_full_candidate_view_carries_the_legacy_eight_and_the_priced_axes(self):
        keys = tuple(s.key for s in OBJECTIVES.available_for(candidate_view()))
        assert keys == (tuple(k for k, _ in LEGACY_EIGHT) + PHYSICS_AXIS_KEYS
                            + TRAFFIC_AXIS_KEYS)

    def test_a_candidate_without_physics_carries_exactly_the_legacy_eight(self):
        """The C2 gate: no declared profile, no vendor-priced axis. The [N3]
        traffic axis survives the gate — hops are a count, not a priced term."""
        keys = tuple(s.key for s in OBJECTIVES.available_for(candidate_view(physics=None)))
        assert keys == tuple(k for k, _ in LEGACY_EIGHT) + TRAFFIC_AXIS_KEYS

    def test_a_candidate_without_an_accuracy_estimate_drops_it(self):
        view = candidate_view(estimated_accuracy=None)
        keys = tuple(s.key for s in OBJECTIVES.available_for(view))
        assert "estimated_accuracy" not in keys
        assert len(keys) == (len(LEGACY_EIGHT) + len(PHYSICS_AXIS_KEYS)
                             + len(TRAFFIC_AXIS_KEYS) - 1)

    def test_a_candidate_without_the_host_segment_census_drops_sync_barriers(self):
        view = candidate_view(host_side_segment_count=None)
        keys = tuple(s.key for s in OBJECTIVES.available_for(view))
        assert "total_sync_barriers" not in keys
        assert "fragmentation_pct" in keys

    def test_no_record_backed_axis_is_available_on_a_candidate(self):
        keys = {s.key for s in OBJECTIVES.available_for(candidate_view())}
        assert not (keys & set(SPEC_ADDED_KEYS))

    def test_the_capability_probe_matches_the_mode_catalog(self):
        for mode in SEARCH_MODES:
            probe = candidate_capability_probe(mode)
            assert OBJECTIVES.available_for(probe) == OBJECTIVES.for_search_mode(mode)


class TestTheFragmentProbe:
    """``candidate_probe_without`` — "which axes need this fact?", asked of the registry."""

    def test_every_fragment_is_droppable_and_costs_at_least_one_axis(self):
        # A fragment nothing reads is a fragment no caller should be paying to
        # compute; a fragment name nothing answers is a typo waiting to happen.
        for fragment in CANDIDATE_FRAGMENTS:
            probe = candidate_probe_without(fragment)
            assert getattr(probe, fragment) is None
            lost = {
                s.key for s in OBJECTIVES.available_for(candidate_view())
            } - {s.key for s in OBJECTIVES.available_for(probe)}
            assert lost, f"no objective reads the {fragment!r} fragment"

    def test_dropping_the_layout_costs_exactly_the_layout_axes(self):
        probe = candidate_probe_without("layout")
        keys = {s.key for s in OBJECTIVES.available_for(probe)}
        # Area needs only the DECLARED chip and its physics, so an area-only
        # hardware search legitimately never packs.
        # Area needs only the declared chip, and energy only the MAC census times
        # the declared activity — neither needs a packing. Latency does (the window
        # runs once per neural segment), and throughput inverts latency.
        assert keys == {
            "estimated_accuracy", "total_params", "total_param_capacity",
            "chip_area_mm2", "energy_per_inference_mj",
        }

    def test_an_unknown_fragment_is_refused_by_name(self):
        # The probe is how a caller decides whether to compute something
        # expensive; a misspelled fragment would silently answer "nothing needs
        # it" and skip the work for every candidate.
        with pytest.raises(ValueError, match="unknown candidate fragment"):
            candidate_probe_without("layout_stats")
        with pytest.raises(ValueError, match="estimated_accuracy"):
            candidate_probe_without("accuracy")


class TestAvailabilityOnTheRecordView:
    def test_a_sealed_record_carries_the_layout_axes_and_the_record_axes(self):
        keys = tuple(s.key for s in OBJECTIVES.available_for(record_view()))
        assert keys == (
            "total_param_capacity",
            "param_utilization_pct",
            "neuron_wastage_pct",
            "axon_wastage_pct",
            "fragmentation_pct",
            "deployed_accuracy",
            "mj_per_sample",
            "latency_steps",
            "total_spikes",
            "pass_count",
            "reprogram_passes",
            "reprogramming_bytes",
            "params_reloaded",
            "noc_inter_tile_packets",
            "noc_total_packets",
            "programming_energy_mj",
            "sync_barrier_energy_mj",
            "throughput_samples_per_s",
            # [N3] derived from the sealed NoC census (Σ per-link loads).
            "noc_total_hops",
        )

    def test_the_search_side_axes_are_unavailable_on_a_record(self):
        keys = {s.key for s in OBJECTIVES.available_for(record_view())}
        assert "estimated_accuracy" not in keys
        assert "total_params" not in keys
        assert "total_sync_barriers" not in keys

    def test_a_record_without_the_energy_fragment_drops_every_energy_axis(self):
        keys = {s.key for s in OBJECTIVES.available_for(record_view(energy=None))}
        for key in (
            "mj_per_sample", "total_spikes", "programming_energy_mj",
            "sync_barrier_energy_mj", "throughput_samples_per_s",
        ):
            assert key not in keys
        assert "pass_count" in keys

    def test_a_record_without_noc_traffic_drops_the_packet_axes(self):
        keys = {s.key for s in OBJECTIVES.available_for(record_view(traffic=None))}
        assert "noc_inter_tile_packets" not in keys
        assert "noc_total_packets" not in keys

    def test_a_record_without_the_measured_timing_census_drops_latency_steps(self):
        view = record_view(timing=make_timing(with_per_segment=False))
        keys = {s.key for s in OBJECTIVES.available_for(view)}
        assert "latency_steps" not in keys
        assert "throughput_samples_per_s" not in keys

    def test_host_op_wall_appears_only_once_a_wall_was_timed(self):
        assert "host_op_wall_s" not in {
            s.key for s in OBJECTIVES.available_for(record_view())
        }
        timing = make_timing()
        timed = replace(
            timing,
            latency=replace(timing.latency, host_ops_s=0.25, host_ops_s_per_pass=0.05),
        )
        view = record_view(timing=timed)
        assert OBJECTIVES.get("host_op_wall_s").value(view) == 0.25

    def test_an_undeclared_core_capacity_makes_the_capacity_axis_unavailable(self):
        record = make_full_record()
        identity = replace(record.identity, platform={"weight_bits": 8})
        view = DeploymentRecordView(record=replace(record, identity=identity))
        assert "total_param_capacity" not in {
            s.key for s in OBJECTIVES.available_for(view)
        }


class TestExtractedValues:
    def test_the_layout_axes_read_the_fields_they_name(self):
        values = OBJECTIVES.extract(candidate_view())
        assert values["param_utilization_pct"] == 33.0
        assert values["neuron_wastage_pct"] == 22.0
        assert values["axon_wastage_pct"] == 11.0
        assert values["fragmentation_pct"] == 44.0

    def test_sync_barriers_add_the_host_segments_to_the_schedule_syncs(self):
        assert OBJECTIVES.extract(candidate_view())["total_sync_barriers"] == 8.0

    def test_the_candidate_carries_its_own_params_capacity_and_accuracy(self):
        values = OBJECTIVES.extract(candidate_view())
        assert values["total_params"] == 512.0
        assert values["total_param_capacity"] == 1024.0
        assert values["estimated_accuracy"] == 0.91

    def test_the_record_axes_read_the_sealed_fragments(self):
        values = OBJECTIVES.extract(record_view())
        assert values["deployed_accuracy"] == 0.97
        assert values["mj_per_sample"] == 1.0
        assert values["latency_steps"] == 64.0
        assert values["total_spikes"] == 987.0
        assert values["pass_count"] == 2.0
        assert values["reprogram_passes"] == 1.0
        assert values["params_reloaded"] == 100.0
        assert values["noc_inter_tile_packets"] == 400.0
        assert values["noc_total_packets"] == 1000.0

    def test_reprogramming_bytes_counts_only_the_reprogrammed_segments(self):
        record = make_full_record()
        reprogrammed = [
            seg for seg in record.schedule.segments() if seg.programming == "reprogram"
        ]
        expected = float(sum(seg.params_bytes for seg in reprogrammed))
        assert expected > 0.0
        assert OBJECTIVES.extract(record_view())["reprogramming_bytes"] == expected

    def test_reprogramming_bytes_follows_the_programming_kind_not_the_byte_field(self):
        """A resident pass moves no payload (schema §2.1) — the same rule the cost
        model's ``payload_bytes_band`` applies, so a stray resident byte count
        cannot inflate the axis."""
        record = make_full_record()
        stages = tuple(
            replace(stage, params_bytes=stage.params_bytes + 4096)
            if isinstance(stage, SegmentRecord) and stage.programming == "resident"
            else stage
            for stage in record.schedule.stages
        )
        inflated = replace(record, schedule=replace(record.schedule, stages=stages))
        assert OBJECTIVES.extract(DeploymentRecordView(record=inflated))[
            "reprogramming_bytes"
        ] == OBJECTIVES.extract(record_view())["reprogramming_bytes"]

    def test_the_record_capacity_uses_the_declared_platform_cores(self):
        cores = make_full_record().identity.platform["cores"]
        assert OBJECTIVES.extract(record_view())["total_param_capacity"] == (
            chip_param_capacity(cores)
        )

    def test_the_layout_axes_agree_between_a_record_and_a_candidate_view(self):
        record = make_full_record()
        utilization = replace(record.utilization, layout=DISTINCT_LAYOUT)
        view = DeploymentRecordView(record=replace(record, utilization=utilization))
        record_values = OBJECTIVES.extract(view)
        candidate_values = OBJECTIVES.extract(candidate_view())
        for key in (
            "param_utilization_pct", "neuron_wastage_pct",
            "axon_wastage_pct", "fragmentation_pct",
        ):
            assert record_values[key] == candidate_values[key]

    def test_the_modeled_axes_are_the_cost_model_terms_not_a_recomputation(self):
        record = make_full_record()
        report = DeploymentCostModel().evaluate(record)
        values = OBJECTIVES.extract(DeploymentRecordView(record=record))
        assert values["programming_energy_mj"] == find_term(
            report.energy, "modeled_programming_mj"
        ).value
        assert values["sync_barrier_energy_mj"] == find_term(
            report.energy, "modeled_sync_mj"
        ).value
        assert values["throughput_samples_per_s"] == find_term(
            report.throughput, "samples_per_s"
        ).value

    def test_every_extracted_value_is_a_float(self):
        for view in (candidate_view(), record_view()):
            for value in OBJECTIVES.extract(view).values():
                assert isinstance(value, float)


class TestFailLoud:
    def test_resolve_active_rejects_an_unknown_name(self):
        with pytest.raises(ValueError, match="unknown objective 'made_up'"):
            OBJECTIVES.resolve_active("joint", ["total_params", "made_up"])

    def test_resolve_active_rejects_a_name_unavailable_in_the_mode(self):
        with pytest.raises(ValueError) as excinfo:
            OBJECTIVES.resolve_active("hardware", ["estimated_accuracy"])
        message = str(excinfo.value)
        assert "estimated_accuracy" in message
        assert "hardware" in message
        assert OBJECTIVES.get("estimated_accuracy").requires in message

    def test_resolve_active_rejects_a_record_only_axis_in_search(self):
        with pytest.raises(ValueError, match="mj_per_sample"):
            OBJECTIVES.resolve_active("joint", ["mj_per_sample"])

    def test_resolve_active_rejects_an_empty_selection(self):
        with pytest.raises(ValueError, match="no objective names"):
            OBJECTIVES.resolve_active("joint", [])

    def test_resolve_active_rejects_a_repeated_name(self):
        with pytest.raises(ValueError, match="repeated"):
            OBJECTIVES.resolve_active("joint", ["total_params", "total_params"])

    def test_resolve_active_preserves_the_caller_order(self):
        names = ["fragmentation_pct", "total_params", "param_utilization_pct"]
        resolved = OBJECTIVES.resolve_active("joint", names)
        assert [s.key for s in resolved] == names

    def test_reading_an_unavailable_objective_raises_with_its_requirement(self):
        spec = OBJECTIVES.get("mj_per_sample")
        with pytest.raises(ValueError) as excinfo:
            spec.value(candidate_view())
        assert spec.requires in str(excinfo.value)
        assert "candidate_static" in str(excinfo.value)

    def test_extract_never_reports_an_unavailable_axis(self):
        view = record_view(energy=None)
        assert "mj_per_sample" not in OBJECTIVES.extract(view)


class TestModesAvailable:
    def test_static_axes_are_available_in_every_search_mode(self):
        assert OBJECTIVES.modes_available("fragmentation_pct") == SEARCH_MODES

    def test_the_training_proxy_is_unavailable_in_hardware_only_search(self):
        assert OBJECTIVES.modes_available("estimated_accuracy") == tuple(
            m for m in SEARCH_MODES if m != "hardware"
        )

    def test_record_backed_axes_are_available_in_no_search_mode(self):
        for key in SPEC_ADDED_KEYS:
            assert OBJECTIVES.modes_available(key) == ()

    def test_modes_available_is_loud_about_an_unknown_key(self):
        with pytest.raises(ValueError, match="unknown objective"):
            OBJECTIVES.modes_available("no_such_axis")


class TestSpecTypes:
    def test_the_registry_holds_v2_specs(self):
        assert all(isinstance(s, ObjectiveSpecV2) for s in OBJECTIVES.all())
