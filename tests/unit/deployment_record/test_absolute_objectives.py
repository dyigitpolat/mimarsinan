"""The absolute axes: one Backing answering at BOTH completenesses, physics-gated."""

from dataclasses import replace

import pytest

from mimarsinan.deployment_record.objectives.catalog import OBJECTIVES
from mimarsinan.deployment_record.objectives.probes import (
    CANDIDATE_FRAGMENTS,
    candidate_capability_probe,
    candidate_probe_without,
    run_capability_probe,
)
from mimarsinan.deployment_record.objectives.views import (
    CandidateStaticView,
    DeploymentRecordView,
)
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.quantities import CandidateQuantityContext
from mimarsinan.pipelining.core.platform_constraints_resolver import (
    build_platform_constraints_resolved,
)

from unit.deployment_record.record_fixtures import (
    make_full_record,
    make_identity,
    make_layout,
)

ABSOLUTE_KEYS = (
    "chip_area_mm2",
    "energy_per_inference_mj",
    "e2e_latency_s",
    "throughput_inferences_s",
)

_TRUENORTH = get_platform_physics("truenorth")


def _candidate(*, physics=_TRUENORTH, **context_over):
    context = CandidateQuantityContext(**{
        "timesteps": 32, "activity_factor": 0.05, "weight_bits": 8,
        "cores_physical": 20, "neurons_physical": 5120, "axons_physical": 5120,
        "host_macs": 0, "onchip_macs": 100000, **context_over,
    })
    return CandidateStaticView(
        layout=make_layout(),
        chip_param_capacity=20 * 256 * 256.0,
        total_params=1000.0,
        host_side_segment_count=1,
        physics=physics,
        quantity_context=context,
    )


def _record_view(profile="truenorth"):
    platform = build_platform_constraints_resolved({
        "cores": [{"max_axons": 256, "max_neurons": 256, "count": 20}],
        "platform_physics_profile": profile,
    })
    record = make_full_record()
    return DeploymentRecordView(
        record=replace(record, identity=replace(make_identity(), platform=platform))
    )


class TestRegistration:
    def test_the_absolute_axes_are_registered(self):
        for key in ABSOLUTE_KEYS:
            assert OBJECTIVES.get(key) is not None

    def test_they_declare_units_directions_and_modeled_provenance(self):
        expected = {
            "chip_area_mm2": ("min", "mm^2"),
            "energy_per_inference_mj": ("min", "mJ"),
            "e2e_latency_s": ("min", "s"),
            "throughput_inferences_s": ("max", "inferences/s"),
        }
        for key, (direction, unit) in expected.items():
            spec = OBJECTIVES.get(key)
            assert (spec.direction, spec.unit) == (direction, unit)
            assert spec.provenance == "modeled"

    def test_they_register_after_the_legacy_eight(self):
        """Catalog order is contract: every optimizer's vector prefix must survive."""
        keys = list(OBJECTIVES.keys())
        assert keys[:8] == [
            "estimated_accuracy", "total_params", "total_param_capacity",
            "total_sync_barriers", "param_utilization_pct", "neuron_wastage_pct",
            "axon_wastage_pct", "fragmentation_pct",
        ]
        for key in ABSOLUTE_KEYS:
            assert keys.index(key) > keys.index("throughput_samples_per_s")

    def test_their_requires_names_the_physics_declaration(self):
        for key in ABSOLUTE_KEYS:
            assert "physics" in OBJECTIVES.get(key).requires


class TestCandidateCompleteness:
    def test_a_candidate_with_physics_answers_the_area_axis(self):
        view = _candidate()
        assert OBJECTIVES.get("chip_area_mm2").available(view)
        assert OBJECTIVES.get("chip_area_mm2").value(view) == pytest.approx(
            20 * 0.0936 + 46.6
        )

    def test_a_candidate_prices_energy_from_the_declared_activity(self):
        view = _candidate()
        assert OBJECTIVES.get("energy_per_inference_mj").available(view)

    def test_a_candidate_without_physics_answers_none_of_them(self):
        view = _candidate(physics=None)
        for key in ABSOLUTE_KEYS:
            assert not OBJECTIVES.get(key).available(view)

    def test_the_legacy_axes_are_untouched_by_the_new_fields(self):
        view = _candidate()
        assert OBJECTIVES.get("total_params").value(view) == 1000.0
        assert OBJECTIVES.get("fragmentation_pct").available(view)


class TestRecordCompleteness:
    def test_a_sealed_record_with_physics_answers_the_area_axis(self):
        view = _record_view()
        assert OBJECTIVES.get("chip_area_mm2").available(view)
        assert OBJECTIVES.get("chip_area_mm2").value(view) == pytest.approx(
            20 * 0.0936 + 46.6
        )

    def test_a_sealed_record_without_physics_answers_none_of_them(self):
        view = DeploymentRecordView(record=make_full_record())
        for key in ABSOLUTE_KEYS:
            assert not OBJECTIVES.get(key).available(view)

    def test_the_measured_axes_still_answer_beside_the_priced_ones(self):
        view = _record_view()
        assert OBJECTIVES.get("mj_per_sample").available(view)
        assert OBJECTIVES.get("mj_per_sample").value(view) == 1.0

    def test_an_unavailable_cost_term_never_raises_through_availability(self):
        """The partial-report trap: energy is refused on this record, and asking
        must answer False rather than exploding inside find_term."""
        view = _record_view()
        assert OBJECTIVES.get("energy_per_inference_mj").available(view) is False


class TestProbes:
    def test_physics_and_quantity_context_are_candidate_fragments(self):
        assert "physics" in CANDIDATE_FRAGMENTS
        assert "quantity_context" in CANDIDATE_FRAGMENTS

    def test_the_capability_probe_advertises_the_absolute_axes(self):
        probe = candidate_capability_probe("joint")
        for key in ABSOLUTE_KEYS:
            assert OBJECTIVES.get(key).available(probe), key

    def test_dropping_physics_drops_exactly_the_absolute_axes(self):
        without = candidate_probe_without("physics")
        lost = {
            spec.key for spec in OBJECTIVES.all()
            if spec.available(candidate_capability_probe("joint"))
            and not spec.available(without)
        }
        assert lost == set(ABSOLUTE_KEYS)

    def test_dropping_the_layout_keeps_area_answerable(self):
        """An area-only hardware search never packs: capacity + physics suffice, so
        the layoutless path must stay legitimate (test_joint_error_contract)."""
        without = candidate_probe_without("layout")
        assert OBJECTIVES.get("chip_area_mm2").available(without)
        assert not OBJECTIVES.get("fragmentation_pct").available(without)

    def test_dropping_the_quantity_context_drops_every_absolute_axis(self):
        """The context carries the run's DECLARATIONS — the core count area is priced
        per, the window length latency is priced over. Without them no absolute axis
        has a multiplicand, and each must say so rather than price a partial chip."""
        without = candidate_probe_without("quantity_context")
        for key in ABSOLUTE_KEYS:
            assert not OBJECTIVES.get(key).available(without), key
        assert OBJECTIVES.get("fragmentation_pct").available(without), (
            "the layout axes are untouched by the physics fragments"
        )


class TestRunGate:
    def test_the_run_probe_without_physics_makes_them_unavailable(self):
        probe = run_capability_probe("joint", None)
        for key in ABSOLUTE_KEYS:
            assert not OBJECTIVES.get(key).available(probe)

    def test_the_run_probe_with_physics_makes_them_available(self):
        probe = run_capability_probe("joint", _TRUENORTH)
        for key in ABSOLUTE_KEYS:
            assert OBJECTIVES.get(key).available(probe), key

    def test_resolve_active_refuses_by_name_when_no_physics_is_declared(self):
        with pytest.raises(ValueError, match="chip_area_mm2"):
            OBJECTIVES.resolve_active(
                "joint", ("estimated_accuracy", "chip_area_mm2"),
                probe=run_capability_probe("joint", None),
            )

    def test_resolve_active_accepts_them_when_physics_is_declared(self):
        specs = OBJECTIVES.resolve_active(
            "joint", ("estimated_accuracy", "chip_area_mm2"),
            probe=run_capability_probe("joint", _TRUENORTH),
        )
        assert [spec.key for spec in specs] == ["estimated_accuracy", "chip_area_mm2"]

    def test_the_parameterless_form_stays_capability_level(self):
        """The wizard resolves the whole mode offer with no run context; that must
        keep working exactly as before."""
        specs = OBJECTIVES.resolve_active("joint", ("estimated_accuracy",))
        assert [spec.key for spec in specs] == ["estimated_accuracy"]
