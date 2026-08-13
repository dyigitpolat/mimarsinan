"""``search.results`` is a PROJECTION of the objectives registry — byte-equal, loud."""

import pytest

from mimarsinan.deployment_record.objectives import OBJECTIVES
from mimarsinan.search.results import (
    ACCURACY_OBJECTIVE_NAME,
    ALL_OBJECTIVES,
    ObjectiveSpec,
    default_objectives_for_mode,
    objectives_for_mode,
    resolve_active_objectives,
)

# The legacy tuple as literal data: every optimizer reads ``.name``/``.goal``
# off these in this order, so the projection is pinned byte-equal here.
LEGACY_TUPLE = (
    ObjectiveSpec("estimated_accuracy", "max"),
    ObjectiveSpec("total_params", "min"),
    ObjectiveSpec("total_param_capacity", "min"),
    ObjectiveSpec("total_sync_barriers", "min"),
    ObjectiveSpec("param_utilization_pct", "max"),
    ObjectiveSpec("neuron_wastage_pct", "min"),
    ObjectiveSpec("axon_wastage_pct", "min"),
    ObjectiveSpec("fragmentation_pct", "min"),
)

#: [C2] The vendor-priced axes the projection gained; they follow the legacy tuple,
#: so an optimizer's existing objective-vector prefix is unchanged.
PHYSICS_TUPLE = (
    ObjectiveSpec("chip_area_mm2", "min"),
    ObjectiveSpec("energy_per_inference_mj", "min"),
    ObjectiveSpec("e2e_latency_s", "min"),
    ObjectiveSpec("throughput_inferences_s", "max"),
)


class TestLegacyProjection:
    def test_all_objectives_opens_with_the_legacy_tuple_byte_equal(self):
        assert ALL_OBJECTIVES[: len(LEGACY_TUPLE)] == LEGACY_TUPLE
        assert ALL_OBJECTIVES == LEGACY_TUPLE + PHYSICS_TUPLE

    def test_the_projection_yields_the_frozen_legacy_dataclass(self):
        for spec in ALL_OBJECTIVES:
            assert type(spec) is ObjectiveSpec
            with pytest.raises(AttributeError):
                spec.goal = "min"

    def test_the_projection_tracks_the_registry_catalog(self):
        assert tuple((s.name, s.goal) for s in ALL_OBJECTIVES) == tuple(
            (s.key, s.direction) for s in OBJECTIVES.search_catalog()
        )

    def test_the_accuracy_name_constant_is_the_registry_key(self):
        assert ACCURACY_OBJECTIVE_NAME == "estimated_accuracy"
        assert OBJECTIVES.get(ACCURACY_OBJECTIVE_NAME).provenance == "training_proxy"

    def test_hardware_mode_excludes_only_accuracy(self):
        assert objectives_for_mode("hardware") == tuple(
            o for o in LEGACY_TUPLE if o.name != ACCURACY_OBJECTIVE_NAME
        ) + PHYSICS_TUPLE

    def test_other_modes_carry_every_objective(self):
        for mode in ("model", "joint"):
            assert objectives_for_mode(mode) == LEGACY_TUPLE + PHYSICS_TUPLE


class TestDefaultsUnchanged:
    def test_hardware_defaults(self):
        assert default_objectives_for_mode("hardware") == (
            "total_param_capacity",
            "param_utilization_pct",
            "neuron_wastage_pct",
            "axon_wastage_pct",
            "fragmentation_pct",
        )

    def test_model_defaults(self):
        assert default_objectives_for_mode("model") == (
            "estimated_accuracy", "total_params",
        )

    def test_joint_defaults(self):
        assert default_objectives_for_mode("joint") == (
            "estimated_accuracy",
            "total_params",
            "param_utilization_pct",
            "neuron_wastage_pct",
            "fragmentation_pct",
        )

    def test_every_default_resolves_in_its_mode(self):
        for mode in ("hardware", "model", "joint"):
            names = default_objectives_for_mode(mode)
            assert tuple(o.name for o in resolve_active_objectives(mode)) == names

    def test_an_empty_selection_falls_back_to_the_defaults(self):
        assert resolve_active_objectives("model", []) == resolve_active_objectives("model")


class TestResolutionIsLoud:
    """The silent drop is gone: an unknown or unavailable name aborts the run."""

    def test_an_unknown_name_raises_instead_of_being_dropped(self):
        with pytest.raises(ValueError, match="unknown objective 'not_an_objective'"):
            resolve_active_objectives("joint", ["total_params", "not_an_objective"])

    def test_accuracy_in_hardware_search_raises_instead_of_being_dropped(self):
        with pytest.raises(ValueError) as excinfo:
            resolve_active_objectives(
                "hardware", ["estimated_accuracy", "fragmentation_pct"]
            )
        assert "estimated_accuracy" in str(excinfo.value)
        assert "hardware" in str(excinfo.value)

    def test_a_wholly_unavailable_selection_never_degrades_to_defaults(self):
        with pytest.raises(ValueError):
            resolve_active_objectives("hardware", ["estimated_accuracy"])

    def test_a_record_only_axis_is_not_searchable_yet(self):
        with pytest.raises(ValueError, match="deployed_accuracy"):
            resolve_active_objectives("joint", ["deployed_accuracy"])

    def test_a_valid_selection_resolves_in_the_caller_order(self):
        resolved = resolve_active_objectives(
            "joint", ["fragmentation_pct", "estimated_accuracy"]
        )
        assert [o.name for o in resolved] == ["fragmentation_pct", "estimated_accuracy"]
        assert [o.goal for o in resolved] == ["min", "max"]
