"""The wizard's honest objective surface: per-mode availability straight from the registry."""

from mimarsinan.deployment_record.objectives import OBJECTIVES, SEARCH_MODES
from mimarsinan.gui.wizard.schema import get_wizard_nas_schema

ROW_KEYS = {
    "id", "label", "goal", "provenance", "available_in_modes", "unavailable_reason",
    "requires_physics", "requires_activity",
}


def _catalog():
    return {row["id"]: row for row in get_wizard_nas_schema()["objective_catalog"]}


class TestObjectiveCatalogSurface:
    def test_every_registered_objective_is_served(self):
        assert tuple(_catalog()) == OBJECTIVES.keys()

    def test_every_row_carries_the_declared_keys(self):
        for row in _catalog().values():
            assert set(row) == ROW_KEYS

    def test_rows_mirror_the_registry_direction_and_provenance(self):
        for key, row in _catalog().items():
            spec = OBJECTIVES.get(key)
            assert row["goal"] == spec.direction
            assert row["provenance"] == spec.provenance

    def test_labels_are_human_readable(self):
        catalog = _catalog()
        assert catalog["estimated_accuracy"]["label"] == "Estimated Accuracy"
        assert catalog["mj_per_sample"]["label"] == "Energy per Sample (mJ)"
        for row in catalog.values():
            assert row["label"]


class TestPerModeAvailability:
    def test_static_axes_are_available_in_every_search_mode(self):
        row = _catalog()["fragmentation_pct"]
        assert row["available_in_modes"] == list(SEARCH_MODES)
        assert row["unavailable_reason"] == ""

    def test_the_training_proxy_is_unavailable_in_hardware_only_search(self):
        row = _catalog()["estimated_accuracy"]
        assert row["available_in_modes"] == [m for m in SEARCH_MODES if m != "hardware"]
        assert OBJECTIVES.get("estimated_accuracy").requires in row["unavailable_reason"]

    def test_record_backed_axes_declare_no_search_mode_and_say_why(self):
        for key in ("mj_per_sample", "deployed_accuracy", "throughput_samples_per_s"):
            row = _catalog()[key]
            assert row["available_in_modes"] == []
            assert OBJECTIVES.get(key).requires in row["unavailable_reason"]


class TestLegacyOptionsUnchanged:
    """W5.3 owns the frontend switch; the served option list stays exactly as it was."""

    def test_objective_options_open_with_the_legacy_eight(self):
        """C2 appended the vendor-priced axes; the legacy eight keep their order."""
        options = get_wizard_nas_schema()["objective_options"]
        assert [o["id"] for o in options][:8] == [
            "estimated_accuracy",
            "total_params",
            "total_param_capacity",
            "total_sync_barriers",
            "param_utilization_pct",
            "neuron_wastage_pct",
            "axon_wastage_pct",
            "fragmentation_pct",
        ]
        assert set(options[0]) == {
            "id", "label", "goal", "requires_training", "requires_physics",
            "requires_activity",
        }
        assert options[0]["requires_training"] is True
        assert all(o["requires_training"] is False for o in options[1:])
        assert all(o["requires_physics"] is False for o in options[:8])
        # [C2] the four vendor-priced axes are physics-gated; [N3] the traffic
        # axis that follows them is a count and deliberately is NOT.
        assert all(o["requires_physics"] is True for o in options[8:12])
        assert [o["id"] for o in options[12:]] == [
            "noc_total_hops",
            # [H3] the chip-sizing axis; [H2] the carry axes, searchable now
            # the candidate sizes its own planned pass structure.
            "chip_occupancy_pct",
            "carry_peak_live_bytes", "carried_raster_bytes",
        ]
        assert options[12]["requires_physics"] is False


class TestEveryAxisIsNamedForAHuman:
    def test_no_row_falls_back_to_its_raw_key(self):
        """A greyed chip showing `chip_area_mm2` beside `Total Parameters` is the
        catalog leaking its schema into the UI."""
        for row in _catalog().values():
            assert row["label"] != row["id"], f"{row['id']} has no human label"
