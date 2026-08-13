"""The physics panel's payload: profiles, constants with evidence, completeness."""

import json

import pytest

from mimarsinan.deployment_record.platform_physics import (
    PHYSICS_GROUPS,
    available_profiles,
)
from mimarsinan.gui.wizard.physics_panel import (
    physics_completeness,
    physics_profile_options,
    physics_profile_panel,
)


class TestTheProfileOptions:
    def test_every_shipped_profile_is_offered(self):
        offered = {row["id"] for row in physics_profile_options()}
        assert offered == set(available_profiles())

    def test_each_option_carries_a_human_label(self):
        for row in physics_profile_options():
            assert row["label"]
            assert row["id"] in row["label"] or len(row["label"]) > len(row["id"])

    def test_the_options_are_json_safe(self):
        json.dumps(physics_profile_options())


class TestThePanelForNoProfile:
    def test_an_unselected_profile_declares_nothing(self):
        panel = physics_profile_panel("", {})
        assert panel["selected"] == ""
        assert panel["groups"] == []
        assert panel["validity"] is None

    def test_it_still_says_which_objectives_are_unavailable(self):
        panel = physics_profile_panel("", {})
        assert panel["completeness"]
        assert all(not row["available"] for row in panel["completeness"])

    def test_the_reason_names_a_missing_constant(self):
        panel = physics_profile_panel("", {})
        area = [r for r in panel["completeness"] if r["key"] == "chip_area_mm2"][0]
        assert area["missing"], "an unavailable axis must say what it needs"


class TestThePanelForADeclaredProfile:
    def _panel(self, overrides=None):
        return physics_profile_panel("truenorth", overrides or {})

    def test_it_carries_the_validity_domain(self):
        validity = self._panel()["validity"]
        assert validity["measurement_kind"] == "silicon"
        assert validity["technology_node_nm"] == 28.0
        assert validity["supply_v"] == 0.775

    def test_groups_follow_the_vocabulary_order(self):
        names = [group["group"] for group in self._panel()["groups"]]
        assert names == [g for g in PHYSICS_GROUPS if g in names]

    def test_every_vocabulary_constant_appears_declared_or_not(self):
        from mimarsinan.deployment_record.platform_physics import PHYSICS_CONSTANTS

        rows = [r for g in self._panel()["groups"] for r in g["constants"]]
        assert {r["key"] for r in rows} == set(PHYSICS_CONSTANTS)

    def test_a_declared_row_carries_its_value_unit_and_evidence(self):
        rows = {r["key"]: r for g in self._panel()["groups"] for r in g["constants"]}
        row = rows["e_synaptic_event_total"]
        assert row["declared"] is True
        assert row["nominal"] == 26.0
        assert row["unit"] == "pJ"
        assert row["evidence_kind"] == "published"
        assert "merolla2014a" in row["evidence_detail"]

    def test_an_undeclared_row_says_so_and_carries_no_number(self):
        rows = {r["key"]: r for g in self._panel()["groups"] for r in g["constants"]}
        row = rows["e_sync_barrier"]
        assert row["declared"] is False
        assert row["nominal"] is None
        assert row["evidence_kind"] is None

    def test_a_banded_row_shows_both_ends(self):
        rows = {r["key"]: r for g in self._panel()["groups"] for r in g["constants"]}
        cycle = rows["t_cycle"]
        assert cycle["low"] == 47.6
        assert cycle["high"] == 1000.0
        assert cycle["banded"] is True

    def test_a_point_row_is_not_banded(self):
        rows = {r["key"]: r for g in self._panel()["groups"] for r in g["constants"]}
        assert rows["e_synaptic_event_total"]["banded"] is False

    def test_every_row_carries_its_documentation(self):
        for group in self._panel()["groups"]:
            for row in group["constants"]:
                assert len(row["doc"]) >= 20


class TestTheCompletenessReadout:
    def test_truenorth_can_back_area_and_latency(self):
        rows = {r["key"]: r for r in physics_completeness("truenorth", {})}
        assert rows["chip_area_mm2"]["available"] is True
        assert rows["e2e_latency_s"]["available"] is True

    def test_an_axis_it_cannot_back_names_what_is_missing(self):
        """TrueNorth publishes no barrier cost, so a run cannot price sync energy
        alone — the readout must name the constant, not just say 'no'."""
        rows = {r["key"]: r for r in physics_completeness("", {})}
        assert rows["energy_per_inference_mj"]["available"] is False
        assert rows["energy_per_inference_mj"]["missing"]

    def test_an_override_can_complete_a_partial_profile(self):
        """The live behaviour the panel promises: declaring the missing constant
        flips its objective in the same render."""
        before = {r["key"]: r for r in physics_completeness("", {})}
        assert before["chip_area_mm2"]["available"] is False
        after = {
            r["key"]: r for r in physics_completeness("", {
                "area_per_core_total": {"nominal": 1.0, "note": "operator estimate"},
                "t_cycle": {"nominal": 1.0, "note": "operator estimate"},
            })
        }
        assert after["chip_area_mm2"]["available"] is True

    def test_every_row_carries_the_pickers_own_label(self):
        """A readout naming `chip_area_mm2` beside a chip reading "Chip Area (mm²)"
        would make the user match them up by hand."""
        for row in physics_completeness("truenorth", {}):
            assert row["label"]
            assert row["label"] != row["key"]

    def test_the_readout_is_json_safe(self):
        json.dumps(physics_completeness("truenorth", {}))


class TestOverridesInThePanel:
    def test_an_override_marks_its_row(self):
        panel = physics_profile_panel(
            "truenorth", {"t_cycle": {"nominal": 500.0, "note": "overclocked"}}
        )
        rows = {r["key"]: r for g in panel["groups"] for r in g["constants"]}
        assert rows["t_cycle"]["overridden"] is True
        assert rows["t_cycle"]["nominal"] == 500.0

    def test_an_override_can_declare_an_undeclared_constant(self):
        panel = physics_profile_panel(
            "truenorth", {"e_sync_barrier": {"nominal": 2.0, "note": "vendor said so"}}
        )
        rows = {r["key"]: r for g in panel["groups"] for r in g["constants"]}
        assert rows["e_sync_barrier"]["declared"] is True
        assert rows["e_sync_barrier"]["overridden"] is True

    def test_an_unrelated_row_is_untouched(self):
        panel = physics_profile_panel(
            "truenorth", {"t_cycle": {"nominal": 500.0, "note": "n"}}
        )
        rows = {r["key"]: r for g in panel["groups"] for r in g["constants"]}
        assert rows["e_synaptic_event_total"]["overridden"] is False

    def test_an_unknown_profile_fails_loud(self):
        with pytest.raises(KeyError, match="nosuchchip"):
            physics_profile_panel("nosuchchip", {})
