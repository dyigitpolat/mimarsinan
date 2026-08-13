"""The wizard's objective-chip rules, executed as the browser executes them.

``wizard/search_objectives.js`` decides which objective chips a search mode may
offer and which of them a draft's declaration selects. The registry ABORTS a
run on an axis the mode cannot measure, so that filter is the only thing
between a hardware-only draft and a dead run — and until it was a pure module,
DELETING it left the whole python suite green (the round-trip test re-implemented
the rule instead of running it). These tests execute the real module under node.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from js_module import JS_ROOT, call_js

from mimarsinan.deployment_record.objectives import SEARCH_MODES
from mimarsinan.gui.wizard.schema import get_wizard_nas_schema
from mimarsinan.pipelining.core.search_mode import derive_search_mode
from mimarsinan.search.results import objectives_for_mode

MODULE = JS_ROOT / "wizard" / "search_objectives.js"


def offered(
    nas: Dict[str, Any], search_mode: str, declares_physics: bool = False
) -> List[str]:
    return [
        option["id"]
        for option in call_js(
            MODULE, "offeredObjectives", nas, search_mode, declares_physics
        )
    ]


def payload(options: List[str], catalog: Dict[str, List[str]]) -> Dict[str, Any]:
    """A served-shaped ``nas`` payload: option list plus availability rows."""
    return {
        "objective_options": [{"id": name, "label": name, "goal": "min"} for name in options],
        "objective_catalog": [
            {"id": name, "available_in_modes": modes} for name, modes in catalog.items()
        ],
    }


class TestTheFilterIsTheRegistrysAnswer:
    @pytest.mark.parametrize("search_mode", sorted(SEARCH_MODES))
    def test_the_browser_offers_exactly_what_the_mode_can_measure(self, search_mode):
        """The SERVED payload through the REAL filter == the registry's availability.

        A draft that declares physics can back every axis its mode carries.
        """
        served = offered(get_wizard_nas_schema(), search_mode, True)
        assert set(served) == {spec.name for spec in objectives_for_mode(search_mode)}

    @pytest.mark.parametrize("search_mode", sorted(SEARCH_MODES))
    def test_a_draft_without_physics_is_never_offered_a_priced_axis(self, search_mode):
        """Offering one would be a chip whose run aborts at objective resolution:
        area in mm² and seconds of latency exist only against declared constants."""
        nas = get_wizard_nas_schema()
        priced = {
            row["id"] for row in nas["objective_catalog"] if row["requires_physics"]
        }
        assert priced, "C2 registered vendor-priced axes"
        assert not priced & set(offered(nas, search_mode))
        assert priced <= set(offered(nas, search_mode, True))

    def test_hardware_only_search_never_offers_the_training_proxy(self):
        nas = get_wizard_nas_schema()
        assert "estimated_accuracy" not in offered(nas, "hardware")
        assert "estimated_accuracy" in offered(nas, "model")

    def test_an_axis_available_elsewhere_is_dropped_here(self):
        nas = payload(["a", "b"], {"a": ["model"], "b": ["hardware", "joint"]})
        assert offered(nas, "hardware") == ["b"]
        assert offered(nas, "model") == ["a"]
        assert offered(nas, "joint") == ["b"]

    def test_an_axis_available_nowhere_is_offered_nowhere(self):
        nas = payload(["a"], {"a": []})
        assert offered(nas, "model") == []

    def test_an_option_with_no_catalog_row_is_offered_everywhere(self):
        """The documented escape hatch: reproduced here so the served payload's
        'every option has a row' assertion is the thing that closes it."""
        nas = payload(["a", "orphan"], {"a": ["model"]})
        assert offered(nas, "hardware") == ["orphan"]

    def test_an_empty_payload_offers_nothing_instead_of_throwing(self):
        assert offered({}, "hardware") == []


class TestTheSeededSelection:
    OFFER = [{"id": "x"}, {"id": "y"}, {"id": "z"}]

    def seeded(self, declared):
        return call_js(MODULE, "seededObjectiveIds", self.OFFER, declared)

    def test_an_undeclared_draft_starts_with_the_whole_offer(self):
        assert self.seeded(None) == ["x", "y", "z"]

    def test_a_declaration_keeps_its_own_subset(self):
        assert self.seeded(["z", "x"]) == ["z", "x"]

    def test_an_axis_this_mode_cannot_offer_is_not_selected(self):
        # The mode-flip case: a draft declared under `model` must not light up
        # a chip `hardware` never rendered — the run would abort on it.
        assert self.seeded(["x", "estimated_accuracy"]) == ["x"]

    def test_an_explicitly_empty_declaration_stays_empty(self):
        assert self.seeded([]) == []


class TestTheWidgetDelegatesToTheseRules:
    """The chips the browser draws come from THIS module, not a second copy.

    Executing `structured.js` itself would need a DOM; what a test can still
    say without one is that the widget imports these rules and does not carry
    its own availability logic beside them.
    """

    WIDGET = JS_ROOT / "wizard" / "structured.js"

    def test_the_widget_imports_the_shared_rules(self):
        source = self.WIDGET.read_text()
        assert "from './search_objectives.js'" in source
        for function in ("deriveSearchMode", "offeredObjectives", "seededObjectiveIds"):
            assert function in source, f"{function} is imported but never called"

    def test_the_availability_rule_exists_only_once(self):
        # `available_in_modes` is the catalog key the filter reads; a second
        # reader in the widget would be a rule this file cannot test.
        assert "available_in_modes" not in self.WIDGET.read_text()


class TestTheModeTheChipsAreFilteredBy:
    @pytest.mark.parametrize("model_mode,hw_mode,expected", [
        ("search", "search", "joint"),
        ("user", "search", "hardware"),
        ("search", "fixed", "model"),
    ])
    def test_it_agrees_with_the_pipelines_own_derivation(self, model_mode, hw_mode, expected):
        dp = {"model_config_mode": model_mode, "hw_config_mode": hw_mode}
        assert call_js(MODULE, "deriveSearchMode", dp) == expected
        assert derive_search_mode(dp) == expected

    def test_a_draft_that_searches_nothing_shows_the_model_chips(self):
        """The one documented divergence: the backend calls this `fixed` (no
        search runs at all), and the panel shows the widest offer rather than
        an empty box. Pinned so the divergence stays deliberate."""
        dp = {"model_config_mode": "user", "hw_config_mode": "fixed"}
        assert call_js(MODULE, "deriveSearchMode", dp) == "model"
        assert derive_search_mode(dp) == "fixed"
        assert "fixed" not in SEARCH_MODES

    def test_a_draft_without_the_keys_at_all_is_readable(self):
        assert call_js(MODULE, "deriveSearchMode", {}) == "model"
        assert call_js(MODULE, "deriveSearchMode", None) == "model"
