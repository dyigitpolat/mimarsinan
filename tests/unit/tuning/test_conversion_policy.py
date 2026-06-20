"""E4 keystone — the characterization-and-policy layer (propose → confirm → escalate).

A thin layer the contract/plan consults to pick a conversion recipe SAFELY:

1. PROPOSE a recipe per (firing × sync) cell — the prior, a mode→recipe table;
2. a cheap pre-flight CHARACTERIZE(model) hook confirms it on THIS model
   (forward-mostly probes; the real probes are research R1, stubs here);
3. ESCALATE to the controller fallback when the model does not match the
   recipe's assumptions, rather than shipping a silent regression.

DEFAULT-OFF / byte-identical: until ``conversion_policy`` is enabled the layer is
inert — it names the CURRENT behavior (driver=controller, no characterization run)
so nothing changes. This is the scaffolding Fix B switches on later.
"""

from __future__ import annotations

import pytest

from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode
from mimarsinan.tuning.orchestration.conversion_policy import (
    OPTIMIZATION_DRIVER_CONTROLLER,
    OPTIMIZATION_DRIVER_FAST,
    AlwaysMatchesCharacterizer,
    Characterizer,
    CharacterizationResult,
    ConversionDecision,
    ConversionPolicy,
    ConversionRecipe,
    escalate_to_controller,
    propose_recipe,
)

_CELLS = [
    ("lif", None),
    ("rate", None),
    ("ttfs", "cascaded"),
    ("ttfs_quantized", "cascaded"),
    ("ttfs_cycle_based", "cascaded"),
    ("ttfs_cycle_based", "synchronized"),
]


def _policy(mode, schedule=None):
    return policy_for_spiking_mode(mode, schedule)


# ── (1) the PROPOSER — the mode→recipe table (the prior) ──────────────────────


class TestProposeRecipe:
    @pytest.mark.parametrize("mode,schedule", _CELLS)
    def test_every_cell_proposes_a_named_recipe(self, mode, schedule):
        recipe = propose_recipe(_policy(mode, schedule))
        assert isinstance(recipe, ConversionRecipe)
        assert recipe.name
        assert recipe.driver in (
            OPTIMIZATION_DRIVER_CONTROLLER,
            OPTIMIZATION_DRIVER_FAST,
        )

    def test_cascaded_cycle_recipe_expects_conversion_health(self):
        # The cascaded fire-once cell is the one whose deployed decode needs the
        # conversion-health revive — the recipe records that prior + its assumptions.
        recipe = propose_recipe(_policy("ttfs_cycle_based", "cascaded"))
        assert recipe.expects_conversion_health is True
        assert recipe.assumptions, "the cascaded recipe relies on named probe checks"

    def test_lif_recipe_expects_no_conversion_health(self):
        recipe = propose_recipe(_policy("lif"))
        assert recipe.expects_conversion_health is False

    def test_synchronized_cycle_recipe_expects_no_conversion_health(self):
        recipe = propose_recipe(_policy("ttfs_cycle_based", "synchronized"))
        assert recipe.expects_conversion_health is False

    def test_recipe_is_frozen(self):
        recipe = propose_recipe(_policy("lif"))
        with pytest.raises(Exception):
            recipe.driver = OPTIMIZATION_DRIVER_FAST  # type: ignore[misc]


# ── (2) the CHARACTERIZER interface + default no-op (always-matches) ──────────


class TestCharacterizerInterface:
    def test_base_characterizer_probes_are_abstract(self):
        base = Characterizer()
        recipe = propose_recipe(_policy("ttfs_cycle_based", "cascaded"))
        with pytest.raises(NotImplementedError):
            base.characterize(model=object(), recipe=recipe)

    def test_always_matches_returns_a_matching_result(self):
        char = AlwaysMatchesCharacterizer()
        recipe = propose_recipe(_policy("ttfs_cycle_based", "cascaded"))
        result = char.characterize(model=object(), recipe=recipe)
        assert isinstance(result, CharacterizationResult)
        assert result.matches is True

    def test_always_matches_is_the_module_default(self):
        # The default characterizer the policy uses when none is supplied.
        assert isinstance(ConversionPolicy.default_characterizer(), Characterizer)
        assert isinstance(
            ConversionPolicy.default_characterizer(), AlwaysMatchesCharacterizer
        )


class TestCharacterizationResult:
    def test_mismatch_carries_a_reason(self):
        r = CharacterizationResult(matches=False, probes={}, reason="cold cascade dead")
        assert r.matches is False
        assert r.reason == "cold cascade dead"

    def test_match_default_reason_is_empty(self):
        r = CharacterizationResult(matches=True, probes={"ramp_monotone": True})
        assert r.matches is True
        assert r.reason == ""
        assert r.probes == {"ramp_monotone": True}


# ── (3) the ESCALATE hook — controller fallback ──────────────────────────────


class TestEscalateHook:
    def test_escalate_forces_the_controller_driver(self):
        proposed = propose_recipe(_policy("ttfs_cycle_based", "cascaded"))
        escalated = escalate_to_controller(proposed)
        assert escalated.driver == OPTIMIZATION_DRIVER_CONTROLLER

    def test_escalate_preserves_the_conversion_health_intent(self):
        # Escalation changes WHO drives the rate (controller fallback), not whether
        # the cell needs conversion-health calibration.
        proposed = propose_recipe(_policy("ttfs_cycle_based", "cascaded"))
        escalated = escalate_to_controller(proposed)
        assert escalated.expects_conversion_health == proposed.expects_conversion_health

    def test_escalate_marks_the_recipe_as_escalated(self):
        proposed = propose_recipe(_policy("lif"))
        escalated = escalate_to_controller(proposed)
        assert escalated.escalated is True
        assert proposed.escalated is False


# ── (4) the ORCHESTRATOR — propose → confirm → escalate, DEFAULT-OFF ──────────


def _resolve(mode, schedule=None, characterizer=None, model=None, **cfg):
    return ConversionPolicy.resolve(
        dict(cfg),
        mode_policy=_policy(mode, schedule),
        model=model,
        characterizer=characterizer,
    )


class TestDefaultOffInert:
    """Until enabled, the layer is inert: it names the CURRENT behavior."""

    @pytest.mark.parametrize("mode,schedule", _CELLS)
    def test_disabled_by_default(self, mode, schedule):
        decision = _resolve(mode, schedule)
        assert decision.enabled is False

    @pytest.mark.parametrize("mode,schedule", _CELLS)
    def test_disabled_decision_is_the_controller_current_behavior(self, mode, schedule):
        # Byte-identical: every cell resolves to the controller driver, no
        # characterization run, no recipe enacted.
        decision = _resolve(mode, schedule)
        assert isinstance(decision, ConversionDecision)
        assert decision.driver == OPTIMIZATION_DRIVER_CONTROLLER
        assert decision.characterized is False
        assert decision.escalated is False

    def test_disabled_does_not_run_the_characterizer(self):
        char = _RecordingCharacterizer(matches=False)
        # Even a characterizer that would reject is never consulted while disabled.
        decision = _resolve("ttfs_cycle_based", "cascaded", characterizer=char)
        assert char.calls == 0
        assert decision.driver == OPTIMIZATION_DRIVER_CONTROLLER
        assert decision.enabled is False


class TestEnabledProposeConfirmEscalate:
    """When opted in, the propose → confirm → escalate flow runs."""

    def test_enabled_proposes_the_recipe(self):
        decision = _resolve("lif", conversion_policy=True)
        assert decision.enabled is True
        assert decision.recipe.name == propose_recipe(_policy("lif")).name

    def test_match_runs_the_proposed_recipe(self):
        # always-matches → the proposed recipe stands, characterization ran.
        decision = _resolve(
            "lif", conversion_policy=True, characterizer=AlwaysMatchesCharacterizer(),
        )
        assert decision.characterized is True
        assert decision.escalated is False
        assert decision.driver == propose_recipe(_policy("lif")).driver

    def test_mismatch_escalates_to_controller(self):
        decision = _resolve(
            "ttfs_cycle_based",
            "cascaded",
            conversion_policy=True,
            characterizer=_RecordingCharacterizer(matches=False, reason="cold dead"),
        )
        assert decision.characterized is True
        assert decision.escalated is True
        assert decision.driver == OPTIMIZATION_DRIVER_CONTROLLER
        assert decision.escalation_reason == "cold dead"

    def test_default_characterizer_is_always_matches_when_enabled(self):
        # Enabled without an explicit characterizer → the no-op always-matches one,
        # so a model that is never probed never escalates (the conservative default).
        decision = _resolve("ttfs_cycle_based", "cascaded", conversion_policy=True)
        assert decision.characterized is True
        assert decision.escalated is False

    def test_enabled_passes_the_model_to_the_characterizer(self):
        char = _RecordingCharacterizer(matches=True)
        sentinel = object()
        _resolve(
            "ttfs_cycle_based",
            "cascaded",
            conversion_policy=True,
            characterizer=char,
            model=sentinel,
        )
        assert char.calls == 1
        assert char.last_model is sentinel


class _RecordingCharacterizer(Characterizer):
    """Test double recording the model it was asked to characterize."""

    def __init__(self, *, matches: bool, reason: str = ""):
        self._matches = matches
        self._reason = reason
        self.calls = 0
        self.last_model = None

    def characterize(self, *, model, recipe, context=None) -> CharacterizationResult:
        self.calls += 1
        self.last_model = model
        return CharacterizationResult(
            matches=self._matches, probes={}, reason=self._reason,
        )
