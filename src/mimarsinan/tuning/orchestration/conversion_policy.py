"""E4 keystone — the characterization-and-policy layer (propose → confirm → escalate).

The thin layer the contract/plan consults to pick a conversion recipe SAFELY, so a
mode-tuned recipe is never shipped as a silent default on a model where the mode
behaves differently. The final-recommendations keystone, in three verbs:

1. **PROPOSE** — :func:`propose_recipe` is the mode→recipe table (the PRIOR): per
   (firing × sync) cell it names the proven recipe (driver, conversion-health
   intent, the two-residual train/deploy-S hints, the probe assumptions it relies
   on). Validated on MNIST/mmixcore — hence a proposal, not a guarantee.
2. **CONFIRM** — a :class:`Characterizer` runs a cheap pre-flight CHARACTERIZE(model)
   pass (forward-mostly probes: cold-cascade liveness / ramp monotonicity /
   staircase-vs-LIF ceiling / firing-gain) and returns a :class:`CharacterizationResult`
   ``matches``/``reason``. The real probes are research thread R1; the default
   :class:`AlwaysMatchesCharacterizer` is the no-op that keeps this scaffolding inert.
3. **ESCALATE** — :func:`escalate_to_controller` rewrites a recipe to the controller
   fallback when the model does not match the recipe's assumptions, so the safe
   path is taken instead of a regression nobody asked for.

:class:`ConversionPolicy` orchestrates the three verbs. It is **DEFAULT-OFF**: until
``conversion_policy`` is set in config the layer is inert and returns a
:class:`ConversionDecision` naming the CURRENT behavior (driver=controller, no
characterization run) — so nothing runs and behavior is byte-identical. This is the
scaffolding that makes Fix B safe to switch on later (propose → confirm → escalate);
it is wired as a seam :class:`DeploymentPlan` / :class:`SpikingDeploymentContract`
expose, NOT enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional

OPTIMIZATION_DRIVER_CONTROLLER = "controller"
OPTIMIZATION_DRIVER_FAST = "fast"

# R2-grounded two-residual S allocation for the cascaded fire-once cell. The genuine
# fine-tune is S-NEGATIVE (train-at-32 collapses d9 to 0.770), so train low-S; deploy
# at the R1 staircase-ceiling S — capped at the train-aligned regime because >S32 HURTS
# d9 (the fire-once surrogate regresses the deep cascade with budget). STE hedge mix=0.5
# escapes the pure-genuine plateau (d6 train16/deploy32 = 0.968).
_CASCADED_TRAIN_S_HINT = 16
_CASCADED_DEPLOY_S_HINT = 32
_CASCADED_STE_MIX = 0.5

__all__ = [
    "OPTIMIZATION_DRIVER_CONTROLLER",
    "OPTIMIZATION_DRIVER_FAST",
    "ConversionRecipe",
    "CharacterizationResult",
    "Characterizer",
    "AlwaysMatchesCharacterizer",
    "ConversionDecision",
    "ConversionPolicy",
    "propose_recipe",
    "escalate_to_controller",
]


@dataclass(frozen=True)
class ConversionRecipe:
    """The proposed recipe for one (firing × sync) cell — the prior, not yet enacted.

    ``driver`` is the optimization-driver arm the recipe proposes (the fast ladder
    for the well-conditioned cells, the controller where adaptive rollback is
    needed). ``expects_conversion_health`` records whether the cell's deployed
    decode needs the conversion-health revive (E3 keying). ``train_s_hint`` /
    ``deploy_s_hint`` carry the two-residual S allocation (train low-S because the
    genuine FT is S-negative; deploy high-S for the quantization floor) — ``None``
    means "no hint, leave the configured S". ``staircase_ste`` / ``ste_mix`` name the
    STE-hedge refinement the cascaded recipe relies on (forward = the genuine cascade,
    backward = a clean staircase/genuine hedge) — ``None`` means "no STE intent".
    ``assumptions`` names the probe checks the recipe relies on, the ones a
    :class:`Characterizer` confirms before the recipe is trusted.
    """

    name: str
    driver: str
    expects_conversion_health: bool = False
    train_s_hint: Optional[int] = None
    deploy_s_hint: Optional[int] = None
    staircase_ste: Optional[bool] = None
    ste_mix: Optional[float] = None
    assumptions: tuple[str, ...] = ()
    escalated: bool = False


@dataclass(frozen=True)
class CharacterizationResult:
    """The verdict of a CHARACTERIZE(model) pass against a proposed recipe.

    ``matches`` is whether the model satisfies the recipe's assumptions (run fast
    when True, escalate when False). ``probes`` archives the per-probe readings for
    reproducibility; ``reason`` explains a mismatch (surfaced as the escalation
    reason). The real probes are research R1; this is the contract they fill.
    """

    matches: bool
    probes: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""


class Characterizer:
    """The pre-flight CHARACTERIZE(model) interface (propose → CONFIRM → escalate).

    The real probes (cold-cascade liveness / ramp monotonicity / staircase-vs-LIF
    ceiling / firing-gain) are research thread R1 — the stubs below are the seam
    they implement, forward-mostly and cheap. Subclasses override
    :meth:`characterize` (and, as the probes land, the per-probe hooks) to return a
    :class:`CharacterizationResult`. The default no-op is
    :class:`AlwaysMatchesCharacterizer`.
    """

    def characterize(
        self, *, model: Any, recipe: ConversionRecipe, context: Any = None
    ) -> CharacterizationResult:
        """Confirm ``recipe`` on ``model``; return whether it matches + why not."""
        raise NotImplementedError(
            "Characterizer.characterize is the R1 pre-flight seam; use "
            "AlwaysMatchesCharacterizer for the inert default or implement the probes"
        )

    # ── forward-mostly probe hooks (R1 — stubs/interfaces here) ──────────────
    def cold_cascade_live(self, *, model: Any, context: Any = None) -> bool:
        """Is the cold genuine cascade alive (no revive needed)? (R1 probe.)"""
        raise NotImplementedError

    def ramp_monotone(self, *, model: Any, context: Any = None) -> bool:
        """Is the rate→accuracy ramp monotone (no cliff)? (R1 probe.)"""
        raise NotImplementedError

    def staircase_lif_ceiling(self, *, model: Any, context: Any = None) -> float:
        """The staircase/LIF accuracy ceiling at this depth and S. (R1 probe.)"""
        raise NotImplementedError

    def firing_gain(self, *, model: Any, context: Any = None) -> float:
        """The per-depth firing-gain deficit (θ). (R1 probe.)"""
        raise NotImplementedError


class AlwaysMatchesCharacterizer(Characterizer):
    """The default no-op: every model matches, so the proposed recipe always stands.

    Keeps the scaffolding inert and conservative — with no real probes a model is
    never wrongly escalated; it is the byte-identical default until R1 lands the
    probes. (When the policy is enabled but no characterizer is supplied, this is
    the one used, so an enabled-but-unprobed model runs its proposed recipe.)
    """

    def characterize(
        self, *, model: Any, recipe: ConversionRecipe, context: Any = None
    ) -> CharacterizationResult:
        return CharacterizationResult(matches=True, probes={}, reason="")


@dataclass(frozen=True)
class ConversionDecision:
    """The resolved propose → confirm → escalate outcome the tuner/plan reads.

    ``enabled`` is whether the policy ran at all (default-off ⇒ False ⇒ the
    current controller behavior). ``recipe`` is the recipe that WON (the proposal,
    or its escalated form). ``driver`` is the optimization-driver arm to run.
    ``characterized`` / ``escalated`` record whether the confirm pass ran and
    whether it forced the controller fallback; ``escalation_reason`` carries the
    probe's mismatch reason.
    """

    enabled: bool
    recipe: ConversionRecipe
    driver: str
    characterized: bool
    escalated: bool
    escalation_reason: str = ""
    characterization: Optional[CharacterizationResult] = None

    @property
    def expects_conversion_health(self) -> bool:
        return self.recipe.expects_conversion_health


def propose_recipe(mode_policy: Any) -> ConversionRecipe:
    """PROPOSE the recipe for a (firing × sync) cell — the prior (a mode→recipe table).

    The conversion-health intent is hoisted off the recipe table onto the
    :class:`SpikingModePolicy` (``does_conversion_health_calibration``) so the
    prior agrees with the E3 calibration keying — the cascaded fire-once cell is
    the one whose deployed decode needs the revive, and its recipe names the probe
    assumptions a :class:`Characterizer` must confirm. Every cell's prior driver is
    ``controller`` here (default-off ⇒ the proposal equals current behavior until
    Fix B raises a fast prior); the proposer is the seam where the proven
    fast/lossless recipe lands per cell.
    """
    needs_health = bool(
        getattr(mode_policy, "does_conversion_health_calibration", False)
    )
    mode = str(getattr(mode_policy, "spiking_mode", "lif"))
    schedule = getattr(mode_policy, "schedule", None)
    name = mode if schedule is None else f"{mode}/{schedule}"

    assumptions: tuple[str, ...] = ()
    train_s_hint: Optional[int] = None
    deploy_s_hint: Optional[int] = None
    staircase_ste: Optional[bool] = None
    ste_mix: Optional[float] = None
    if needs_health:
        # The cascaded fire-once recipe (two-stage revive → refine) relies on these
        # forward-mostly probes; a model that fails them escalates to the controller.
        assumptions = (
            "cold_cascade_live",
            "ramp_monotone",
            "staircase_lif_ceiling",
            "firing_gain",
        )
        # R2: the two-residual S allocation (train low-S, deploy at the R1 ceiling-S
        # capped at the train-aligned regime) + the STE hedge that escapes the
        # pure-genuine deep-cascade plateau. Carried as hints, NOT enacted until Fix B
        # flips the policy on per cell (default-off ⇒ never consumed ⇒ byte-identical).
        train_s_hint = _CASCADED_TRAIN_S_HINT
        deploy_s_hint = _CASCADED_DEPLOY_S_HINT
        staircase_ste = True
        ste_mix = _CASCADED_STE_MIX

    return ConversionRecipe(
        name=name,
        driver=OPTIMIZATION_DRIVER_CONTROLLER,
        expects_conversion_health=needs_health,
        train_s_hint=train_s_hint,
        deploy_s_hint=deploy_s_hint,
        staircase_ste=staircase_ste,
        ste_mix=ste_mix,
        assumptions=assumptions,
    )


def escalate_to_controller(recipe: ConversionRecipe) -> ConversionRecipe:
    """ESCALATE: rewrite ``recipe`` to the controller fallback (driver=controller).

    The safety move when CHARACTERIZE(model) does not match the recipe's
    assumptions — switch WHO drives the rate to the adaptive controller, preserving
    the conversion-health intent (whether the cell needs the revive does not change
    with the driver). Marks the recipe ``escalated`` for provenance.
    """
    return replace(
        recipe, driver=OPTIMIZATION_DRIVER_CONTROLLER, escalated=True,
    )


class ConversionPolicy:
    """Orchestrate propose → confirm → escalate; DEFAULT-OFF (byte-identical).

    The seam :class:`DeploymentPlan` / :class:`SpikingDeploymentContract` expose.
    Until ``conversion_policy`` is set in config the layer is inert: :meth:`resolve`
    returns a :class:`ConversionDecision` naming the CURRENT behavior
    (driver=controller, no characterization run). When opted in, it proposes the
    recipe, confirms it on the model via the characterizer, and escalates to the
    controller on a mismatch.
    """

    CONFIG_KEY = "conversion_policy"

    @staticmethod
    def default_characterizer() -> Characterizer:
        """The conservative no-op characterizer used when none is supplied."""
        return AlwaysMatchesCharacterizer()

    @classmethod
    def resolve(
        cls,
        config: Mapping[str, Any],
        *,
        mode_policy: Any,
        model: Any = None,
        characterizer: Optional[Characterizer] = None,
        context: Any = None,
    ) -> ConversionDecision:
        recipe = propose_recipe(mode_policy)

        if not bool(config.get(cls.CONFIG_KEY, False)):
            # Inert: the current controller behavior, characterizer never consulted.
            return ConversionDecision(
                enabled=False,
                recipe=replace(recipe, driver=OPTIMIZATION_DRIVER_CONTROLLER),
                driver=OPTIMIZATION_DRIVER_CONTROLLER,
                characterized=False,
                escalated=False,
            )

        char = characterizer or cls.default_characterizer()
        # ``context`` is the calibration-batch source (the trainer); a None-context
        # characterizer (the inert default) ignores it. Backward-compatible keyword.
        result = char.characterize(model=model, recipe=recipe, context=context)

        if result.matches:
            return ConversionDecision(
                enabled=True,
                recipe=recipe,
                driver=recipe.driver,
                characterized=True,
                escalated=False,
                characterization=result,
            )

        escalated = escalate_to_controller(recipe)
        return ConversionDecision(
            enabled=True,
            recipe=escalated,
            driver=escalated.driver,
            characterized=True,
            escalated=True,
            escalation_reason=result.reason,
            characterization=result,
        )
