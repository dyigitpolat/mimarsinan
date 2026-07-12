"""Advisory engine: evaluation entry points over the tentative-theory rule set."""

from __future__ import annotations

from typing import Any

from mimarsinan.advisories.advisory import (
    SEVERITIES as SEVERITIES,
    SEVERITY_INFO as SEVERITY_INFO,
    SEVERITY_RISK as SEVERITY_RISK,
    SEVERITY_UNSUPPORTED as SEVERITY_UNSUPPORTED,
    Advisory as Advisory,
    lossless_mandate_applies as lossless_mandate_applies,
)
from mimarsinan.advisories.rules_config import (
    ADV_CASC_UNSUPPORTED,
    ADV_ENVELOPE_GATE,
    ADV_NOVENA_CHARGE,
    ADV_STRICT_LT_LATTICE,
    CONFIG_RULES,
    rule_envelope_gate,
)
from mimarsinan.advisories.rules_graph import (
    ADV_FANIN_DEPTH_IMBALANCE,
    ADV_STAIRCASE_DEPTH,
    GRAPH_RULES,
)
from mimarsinan.advisories.rules_graph_scale import (
    ADV_BIAS_GRID_DOMINANCE,
    ADV_NORMFREE_CHAIN,
    ADV_SCALE_SPREAD,
)
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

ALL_ADVISORY_IDS = frozenset({
    ADV_CASC_UNSUPPORTED,
    ADV_NOVENA_CHARGE,
    ADV_STRICT_LT_LATTICE,
    ADV_ENVELOPE_GATE,
    ADV_STAIRCASE_DEPTH,
    ADV_SCALE_SPREAD,
    ADV_NORMFREE_CHAIN,
    ADV_BIAS_GRID_DOMINANCE,
    ADV_FANIN_DEPTH_IMBALANCE,
})


def _plan_of(plan_or_config: Any) -> Any:
    """A resolved ``DeploymentPlan`` from either a plan or a resolved config
    dict — the rules read decision axes from the plan SSOT, never raw flags."""
    if isinstance(plan_or_config, dict):
        return DeploymentPlan.resolve(plan_or_config)
    if isinstance(plan_or_config, DeploymentPlan):
        return plan_or_config
    raise TypeError(
        "expected a DeploymentPlan or a resolved config dict; "
        f"got {type(plan_or_config).__name__}"
    )


def evaluate_config_advisories(plan_or_config: Any) -> list[Advisory]:
    """Config-time advisories for a resolved deployment plan (or config)."""
    plan = _plan_of(plan_or_config)
    advisories: list[Advisory] = []
    for rule in CONFIG_RULES:
        advisories.extend(rule(plan))
    return advisories


def evaluate_graph_advisories(
    model_repr: Any, plan_or_config: Any, *, channel_stats: dict | None = None
) -> list[Advisory]:
    """Graph advisories over the mapper ``ModelRepresentation``.

    ``channel_stats`` maps ``id(perceptron)`` to per-channel activation q99
    values (install_resolution capture); absent stats fall back to
    weight-based proxies, stated in the rule detail.
    """
    plan = _plan_of(plan_or_config)
    advisories: list[Advisory] = []
    for rule in GRAPH_RULES:
        advisories.extend(rule(model_repr, plan, channel_stats))
    return advisories


def evaluate_post_pretrain_advisories(
    pretrain_acc: float, config: dict, *, acceptance_target: Any = None
) -> list[Advisory]:
    """Post-pretraining advisories: the envelope-gate reachability rule."""
    return rule_envelope_gate(pretrain_acc, config, acceptance_target)
