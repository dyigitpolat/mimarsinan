"""Registry entries: the executed residency plan selection and position multiplexing."""

from __future__ import annotations

from collections.abc import Mapping

from mimarsinan.config_schema.registry.decision_classes import COST_ONLY
from mimarsinan.config_schema.registry.relevance import Relevance as R
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
)
from mimarsinan.config_schema.registry.value_schema import RequiredIff, ValueSchema
from mimarsinan.mapping.support.period_census import RESIDENCY_STEP_RULES, STEP_INDEX

RESIDENCY_PLAN_SCHEMA = ValueSchema(
    kind=T.JSON,
    fields={
        "step": ValueSchema(kind=T.ENUM, options=RESIDENCY_STEP_RULES, decision_class=COST_ONLY,
                            doc="which residency step executes: passes and pinned set only"),
        "index": ValueSchema(kind=T.INT, low=0, decision_class=COST_ONLY,
                             doc="index of the executed step: passes and pinned set only"),
    },
    required=("step",),
    required_iff=(RequiredIff("index", "step", (STEP_INDEX,)),),
)

ENTRIES = (
    _E("residency_plan", section="platform_constraints", group="mapping_strategy",
       owner="mapping/support/schedule/residency_composition", type=T.JSON,
       category=Category.ADVANCED, exposure="user", label="Residency Plan",
       effect="Executes a planned residency step: pinned cores stay programmed across periods",
       doc="Selects the residency plan the scheduled build executes: floor runs the minimum-r "
           "step, stateless the stateless plan, index the step of that index. The plan pins "
           "cores across periods and rewrites the rest; outputs are those of the U1 schedule.",
       value_schema=RESIDENCY_PLAN_SCHEMA, passthrough=True,
       relevant=R.when_true("allow_scheduling"),
       empty_means="U1's pass planner, byte for byte"),
    _E("position_multiplexing", section="platform_constraints", group="mapping_strategy",
       owner="mapping/support/position_multiplexing", type=T.BOOL,
       category=Category.ADVANCED, exposure="user", label="Position Multiplexing",
       effect="One copy of each shared weight bank serves its output positions in turn, pass by pass",
       doc="Mapping decision: true holds one copy of every shared weight bank (a conv's kernel, a token-wise "
           "layer's weights) and serves its output positions in turn over passes on that copy, so the chip "
           "commits each weight once per run at the cost of passes, latency and carry; false or absent "
           "gives every position its own copy wherever the chip holds them. Outputs are unchanged.",
       passthrough=True, relevant=R.when_true("allow_scheduling"),
       legal_values=lambda cfg: _multiplexing_legal(cfg),
       empty_means="every position its own copy, the planner byte for byte"),
)

#: Declarations whose cores the shape-only plane cannot see: IR elimination compacts bank instances into owned cores
#: (no shared copy on the executed plane), and depth balancing inserts relay cores the layout walk does not record.
_UNSEEN_BY_THE_SHAPE_ONLY_PLANE = ("pruning", "lif_depth_balancing_relays")


def _multiplexing_legal(cfg):
    """Only false where a declaration changes the executed cores the shape-only plane plans: both planes refuse it.
    An operand core type (N2) is one: the shape-only plane keeps a product as a host op that the executed plane lowers
    onto operand cores, whose payload is rewritten every pass, so no run of passes serves one copy of it."""
    operand = any(isinstance(c, Mapping) and c.get("operand") is not None for c in cfg.get("cores") or ())
    return (False,) if operand or any(cfg.get(key) for key in _UNSEEN_BY_THE_SHAPE_ONLY_PLANE) else (False, True)
