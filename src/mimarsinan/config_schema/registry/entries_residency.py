"""Registry entries: the executed residency plan selection."""

from __future__ import annotations

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
)
