"""Registry entry: the on-chip arithmetic lanes of a value-domain chip (WI-5e, the lane subset of N3)."""

from __future__ import annotations

from mimarsinan.config_schema.registry.decision_classes import COST_ONLY, FUNCTION_BEARING
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
)
from mimarsinan.config_schema.registry.value_schema import ValueSchema
from mimarsinan.mapping.lanes.declaration import BUILT_LANE_CLASSES, LANE_KEY, RESERVED_LANE_CLASSES

ONCHIP_ARITHMETIC_SCHEMA = ValueSchema(
    kind=T.JSON,
    fields={
        "classes": ValueSchema(
            kind=T.JSON, item=ValueSchema(kind=T.ENUM, options=BUILT_LANE_CLASSES, reserved=RESERVED_LANE_CLASSES),
            decision_class=FUNCTION_BEARING,
            doc="the lane classes the cores execute (L clamp, E fused add, P pool): moves rounding and placement",
        ),
        "lanes_per_core": ValueSchema(kind=T.INT, low=1, decision_class=COST_ONLY,
                                      doc="output lanes per core of the lane block: prices its area and time",
                                      absent_means="the lane block's area and time are named unpriced"),
        "lane_block": ValueSchema(kind=T.STR, min_length=1, decision_class=COST_ONLY,
                                  doc="the shipped lane block (deployment_record/platform_physics/lane_blocks) whose "
                                      "area, energy and time per lane the chip prices: moves cost only",
                                  absent_means="no lane constants: the lane terms are named unpriced"),
    },
    required=("classes",),
)

ENTRIES = (
    _E(LANE_KEY, domain="value", section="platform_constraints", group="hardware", owner="mapping/lanes/declaration",
       type=T.JSON, category=Category.ADVANCED, exposure="user", label="On-Chip Arithmetic Lanes",
       effect="Which host ops the value-domain cores execute as lanes of the integer law",
       doc="classes: L runs ReLU as the law's clamp on an unsigned 2^b-code grid; E fuses a two-tensor add into "
           "its producing core (one operand row per column, one rounding for the sum); P runs an average pool as "
           "a requantizing reduction core. Lanes run on the integer path (readout_law); a float value program keeps "
           "its host ops. classes [] declares a chip without lanes (the off ablation). lanes_per_core and "
           "lane_block price the lane block (W08).",
       value_schema=ONCHIP_ARITHMETIC_SCHEMA, passthrough=True,
       empty_means="no lanes: every activation, add and pool on the host (the MVM contract), byte for byte"),
)
