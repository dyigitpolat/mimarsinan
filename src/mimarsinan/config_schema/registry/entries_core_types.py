"""Registry entry: the layer-to-type rule of a chip whose core types carry their own weight width (WI-5g)."""

from __future__ import annotations

from mimarsinan.config_schema.registry.decision_classes import FUNCTION_BEARING
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
)
from mimarsinan.config_schema.registry.value_schema import ValueSchema
from mimarsinan.mapping.support.core_types import LAYER_RULE_KEY

LAYER_RULE_SCHEMA = ValueSchema(
    kind=T.JSON, min_length=1,
    item=ValueSchema(
        kind=T.JSON,
        fields={
            "layers": ValueSchema(kind=T.JSON, min_length=1, item=ValueSchema(kind=T.STR, min_length=1),
                                  decision_class=FUNCTION_BEARING,
                                  doc="FX-name globs of the perceptrons (and pool lanes) the rule places"),
            "weight_bits": ValueSchema(kind=T.INT, low=2, high=32, decision_class=FUNCTION_BEARING,
                                       doc="the width of the core types the selected layers run on: their grid"),
        },
        required=("layers", "weight_bits"),
    ),
)

ENTRIES = (
    _E(LAYER_RULE_KEY, domain="value", section="platform_constraints", group="hardware",
       owner="mapping/support/core_types", type=T.JSON, category=Category.ADVANCED, exposure="user",
       label="Layer-to-Core-Type Rule",
       effect="Which core type (by its weight width) each layer runs on, so each layer quantizes on its type's grid",
       doc="A list of {layers: [FX-name glob, ...], weight_bits: b}: the perceptrons (and pool lanes) a rule's globs "
           "select run on the core types of width b, quantize their weights on its grid and pack only onto those "
           "types' cores; every other layer runs on the types of the platform weight_bits. A core type declares "
           "its width in cores[i].weight_bits. Built for value-domain cores with a weight grid.",
       value_schema=LAYER_RULE_SCHEMA, passthrough=True,
       empty_means="every layer at the platform weight_bits, unbound to a core type, byte for byte"),
)
