"""Registry entries: the declared value grids of Wave 2 (WI-5a quantizer state, WI-5b scale rule, WI-5c bias)."""

from __future__ import annotations

from mimarsinan.config_schema.registry.decision_classes import FUNCTION_BEARING
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
)
from mimarsinan.config_schema.registry.value_schema import ValueSchema
from mimarsinan.mapping.support.grid_declarations import (
    BIAS_RESOLUTIONS, WEIGHT_SCALE_GRANULARITIES, WEIGHT_SCALE_GRANULARITY_KEY, WEIGHT_SCALE_RULES,
)

QUANTIZER_STATE_SCHEMA = ValueSchema(
    kind=T.JSON,
    fields={
        "file": ValueSchema(kind=T.STR, min_length=1, decision_class=FUNCTION_BEARING,
                            doc="the map of declared grids and ranges: moves every grid it names"),
        "sha256": ValueSchema(kind=T.STR, min_length=64, decision_class=FUNCTION_BEARING,
                              doc="the map's digest: a map of other bytes is refused"),
    },
    required=("file", "sha256"),
)

_OWNER = "mapping/support/quantizer_state"
ENTRIES = (
    _E("quantizer_state", domain="value", section="platform_constraints", group="hardware", owner=_OWNER,
       type=T.JSON, category=Category.ADVANCED, exposure="user", label="Declared Quantizer State",
       effect="Declares readout ranges and entry grids (scale, offset, rounding, codes) per FX node",
       doc="A sha-pinned JSON map keyed by FX node name: per perceptron a readout range and, at a segment "
           "entry, its boundary grid's scale, offset, rounding (nearest or floor) and code range; the named "
           "grids replace the calibrated ones, the others stay calibrated.",
       value_schema=QUANTIZER_STATE_SCHEMA, passthrough=True,
       empty_means="every boundary scale and readout range calibrated (or the fixed range_value), byte for byte"),
    _E("weight_scale_rule", domain="value", section="platform_constraints", group="hardware",
       owner="transformations/normalization_aware_perceptron_quantization", type=T.ENUM,
       options=WEIGHT_SCALE_RULES, category=Category.ADVANCED, exposure="user", label="Weight Scale Rule",
       effect="The weight grid's scale: the integer threshold lattice, or q_max / max|w|",
       doc="integer: max(1, floor(q_max / max|w|)), the integer threshold register of a spiking chip; "
           "fractional: q_max / max|w| for a value-domain program, whose readout law carries the scale in "
           "its multiplier (or whose float executor divides by it).",
       passthrough=True, empty_means="integer — the integer-lattice scale, byte for byte"),
    _E("bias_resolution", domain="value", section="platform_constraints", group="hardware",
       owner="mapping/export/chip_quantize", type=T.ENUM, options=BIAS_RESOLUTIONS,
       category=Category.ADVANCED, exposure="user", label="Bias Resolution",
       effect="Where the bias is held: on the weight grid, or at the readout law's resolution",
       doc="weight_grid: the bias quantized on the weight grid (its own integer-ratio grid under two-scale); "
           "law: the bias stays real through Weight Quantization and the readout law rounds it into B at "
           "2^-s output steps (float on the float path); needs an on-chip bias.",
       passthrough=True, empty_means="weight_grid — the bias on the weight grid, byte for byte"),
    _E(WEIGHT_SCALE_GRANULARITY_KEY, domain="event", section="platform_constraints", group="hardware",
       owner="mapping/support/neuron_thresholds", type=T.ENUM, options=WEIGHT_SCALE_GRANULARITIES,
       category=Category.ADVANCED, exposure="user", label="Weight Scale Granularity",
       effect="The weight grid's scale: one per perceptron, or one per output neuron (its own threshold register)",
       doc="per_perceptron: one scale per perceptron, the integer threshold register its cores share; per_neuron: "
           "each output neuron on its own scale s_c = max(1, floor(q_max / max_j |w_cj|)) with threshold theta_c = "
           "s_c and a membrane lattice per neuron, for a chip whose cores hold a threshold register per neuron "
           "(core_value_granularity threshold per_neuron); refused by name on a chip with one register per core.",
       passthrough=True, empty_means="per_perceptron, one scale per perceptron, byte for byte"),
)
