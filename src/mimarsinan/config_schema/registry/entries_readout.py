"""Registry entries: the value-domain readout law (ADC on the accumulator, output requantization)."""

from __future__ import annotations

from mimarsinan.config_schema.registry.decision_classes import FUNCTION_BEARING
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
)
from mimarsinan.config_schema.registry.value_schema import RequiredIff, ValueSchema
from mimarsinan.mapping.support.period_census import READOUT_RANGE_RULES
from mimarsinan.mapping.support.grid_declarations import OUTPUT_LAWS

#: ``readout_law.range``: the ADC range from calibration, or the declared ``range_value``.
READOUT_RANGE_MODES = ("calibrated", "fixed")

READOUT_LAW_SCHEMA = ValueSchema(
    kind=T.JSON,
    fields={
        "adc_bits": ValueSchema(kind=T.INT, low=2, decision_class=FUNCTION_BEARING,
                                doc="ADC resolution on the accumulator: clips and rounds outputs"),
        "range": ValueSchema(kind=T.ENUM, options=READOUT_RANGE_MODES,
                             decision_class=FUNCTION_BEARING,
                             doc="how the ADC range is set: moves the output grid"),
        "range_value": ValueSchema(kind=T.FLOAT, low=0.0, low_exclusive=True,
                                   decision_class=FUNCTION_BEARING,
                                   doc="the declared ADC range: moves the output grid"),
        "mult_bits": ValueSchema(kind=T.INT, low=2, decision_class=FUNCTION_BEARING,
                                 doc="requantization multiplier width: rounds outputs"),
        "output_law": ValueSchema(kind=T.ENUM, options=OUTPUT_LAWS, decision_class=FUNCTION_BEARING,
                                  doc="output cores requantize or write exact integers: moves the logits",
                                  absent_means="requantize — every core writes the activation grid"),
    },
    required=("adc_bits", "range", "mult_bits"),
    required_iff=(RequiredIff("range_value", "range", ("fixed",)),),
)

ENTRIES = (
    _E("readout_law", domain="value", section="platform_constraints", group="hardware",
       owner="mapping/ir/core_law", type=T.JSON, category=Category.ADVANCED, exposure="user",
       label="Readout Law",
       effect="Declares the ADC readout and output requantization inside the certified program",
       doc="The value-domain readout law of every core: ADC bits on the integer accumulator "
           "(identity when at least the accumulator width), a calibrated or fixed range, and "
           "the requantization multiplier width. activation_bits stays the input (DAC) grid.",
       value_schema=READOUT_LAW_SCHEMA, passthrough=True,
       empty_means="no readout law — the float executor, byte for byte"),
    _E("readout_ranges", domain="value", section="platform_constraints", group="hardware",
       owner="mapping/export/law_grids", type=T.ENUM, options=READOUT_RANGE_RULES,
       category=Category.ADVANCED, exposure="user", label="Readout Ranges",
       effect="One output range per perceptron, or one per consumer group of its columns",
       doc="How the readout law's output ranges are calibrated: per perceptron (G2's law), or per group of "
           "a perceptron's columns that its readers read whole (N2: the K and V of each head leave the qkv "
           "projection on their own payload grids). Groups apply only to the perceptrons an operand product "
           "reads, and groups an operand bank writes take its payload_bits; without an operand core type no "
           "perceptron is grouped and the law is per_perceptron's.",
       passthrough=True,
       empty_means="per_perceptron — one output range per perceptron, G2's law byte for byte"),
)
