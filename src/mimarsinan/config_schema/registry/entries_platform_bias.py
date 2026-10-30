"""Registry entries: how a parameter-encoded bias rides the crossbar."""

from __future__ import annotations

from mimarsinan.config_schema.registry.relevance import Relevance as R
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
)

_PC = "platform_constraints"

# Literals, not an import: config_schema.registry is pulled in from inside
# mimarsinan.mapping's own import, so reaching back into the mapping SSOT
# (mapping/support/bias_rows.py) here would close an import cycle. The drift
# guard is a test that asserts these against that SSOT.
BIAS_ROW_SPLITTING_MODE_KEY = "bias_row_splitting"
BIAS_ROWS_KEY = "bias_rows_per_perceptron"
BIAS_ROW_SPLITTING_MODES = ("off", "auto", "fixed")

ENTRIES = (
    _E(BIAS_ROW_SPLITTING_MODE_KEY, section=_PC, group="hardware",
       owner="mapping/bias", type=T.ENUM, category=Category.BASIC,
       exposure="user", label="Bias Row Splitting",
       effect="Frees the weight-quantization grid from the bias, at k axon rows",
       options=BIAS_ROW_SPLITTING_MODES,
       doc="How a bias rides a core with NO on-chip bias lane. 'off' keeps one "
           "always-on row on the shared max(|w|,|b|) grid — a dominant bias then "
           "sets the grid and the weights collapse toward zero. 'auto' splits the "
           "bias across the computed bound k = ceil(max|b| * s_w / q_max) rows, "
           "freeing the grid to come from max|w| alone; 'fixed' pins k. The k "
           "rows' integer weights sum EXACTLY to the deployed bias. Platforms "
           "with an on-chip bias register have no row to split (capability gate)."),
    _E(BIAS_ROWS_KEY, section=_PC, group="hardware", owner="mapping/bias",
       type=T.INT, category=Category.ADVANCED, exposure="user",
       label="Bias Rows Per Perceptron",
       doc="Declared always-on row count per perceptron under "
           "bias_row_splitting='fixed' (0 = undeclared, which that mode "
           "refuses); the projection refuses a k the computed bound exceeds.",
       bounds=(0, None),
       relevant=R.when(BIAS_ROW_SPLITTING_MODE_KEY, in_=("fixed",))),
)
