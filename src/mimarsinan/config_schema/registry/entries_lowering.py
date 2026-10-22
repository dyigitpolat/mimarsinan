"""Registry entries: lowering choices of the converter (leaf-module decomposition)."""

from __future__ import annotations

from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
)

ENTRIES = (
    _E("decompose_leaf_modules", group="mapping_strategy",
       owner="torch_mapping/leaf_decomposition", type=T.BOOL, category=Category.ADVANCED,
       exposure="user", label="Decompose Leaf Modules",
       effect="Lowers attention and recurrent leaf modules into core-mappable projections",
       doc="Whether the converter decomposes the leaf modules the tracer keeps whole "
           "(multi-head attention, transformer encoder layers, LSTM, GRU, RNN) into their "
           "projections and host operations, so the projections map onto cores. Absent or "
           "false keeps every such leaf one host op, the IR byte for byte.",
       empty_means="false — every leaf module stays one host op"),
)
