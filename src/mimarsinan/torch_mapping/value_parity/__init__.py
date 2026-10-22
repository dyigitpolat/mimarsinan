"""Value-parity conversion probe (E270): converted model and identity value program against the native model."""

from mimarsinan.torch_mapping.value_parity.comparison import relative_error
from mimarsinan.torch_mapping.value_parity.contract import (
    NATIVE_EVAL_IDENTITY,
    FunctionContract,
)
from mimarsinan.torch_mapping.value_parity.probe import (
    WitnessConversion,
    probe_conversion,
    probe_conversion_values,
)
from mimarsinan.torch_mapping.value_parity.rounding import (
    EXACTNESS_ACCEPTANCE,
    conversion_value_bound,
    declared_rounding_bound,
    lowering_value_bound,
    weight_layer_depth,
)
from mimarsinan.torch_mapping.value_parity.verdict import (
    STAGE_CONVERTED,
    STAGE_NATIVE,
    STAGE_VALUE_PROGRAM,
    ConversionValueParityError,
    ValueParityResult,
)
from mimarsinan.torch_mapping.value_parity.witness import (
    fp64_snapshot,
    probe_inputs,
    randomize_batchnorm_statistics,
    randomize_zero_parameters,
)

__all__ = [
    "EXACTNESS_ACCEPTANCE",
    "NATIVE_EVAL_IDENTITY",
    "STAGE_CONVERTED",
    "STAGE_NATIVE",
    "STAGE_VALUE_PROGRAM",
    "ConversionValueParityError",
    "FunctionContract",
    "ValueParityResult",
    "WitnessConversion",
    "conversion_value_bound",
    "declared_rounding_bound",
    "lowering_value_bound",
    "fp64_snapshot",
    "probe_conversion",
    "probe_conversion_values",
    "probe_inputs",
    "randomize_batchnorm_statistics",
    "randomize_zero_parameters",
    "relative_error",
    "weight_layer_depth",
]
