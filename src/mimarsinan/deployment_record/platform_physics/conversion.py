"""Conversion models: how many ADC conversions a target's own dataflow requires.

An analog in-memory accelerator's energy and area are dominated by data conversion,
but no deployment record counts conversions — the count is a function of the
TARGET's dataflow (array geometry, converter sharing, input bit-slicing), not of the
workload alone. So the target declares its model, and the model derives the
conversion quantities from the ones the record already carries.

A digital target declares nothing and gets zero, which is a FACT about it rather than
an absence: a digital chip has no converter, whereas a target that has not said
disables the objectives that would need one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue

Derivation = Callable[[Quantities, Mapping[str, float]], Dict[str, QuantityValue]]

MODEL_KEY = "model"
DIGITAL = "digital"
BIT_SLICED_CROSSBAR = "bit_sliced_crossbar"


def _derived(value: float) -> QuantityValue:
    """MODELED, not measured: a conversion count rests on the target's declared
    dataflow, and the provenance must carry that downstream."""
    return QuantityValue(value=float(value), provenance="modeled")


def _digital(
    quantities: Quantities, params: Mapping[str, float]
) -> Dict[str, QuantityValue]:
    """No converter exists, so the counts are zero — a fact, not an absence."""
    del quantities, params
    return {"adc_conversions": _derived(0.0), "adc_count": _derived(0.0)}


def _bit_sliced_crossbar(
    quantities: Quantities, params: Mapping[str, float]
) -> Dict[str, QuantityValue]:
    """The analog-IMC dataflow ISAAC and PRIME describe: weights bit-sliced across
    COLUMNS, inputs bit-sliced across TIME, every column converted every cycle.

    One logical weight occupies ``cells_per_weight`` adjacent columns, so one logical
    MAC touches that many cells; a length-``array_rows`` column integration therefore
    covers ``array_rows / cells_per_weight`` logical MACs. Each integration is then
    converted once per input slice (``input_bits / dac_bits`` cycles):

        adc_conversions = ceil(macs * cells_per_weight / array_rows)
                          * (input_bits / dac_bits)

    Dropping the column-slicing factor is the mistake worth naming: for ISAAC-CE
    (16-bit weights over 2-bit cells, 16 1-bit input slices, 128 rows) the formula
    gives exactly ONE 8-bit conversion per 16-bit MAC, and ignoring the slicing would
    have under-counted conversion — the dominant cost of an analog accelerator — by
    8x.

    A partial integration still costs a whole one, so the division rounds UP:
    rounding down would price a read the chip performs at nothing.

    Converters: one ADC per ``adc_sharing_factor`` columns, per array, over as many
    arrays as the declared cell capacity implies.
    """
    derived: Dict[str, QuantityValue] = {}
    if quantities.has("macs"):
        integrations = math.ceil(
            quantities.get("macs").value * params["cells_per_weight"]
            / params["array_rows"]
        )
        slices = params["input_bits"] / params["dac_bits"]
        derived["adc_conversions"] = _derived(integrations * slices)
    if quantities.has("cells_physical"):
        cells_per_array = params["array_rows"] * params["array_cols"]
        arrays = math.ceil(quantities.get("cells_physical").value / cells_per_array)
        per_array = math.ceil(params["array_cols"] / params["adc_sharing_factor"])
        derived["adc_count"] = _derived(arrays * per_array)
    return derived


@dataclass(frozen=True)
class ConversionModel:
    """One named dataflow, its required parameters, and its derivation."""

    name: str
    doc: str
    required: Tuple[str, ...]
    derive_fn: Derivation
    params: Mapping[str, float] = None  # type: ignore[assignment]

    def derive(self, quantities: Quantities) -> Dict[str, QuantityValue]:
        """The conversion quantities this target's dataflow implies."""
        return self.derive_fn(quantities, self.params or {})


_MODEL_SPECS: Tuple[Tuple[str, str, Tuple[str, ...], Derivation], ...] = (
    (
        DIGITAL,
        "A fully digital datapath: there is no analog-to-digital conversion, so the "
        "conversion counts are identically zero. The default for any target that "
        "declares no model.",
        (),
        _digital,
    ),
    (
        BIT_SLICED_CROSSBAR,
        "Weights bit-sliced across columns, inputs bit-sliced across time, every "
        "column converted every cycle: conversions = ceil(macs * cells_per_weight / "
        "array_rows) * (input_bits / dac_bits). The dataflow ISAAC and PRIME "
        "describe, and the reason an analog profile can price conversion at all.",
        ("array_rows", "array_cols", "adc_sharing_factor", "input_bits",
         "cells_per_weight", "dac_bits"),
        _bit_sliced_crossbar,
    ),
)

CONVERSION_MODELS: Mapping[str, ConversionModel] = {
    name: ConversionModel(name=name, doc=doc, required=required, derive_fn=fn)
    for name, doc, required, fn in _MODEL_SPECS
}


def conversion_model_for(
    declaration: Optional[Mapping[str, Any]]
) -> ConversionModel:
    """The model a profile declares, with its parameters validated at declaration.

    A missing or empty declaration is the DIGITAL model: most targets convert
    nothing, and saying so explicitly beats an absent count nobody can price.
    """
    declaration = dict(declaration or {})
    name = str(declaration.pop(MODEL_KEY, DIGITAL))
    try:
        model = CONVERSION_MODELS[name]
    except KeyError:
        raise KeyError(
            f"unknown conversion model {name!r}; the declared models are "
            f"{sorted(CONVERSION_MODELS)}"
        ) from None
    missing = [key for key in model.required if key not in declaration]
    if missing:
        raise ValueError(
            f"conversion model {name!r} requires {missing} — a dataflow it cannot "
            f"describe would produce a conversion count nobody can defend"
        )
    unknown = set(declaration) - set(model.required)
    if unknown:
        raise ValueError(
            f"conversion model {name!r} does not take {sorted(unknown)}; it takes "
            f"{list(model.required)}"
        )
    params = {key: float(declaration[key]) for key in model.required}
    for key, value in params.items():
        if value <= 0:
            raise ValueError(f"{name}.{key} must be positive, got {value}")
    return ConversionModel(
        name=model.name, doc=model.doc, required=model.required,
        derive_fn=model.derive_fn, params=params,
    )


__all__ = [
    "BIT_SLICED_CROSSBAR",
    "CONVERSION_MODELS",
    "DIGITAL",
    "ConversionModel",
    "conversion_model_for",
]
