"""Axes a paper publishes that the pricer expresses as a ratio of two priced terms.

Average power is what almost every chip paper prints on its front page, and it is
exactly energy per inference divided by the time that inference took. Deriving it here
rather than in the pricer keeps the four headline axes — and every registry, picker and
study table keyed on them — unchanged, while still letting a reference case be stated
in the units its source actually used.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Tuple

from mimarsinan.deployment_record.cost.absolute.formulas import div_band
from mimarsinan.deployment_record.cost.terms import CostTerm
from mimarsinan.deployment_record.schema.provenance import Band

DerivedValue = Tuple[Optional[float], Optional[Band], str]


def _average_power_mw(terms: Mapping[str, CostTerm]) -> DerivedValue:
    """mJ per inference / seconds per inference = mJ/s = mW."""
    energy, latency = terms.get("energy_per_inference_mj"), terms.get("e2e_latency_s")
    if energy is None or latency is None:
        missing = [n for n, t in (("energy_per_inference_mj", energy),
                                  ("e2e_latency_s", latency)) if t is None]
        return None, None, f"needs {' and '.join(missing)}"
    basis = "energy_per_inference_mj / e2e_latency_s"
    band = None
    if energy.band is not None and latency.band is not None and latency.band.low > 0:
        band = div_band(energy.band, latency.band, basis)
    return energy.value / latency.value, band, basis


#: axis name -> how it is computed from the priced terms.
DERIVED_AXES: Mapping[str, Callable[[Mapping[str, CostTerm]], DerivedValue]] = {
    "average_power_mw": _average_power_mw,
}
