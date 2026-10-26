"""[E6, WI-5g] Typed cell widths (condition ``H8``, rule ``cell_width``): a core bound to a core type of its own weight
width (``weight_bits``, set by the packers from the declared type) holds only codes of that type's grid on its data
rows (``quantization_bounds``; the bias rows are on the bias grid, as ``core_types.check_core_codes`` reads them). The
declared cell is the narrowest executor of the core: a code outside its grid is no weight the chip stores, while every
simulator computes with it."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from mimarsinan.certification.program_check.report import ConditionResult, Violation
from mimarsinan.certification.program_check.view import K_ON, OFF, ProgramView
from mimarsinan.transformations.quantization_bounds import CORE_WEIGHT_BITS, quantization_bounds


def check_cell_widths(views: Dict[str, ProgramView], h8: ConditionResult, counts: Dict[str, Any]) -> None:
    """``cell_width`` on every typed core of the deployment and of the identity program."""
    typed = 0
    for label, view in views.items():
        for unit in view.units:
            for h, cv in enumerate(unit.cores):
                bits = getattr(cv.core, CORE_WEIGHT_BITS, None)
                if bits is None:
                    continue
                typed += 1
                h8.checked += 1
                data = (cv.codes != OFF) & (cv.kinds != K_ON)
                weights = np.asarray(cv.weights(), dtype=np.float64)[:len(data)][data]
                if not weights.size:
                    continue
                q_min, q_max = quantization_bounds(int(bits))
                low, high = float(weights.min()), float(weights.max())
                if low < q_min or high > q_max:
                    h8.add(Violation("H8", "cell_width", f"{label} unit {unit.index} ({unit.stage.name}) core {h}",
                                     f"a {int(bits)}-bit core holds weight codes in [{low:g}, {high:g}]; its type's "
                                     f"grid is [{q_min}, {q_max}]"))
    counts["typed_cores"] = typed
