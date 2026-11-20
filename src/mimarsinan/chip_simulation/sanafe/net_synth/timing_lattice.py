"""The executor's lattice record stamped on SANA-FE's timing somas, so each decides on the integer program's value.

The quantized timing soma is handed its membrane by the stage (``apply_ttfs_preset_membranes``); the per-cycle soma
reconstructs it from its single-spike arrivals. Each decides through the one timing decision it shares with
nevresim (``timing_decision.hpp``), which decides a neuron in integers exactly where the Python executor did
(``membrane_lattice.lattice_membrane``, this sample): the stamp hands every used neuron the executor's ``lattice`` flag
and, to a soma handed a preset, the membrane's integer numerator over ``2S`` (``preset_numerator``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mimarsinan.chip_simulation.sanafe.arch_synth.spec import _plugin_path
from mimarsinan.chip_simulation.timing_decision import require_timing_decision_range, require_timing_decision_version
from mimarsinan.mapping.support.core_geometry import used_neurons as _used_neurons

#: The soma attributes of the record (the plugins register them).
LATTICE_ATTRIBUTE = "lattice"
NUMERATOR_ATTRIBUTE = "preset_numerator"

LatticeRecord = Optional[Tuple[np.ndarray, np.ndarray]]


def timing_soma_library(preset: bool) -> Optional[str]:
    """The toolchain's built plugin of the soma the stamp writes (every synthesized architecture names it): the
    quantized soma takes a preset membrane, the per-cycle soma reconstructs its own. None when not built (a custom
    architecture then names its own somas, read at its load)."""
    return _plugin_path("ttfs_quantized_soma" if preset else "ttfs_cycle_soma")


def stamp_timing_lattice(
    core_to_group: Dict[int, Any],
    hcm: Any,
    membrane_lattice: Optional[Sequence[LatticeRecord]],
    *,
    simulation_length: int,
    preset: bool,
) -> None:
    """Stamp the executor's lattice record of sample 0 on every used neuron of every timing core.

    ``membrane_lattice`` is the contract stage's per-core ``(on_lattice, m)`` (None for a core with no lattice
    neuron); ``preset`` marks a soma handed a preset membrane, which also takes the numerator ``m``. A program the
    executor decided on no lattice (a float program, or a continuous timing one) keeps its somas' float paths."""
    records: List[LatticeRecord] = list(membrane_lattice or [])
    if all(record is None for record in records):
        return
    require_timing_decision_range(hcm.cores, where=f"SANA-FE's timing soma synthesis at S {int(simulation_length)}")
    library = timing_soma_library(preset)
    if library is not None:
        require_timing_decision_version(library)  # loaded once: a stale soma would read the record as unknown
    for core_idx, group in core_to_group.items():
        n_used = _used_neurons(hcm.cores[core_idx])
        if n_used <= 0:
            continue
        record = records[core_idx] if core_idx < len(records) else None
        on_lattice = (np.zeros(n_used, dtype=bool) if record is None
                      else np.asarray(record[0], dtype=bool)[0, :n_used])
        numerator = (np.zeros(n_used, dtype=np.float64) if record is None
                     else np.asarray(record[1], dtype=np.float64)[0, :n_used])
        for n_idx in range(n_used):
            attrs: Dict[str, Any] = {LATTICE_ATTRIBUTE: bool(on_lattice[n_idx])}
            if preset:
                attrs[NUMERATOR_ATTRIBUTE] = float(numerator[n_idx]) if on_lattice[n_idx] else 0.0
            group[n_idx].set_attributes(model_attributes=attrs)
