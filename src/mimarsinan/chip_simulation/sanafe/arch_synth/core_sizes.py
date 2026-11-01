"""Each core of the simulated SANA-FE chip sized by the cores the program places on it [U-SF].

SANA-FE 2.1.1 gives each declared input slot of a core (``inputs[k]``, one soma unit per axon: the net builder binds
input axon ``a`` to ``inputs[a]``) its own hardware unit, about 9.3 KiB of chip and 1.5 KiB of architecture, and each
core a timestep buffer of ``max_neurons_supported`` entries of 56 B. Every executed unit (a neural stage, or each of
its retimed level stages) is loaded from core 0 of a fresh chip (``runner.neural_stage``), so core ``p`` of the chip
holds core ``p`` of each unit in turn: it is declared at the largest axons and neurons any unit places there, and a core
no unit places on at the smallest size, ``UNPLACED``. Sized at the program's widest core instead, C5's T-LIF chip (1,600
cores of 1,280 slots where its cores need 256 to 1,280) held 19 GiB.

The simulated function does not read the size: a slot no neuron is mapped to is never in use (the in-use units keep
their order), and ``max_neurons_supported`` is only the core's mapping capacity and its buffer's length (indexed by a
neuron's offset within the core; the mimarsinan plugins size their per-neuron state on demand).
"""

from __future__ import annotations

from typing import Any, List, Tuple

#: A core no executed unit places on: no axons, no neurons (it declares one input slot).
UNPLACED: Tuple[int, int] = (0, 0)


def executed_units(mapping: Any) -> List[Any]:
    """The hard core mappings of every unit the runner loads from core 0 of a fresh chip: each neural stage and each of
    its retimed level stages [U-TT T-1]."""
    return [unit.hard_core_mapping for stage in getattr(mapping, "stages", None) or []
            if getattr(stage, "kind", None) == "neural"
            for unit in [stage, *(getattr(stage, "retimed_level_stages", None) or [])]
            if getattr(unit, "hard_core_mapping", None) is not None]


def placed_core_sizes(mapping: Any) -> Tuple[Tuple[int, int], ...]:
    """Per core position: the largest (axons, neurons) any executed unit places there; empty without executed units
    (then every core is declared at the widest core)."""
    sizes: List[Tuple[int, int]] = []
    for hcm in executed_units(mapping):
        for position, core in enumerate(hcm.cores):
            axons, neurons = int(core.axons_per_core), int(core.neurons_per_core)
            if position == len(sizes):
                sizes.append((axons, neurons))
            else:
                sizes[position] = (max(sizes[position][0], axons), max(sizes[position][1], neurons))
    return tuple(sizes)


def declared_core_size(spec: Any, tile_idx: int, local_idx: int) -> Tuple[int, int]:
    """(axons, neurons) the chip declares for core ``local_idx`` of tile ``tile_idx``: the inverse of the net builder's
    sequential tile fill (``noc_geometry.tile_and_local_of_core``) read in ``spec.core_sizes``."""
    if not spec.core_sizes:
        return int(spec.axons_per_core), int(spec.neurons_per_core)
    position = tile_idx * max(int(spec.cores_per_tile_resolved), 1) + local_idx
    return spec.core_sizes[position] if position < len(spec.core_sizes) else UNPLACED


def input_slots(axons: int) -> int:
    """Input soma units a core of ``axons`` declares (at least one: SANA-FE refuses a core without a soma range)."""
    return max(1, int(axons))


def neuron_capacity(axons: int, neurons: int) -> int:
    """``max_neurons_supported`` of a core: twice what the net builder can map on it (its neurons, and one neuron per
    input slot: an input axon's, or the always-on neuron on an always-on axon's slot)."""
    return 2 * (int(neurons) + input_slots(axons))
