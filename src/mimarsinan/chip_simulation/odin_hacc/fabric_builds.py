"""Which fabric a bundle's passes are built for — the ONE place that chooses.

The choice is made on the CLAIMS a bundle carries (``ChipConfig.bundle_claims``),
which is the same predicate the shipped reader dispatches its AER wording on. A
crossbar that signs a whole PRE-synaptic row is the vendored core and nothing
else; a synapse cell that signs itself is a generated one, and a generated core
only exists as a NAMED chip configuration — so those claims must resolve to one
rather than describing a geometry no cosimulation ever proved.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set, Tuple

from mimarsinan.chip_simulation.odin_fpga.chip_selection import chip_config_for
from mimarsinan.chip_simulation.odin_hacc.pass_build import PassBuild
from mimarsinan.chip_simulation.odin_hacc.variant_pass_build import VariantPassBuild
from mimarsinan.chip_simulation.odin_rtl.reference import CycleTrace
from mimarsinan.chip_simulation.soma_axes import PER_SYNAPSE_SIGN


class PassOrderError(ValueError):
    """This segment's cores admit no causal host-mediated pass schedule."""


def causal_core_order(mapping: Any) -> Tuple[int, ...]:
    """The order the passes must RUN in: every core after the cores it reads.

    A pass is host-mediated — the host reads one core's counts back and words
    them as the next core's stimulus — so a consumer scheduled before its
    producer would be stimulated with counts nobody measured, which is exactly
    what the shipped executor refuses. Index order is that order only by luck
    (it was, for every two-hop bundle before the three-hop cascade).

    Kahn's algorithm with the lowest ready index first, so the order is
    deterministic — a bundle's ``pass_order`` is part of its self-hash.
    """
    cores = list(mapping.cores)
    producers: Dict[int, Set[int]] = {}
    for index, core in enumerate(cores):
        reads: Set[int] = set()
        for slot, source in enumerate(core.axon_sources or ()):
            producer = int(getattr(source, "core_", -1))
            if producer < 0:
                continue
            if producer >= len(cores):
                raise PassOrderError(
                    f"core {index} slot {slot} reads core {producer}, which "
                    f"this segment does not carry ({len(cores)} core(s)). A "
                    f"pass schedule cannot name a producer that is not here.")
            if producer != index:
                reads.add(producer)
        producers[index] = reads
    order: List[int] = []
    placed: Set[int] = set()
    while len(order) < len(cores):
        ready = [i for i in range(len(cores))
                 if i not in placed and producers[i] <= placed]
        if not ready:
            stuck = sorted(set(range(len(cores))) - placed)
            raise PassOrderError(
                f"cores {stuck} sit on a cycle in the segment graph: each one "
                f"reads a core that has not run. A host-mediated pass schedule "
                f"is only defined on a DAG.")
        chosen = ready[0]
        order.append(chosen)
        placed.add(chosen)
    return tuple(order)


def pass_builds(mapping: Any, traces: Sequence[CycleTrace], *, weight_bits: int,
                effective_max_axons: int, weight_sign_granularity: str,
                soma_law: Any, membrane_init: int) -> List[Any]:
    """One build per core, on the fabric these claims identify, IN PASS ORDER."""
    order = causal_core_order(mapping)
    if str(weight_sign_granularity) == PER_SYNAPSE_SIGN:
        chip = chip_config_for(
            weight_bits=int(weight_bits),
            weight_sign_granularity=str(weight_sign_granularity),
            effective_max_axons=int(effective_max_axons),
            membrane_bits=int(soma_law.membrane_bits))
        return [
            VariantPassBuild(mapping, index, traces[0], spec=chip.core_spec,
                             membrane_init=int(membrane_init))
            for index in order
        ]
    return [
        PassBuild(mapping, index, traces[0], weight_bits=int(weight_bits),
                  effective_max_axons=int(effective_max_axons),
                  soma_law=soma_law,
                  weight_sign_granularity=str(weight_sign_granularity),
                  membrane_init=int(membrane_init))
        for index in order
    ]
