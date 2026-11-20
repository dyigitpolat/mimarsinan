"""[U-J] The candidate's host-chip boundary as quantities, where its resolved physics prices the host-chip link.

The walk's census (``mapping/support/host_boundary``: crossing elements per direction, transfers at segment
boundaries, the MAC-free host operators' elements) sized at the run's carried-value grid, the pass carry's: on a
value-domain chip ``activation_bits`` per value (``value_domain.value_bytes``), on a spiking chip the dense window
(``pass_carry.boundary_transfer_bytes``); the pass boundaries that carry values across the boundary add one transfer
per direction each (``ProgramPasses.boundary_widths``; [targeted M2] under a declared on-chip carry buffer net of the
widths a wavefront group holds, ``crossing_widths``, the split ``value_domain.onchip_carry`` makes of the bytes).
Formed only where the physics declares the link or host memory (an edge host block,
``cost.host_terms.prices_host_link``), so every other candidate's quantities stay as they were; a part the candidate
cannot size is withheld by name, so the term that reads it names it unpriced. [U-J3] A carried pass boundary crosses
the link only to a store the chip does not own (``platform_physics.hosts.store``): on a chip-owned store the carry
stays with the chip's own DRAM, so only the segment boundaries transfer."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Mapping, Optional, Tuple

from mimarsinan.deployment_record.cost.host_terms import MEMORY_BOUND, TRANSFERS, prices_host_link
from mimarsinan.deployment_record.platform_physics.hosts.store import CHIP, owner
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.quantities.batch_rounds import rounds_per_inference
from mimarsinan.deployment_record.quantities.spec import QuantityValue
from mimarsinan.deployment_record.quantities.value_domain import (
    declared_buffer_bytes, names_value_domain, value_bits, value_bytes,
)
from mimarsinan.mapping.support.host_boundary import HOST_VALUE_BYTES
from mimarsinan.mapping.support.schedule.pass_carry import boundary_transfer_bytes

NO_GRID = "the run declares no carried-value grid (activation_bits, or a spiking window): crossing values have no size"
NO_CENSUS = "the candidate's walk states no host-chip boundary census (no wire census)"
NO_PASSES = "the candidate states no pass rows: its carried pass boundaries are not counted"
NO_SPLIT = ("the candidate's passes state no on-chip split of their boundaries, so under the declared "
            "carry_buffer_bytes the boundaries its carried values cross are not counted")


def _bytes_of(pcfg: Mapping[str, Any], timesteps: Optional[int]):
    if names_value_domain(pcfg):
        bits = value_bits(pcfg)
        return None if bits is None else (lambda n: value_bytes(int(n), bits))
    return None if timesteps is None else (lambda n: boundary_transfer_bytes(width=int(n), timesteps=int(timesteps)))


def crossing_widths(passes: Any, pcfg: Mapping[str, Any]) -> Optional[Tuple[Any, ...]]:
    """Per pass boundary, the carried widths that cross the chip boundary: every carried width without a declared
    ``carry_buffer_bytes``; with one, net of a wavefront group's own (``onchip_boundary_widths``); None where the
    passes state no such split."""
    if declared_buffer_bytes(pcfg) is None:
        return tuple(passes.boundary_widths)
    if passes.onchip_boundary_widths is None:
        return None
    held = {after: (out, inbound) for after, out, inbound in passes.onchip_boundary_widths}
    return tuple((after, out - held.get(after, (0, 0))[0], inbound - held.get(after, (0, 0))[1])
                 for after, out, inbound in passes.boundary_widths)


def host_link_context(context: Any, pcfg: Mapping[str, Any], census: Any, passes: Any) -> Any:
    """``context`` with the host-chip boundary's quantities (``host_link_quantities``, ``host_link_withheld``)
    where the resolved physics prices the host-chip link; the context itself elsewhere."""
    payload = pcfg.get("platform_physics_resolved")
    physics = PlatformPhysics.from_dict(payload) if payload else None
    if not prices_host_link(physics):
        return context
    values: Dict[str, float] = {}
    withheld: Dict[str, str] = {}
    keys = ("segment_in_bytes", "segment_out_bytes", TRANSFERS, MEMORY_BOUND)
    if census is None:
        withheld.update(dict.fromkeys(keys, NO_CENSUS))
    else:
        size = _bytes_of(pcfg, context.timesteps)
        for key, elements in (("segment_in_bytes", census.to_chip_elements),
                              ("segment_out_bytes", census.from_chip_elements)):
            if size is not None:
                values[key] = float(size(elements))
            elif elements:
                withheld[key] = NO_GRID
            else:
                values[key] = 0.0
        crossing = () if owner(physics) == CHIP else None if passes is None else crossing_widths(passes, pcfg)
        if crossing is None:
            withheld[TRANSFERS] = NO_PASSES if passes is None else NO_SPLIT
        else:
            carried = sum(int(bool(out)) + int(bool(inbound)) for _after, out, inbound in crossing)
            factor, refusal = rounds_per_inference(pcfg)  # [thru:w3] a run's carry crossings per inference
            if refusal is not None:
                withheld[TRANSFERS] = refusal
            else:
                values[TRANSFERS] = float(census.transfers + (carried if factor is None else carried * factor))
        values[MEMORY_BOUND] = float(census.memory_bound_elements * HOST_VALUE_BYTES)
    return replace(context, host_link_quantities=tuple(sorted(
        (key, QuantityValue(value, "static")) for key, value in values.items())),
        host_link_withheld=tuple(sorted(withheld.items())))
