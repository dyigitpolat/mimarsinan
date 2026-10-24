"""The price of a lowering (N2 design §2.2), by the toolchain's own laws: no second write-energy formula.

A lowering writes its operand regions and streams its other operand. The writes are priced by G3's write law
(``price_write_terms``) over a one-pass census of the cells the packed program writes per inference (every streamed
instance holds its whole bank, ``producer_census.placed_operand_cells``), with the declared write source: the
host transfer only when the payloads come from the host. The streamed vectors are injected into the operand cores
at ``e_dma_per_byte``, the constant the carry-inject term prices. The output elements are m x n per bank under
either lowering, so they never order the two and are not priced.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, cast

from mimarsinan.deployment_record.build.producer_census import placed_operand_cells
from mimarsinan.deployment_record.cost.absolute.writes import price_write_terms
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.platform_physics.resolve import resolve_platform_physics, technology_physics
from mimarsinan.deployment_record.quantities.write_law import WritePolicy, write_policy_of
from mimarsinan.deployment_record.quantities.write_technology import operand_technology, technology_policy
from mimarsinan.mapping.support.operand_form import Lowering
from mimarsinan.mapping.support.period_census import PassCensus, PassPeriodCensus

_MJ_TO_J = 1e-3


def declared_physics(platform: Mapping[str, Any]) -> Optional[PlatformPhysics]:
    """The platform's physics: the resolved constants a pipeline's platform carries, else its profile and
    overrides as declared; None when it declares none."""
    resolved = platform.get("platform_physics_resolved")
    if resolved:
        return PlatformPhysics.from_dict(resolved)
    return resolve_platform_physics(platform.get("platform_physics_profile"),
                                    platform.get("platform_physics_overrides"))


def operand_region_census(lowering: Lowering) -> PassPeriodCensus:
    """One pass holding the lowering's operand regions: the cells it writes per inference (producer cells)."""
    cells = placed_operand_cells(lowering.form.banks, lowering.instances, lowering.axons * lowering.neurons)
    return PassPeriodCensus(passes=(PassCensus(
        position=0, segment_index=0, resident=False, cores=1, resident_cores=0, written_cells_backing_store=0,
        written_cells_producer=cells, payload_bytes=0, compute_steps=0),), label="one lowering's operand regions")


def lowering_price(platform: Mapping[str, Any]) -> Callable[[Lowering], float]:
    """Energy per inference of a lowering (J): G3's write energy of its operand regions plus its streamed vectors
    at ``e_dma_per_byte``; 0 where the platform declares neither (the choice then falls to its tie rule). A write
    law that refuses its energy (no commit energy declared) prices the writes at 0: under the packer's placement
    both lowerings write the same cells, so the refusal never orders them. On a chip of several write technologies
    the regions are priced under the operand core types' (N2b; two of them refuse by name)."""
    physics = declared_physics(platform)
    if physics is None:
        return lambda lowering: 0.0
    declared = platform if platform.get("inference_batch_K") is not None else {**platform, "inference_batch_K": 1}
    policy = cast(WritePolicy, write_policy_of(declared))  # a declared batch always gives a policy
    technology = operand_technology(policy)
    if technology is not None:
        physics = technology_physics(physics, technology)
        policy = technology_policy(policy, (technology,))
    value_bytes = float(platform.get("activation_bits") or 8) / 8.0
    inject = float(physics.band("e_dma_per_byte").nominal) if physics.has("e_dma_per_byte") else 0.0

    def price(lowering: Lowering) -> float:
        priced = price_write_terms(operand_region_census(lowering), physics, policy)
        writes = 0.0 if priced.energy is None else float(priced.energy.nominal) * _MJ_TO_J
        return writes + lowering.streamed_elements * value_bytes * inject

    return price
