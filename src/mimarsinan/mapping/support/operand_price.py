"""The price of a lowering (N2 design §2.2), by the toolchain's own laws: no second write-energy formula.

A lowering writes its operand regions and streams its other operand. The writes are priced by G3's write law
(``price_write_terms``) over a one-pass census of the cells the packed program writes per inference (every streamed
instance holds its whole bank, ``producer_census.placed_operand_cells``), with the declared write source: the
host transfer only when the payloads come from the host. The streamed vectors are injected into the operand cores
at the constant the carry-inject term prices. The output elements are m x n per bank under either lowering, so they
never order the two and are not priced. [U-J4] Both read the rule of ``platform_physics.hosts.store``: beside the
host-chip link the payload's transfer is the link's and host memory's (the write law prices the cells only), and the
streamed vectors are injected at the carry's per-byte energy on the declared store (``injection_byte_energy``:
``e_dma_per_byte`` on the chip's own store or without the link, the link and host memory on a host-owned one).
[U-J4 targeted, review M1] Both lowerings place the same cells and move the same payload, so the streamed vectors
order them, and where both price equal the tie rule picks fewer bank cells, the lowering streaming more. So the
choice moves where the injection's per-byte energy on a host-owned store changed between zero and nonzero: a chip
without ``e_dma_per_byte`` beside the edge host now injects at the link and host memory and takes the lowering that
streams less (it tied before); beside a block declaring neither link energy row it ties (it took the lowering
streaming less before). A row the price reads and the physics does not declare is named in the lowering's printed
reason (``LoweringPrice.unpriced``, the headline's text ``hosts.store.unpriced_link_row``), never a silent 0.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple, cast

from mimarsinan.deployment_record.build.producer_census import placed_operand_cells
from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs, price_write_terms
from mimarsinan.deployment_record.cost.host_terms import write_link_bytes
from mimarsinan.deployment_record.platform_physics.hosts.store import (
    host_owned, injection_byte_energy, prices_host_link, store_byte_energy, undeclared_link_energy,
)
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.platform_physics.resolve import resolve_platform_physics, technology_physics
from mimarsinan.deployment_record.quantities.write_law import WritePolicy, write_policy_of
from mimarsinan.deployment_record.quantities.write_technology import operand_technology, technology_policy
from mimarsinan.mapping.support.operand_form import Lowering, LoweringPrice
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


def lowering_price(platform: Mapping[str, Any]) -> LoweringPrice:
    """Energy per inference of a lowering (J): G3's write energy of its operand regions ([U-J4] beside the host-chip
    link with the payload's transfer at the link and host memory) plus its streamed vectors at the carry's injection
    energy on the declared store; 0 where the platform declares neither (the choice then falls to its tie rule). A
    write law that refuses its energy (no commit energy declared) prices the writes at 0: under the packer's placement
    both lowerings write the same cells, so the refusal never orders them. On a chip of several write technologies
    the regions are priced under the operand core types' (N2b; two of them refuse by name). [U-J4 targeted] Its
    ``unpriced`` names each link energy row a lowering's bytes would pay and the physics does not declare: the
    host-sourced payload's beside the link, the streamed vectors' on a host-owned store."""
    physics = declared_physics(platform)
    if physics is None:
        return LoweringPrice(lambda lowering: 0.0)
    declared = platform if platform.get("inference_batch_K") is not None else {**platform, "inference_batch_K": 1}
    policy = cast(WritePolicy, write_policy_of(declared))  # a declared batch always gives a policy
    technology = operand_technology(policy)
    if technology is not None:
        physics = technology_physics(physics, technology)
        policy = technology_policy(policy, (technology,))
    value_bytes = float(platform.get("activation_bits") or 8) / 8.0
    injected = injection_byte_energy(physics)  # [U-J4] the carry's re-injection on the declared store
    inject = 0.0 if injected is None else float(injected.nominal)
    beside = prices_host_link(physics)
    crossed = store_byte_energy(physics, "a byte over the host-chip link") if beside else None
    link = 0.0 if crossed is None else float(crossed.nominal)
    payload_rows = undeclared_link_energy(physics, "the lowering's host-sourced payload over the host-chip link")
    stream_rows = undeclared_link_energy(physics, "the lowering's streamed vectors injected from the host-owned store")

    def moved(lowering: Lowering) -> float:
        crossing = write_link_bytes(WriteInputs(operand_region_census(lowering), policy))  # [U-J4] over the link
        return 0.0 if crossing is None else crossing.program

    def price(lowering: Lowering) -> float:
        priced = price_write_terms(operand_region_census(lowering), physics, policy)
        writes = 0.0 if priced.energy is None else float(priced.energy.nominal) * _MJ_TO_J
        return writes + moved(lowering) * link + lowering.streamed_elements * value_bytes * inject

    def unpriced(lowering: Lowering) -> Tuple[str, ...]:  # [U-J4 targeted] the rows its bytes read, undeclared
        return ((payload_rows if beside and moved(lowering) else ())
                + (stream_rows if host_owned(physics) and lowering.streamed_elements else ()))

    return LoweringPrice(price, unpriced)
