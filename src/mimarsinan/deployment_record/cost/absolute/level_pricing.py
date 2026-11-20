"""A capability level priced on a declared chip (CL4): the level view's quantities into ``price_absolute`` and its
period census into the write law (``price_write_terms``) at the run's declared batch, through one call."""

from __future__ import annotations

from mimarsinan.deployment_record.cost.absolute.context import AbsolutePricing
from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
from mimarsinan.deployment_record.cost.absolute.validity import declared_geometry
from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.quantities.from_level import LevelView


def price_level(view: LevelView, physics: PlatformPhysics) -> AbsolutePricing:
    """Every term the physics backs for the level on its chip, judged at the chip's declared core type; where the
    chip declares ``inference_batch_K`` the period census's write energy, exposed write time and lifetime join the
    headlines, as on the record and candidate planes. [U-ENERGY] The level's passes are its load censuses: the
    deployment's first load and the reprogramming per round are priced by the one load law, as on the other planes."""
    writes = None if view.writes is None else WriteInputs.of(view.writes)
    return price_absolute(view.quantities, physics, writes=writes, geometry=declared_geometry(view.chip["cores"]),
                          loads=view.loads)
