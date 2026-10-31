"""The region of a declared SANA-FE floorplan a program occupies: what the simulated chip defines [U-TT T-1].

Every executed neural unit (a neural stage, or each of its retimed level stages) is loaded from core 0 of a fresh chip
(``runner.neural_stage``), so a chip holds at most the widest unit's cores, ``ArchSpec.placed_cores``. Sequential tile
fill puts them on tiles ``0 .. ceil(placed / cores per tile) - 1`` and column-major placement
(``noc_geometry.xy_of_tile``) puts those in the first columns of the declared mesh. The chip defines that region only:
whole columns of the declared mesh height (one partial column when the placed tiles fit in one), each tile at its
declared coordinates with its declared cores. An XY route between two placed tiles stays inside the rectangle they
span, so every message walks the tiles it walks on the declared floorplan. The declared floorplan stays the record's
(name, mesh, tile coordinates), and the declared capacity stays the admission bound (``derive_arch_spec``). The chip's
memory follows the program: a discovery budget declaring 200,000 cores built a 200,000-core chip before (about 7.6 MiB
of chip per declared core).
"""

from __future__ import annotations

from dataclasses import replace

from mimarsinan.chip_simulation.sanafe.arch_synth.spec import ArchSpec


def placed_region(spec: ArchSpec) -> ArchSpec:
    """The floorplan region holding ``spec.placed_cores`` (``spec`` itself where they fill it, or none is named)."""
    placed = int(spec.placed_cores)
    needed = -(-placed // max(int(spec.cores_per_tile_resolved), 1))
    if placed <= 0 or needed >= spec.n_tiles:
        return spec
    height = min(max(int(spec.mesh_height), 1), needed)
    width = -(-needed // height)
    tiles = width * height
    return replace(spec, n_tiles=tiles, n_cores_per_tile=list(spec.n_cores_per_tile[:tiles]),
                   mesh_width=width, mesh_height=height)
