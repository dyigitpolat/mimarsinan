"""SANA-FE mesh geometry conventions — the one home for every NoC reader.

Three conventions decide every NoC number the record seals and the candidate
estimator models: core -> (tile, local core) assignment, tile -> (x, y)
placement on the mesh, and the XY route walk. ``arch_synth`` (YAML floorplan),
``net_synth`` (core placement), the runner (geometry records), the trace
analysis (link loads), and the candidate NoC estimator all answer through
these functions, so the model and the measurement cannot disagree on geometry.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

#: One directed mesh edge: ``(from_x, from_y, to_x, to_y)``.
Edge = Tuple[int, int, int, int]


def tile_and_local_of_core(
    core_global_idx: int, cores_per_tile: int,
) -> Tuple[int, int]:
    """Sequential tile fill: core ``i`` sits on tile ``i // cores_per_tile``.

    Non-positive ``cores_per_tile`` means a single tile holds everything.
    """
    idx = int(core_global_idx)
    cpt = int(cores_per_tile)
    if cpt <= 0:
        return 0, idx
    return idx // cpt, idx % cpt


def xy_of_tile(tile_idx: int, mesh_height: int) -> Tuple[int, int]:
    """Column-major placement: ``(x, y) = (i // height, i % height)``."""
    mh = max(1, int(mesh_height))
    return int(tile_idx) // mh, int(tile_idx) % mh


def xy_route_edges(
    src_xy: Sequence[int], dst_xy: Sequence[int],
) -> List[Edge]:
    """Mesh edges one message traverses under XY routing (x first, then y)."""
    sx, sy = int(src_xy[0]), int(src_xy[1])
    dx, dy = int(dst_xy[0]), int(dst_xy[1])
    edges: List[Edge] = []
    cx, cy = sx, sy
    step_x = 1 if dx > sx else -1 if dx < sx else 0
    step_y = 1 if dy > sy else -1 if dy < sy else 0
    while cx != dx:
        nx = cx + step_x
        edges.append((cx, cy, nx, cy))
        cx = nx
    while cy != dy:
        ny = cy + step_y
        edges.append((cx, cy, cx, ny))
        cy = ny
    return edges


def xy_route_hops(src_xy: Sequence[int], dst_xy: Sequence[int]) -> int:
    """Manhattan distance — the length of the XY walk without building it."""
    return abs(int(dst_xy[0]) - int(src_xy[0])) + abs(
        int(dst_xy[1]) - int(src_xy[1])
    )


def most_square_exact_dims(n_tiles: int) -> Tuple[int, int]:
    """Most-square exact factorization ``(width >= height, width*height == n)``.

    Exact, never padded: phantom tiles the YAML does not define make
    SANA-FE's C++ NoC SIGFPE indexing them.
    """
    n = max(1, int(n_tiles))
    height = 1
    for h in range(int(math.isqrt(n)), 0, -1):
        if n % h == 0:
            height = h
            break
    return n // height, height


def replicated_mesh(
    *, packed_cores: int, cores_per_tile: int, rows: int, cols: int,
) -> Tuple[int, int, int]:
    """Declared-floorplan mesh: replicate ROWS until the pack fits.

    Returns ``(n_tiles, mesh_width, mesh_height)`` — the declared grid held
    fixed in width, extended in height by whole replicas (the multi-segment
    packing rule of ``derive_arch_spec``).
    """
    slots = int(rows) * int(cols) * int(cores_per_tile)
    replicas = max(1, -(-int(packed_cores) // max(1, slots)))
    total_rows = int(rows) * replicas
    return total_rows * int(cols), int(cols), total_rows


def legacy_mesh(
    *, packed_cores: int, cores_per_tile: int,
) -> Tuple[int, int, int, int]:
    """Packed-count mesh (no declared platform): derive cores/tile if unset,
    then the most-square exact grid over ``ceil(packed / cores_per_tile)``.

    Returns ``(cores_per_tile, n_tiles, mesh_width, mesh_height)``.
    """
    cpt = int(cores_per_tile)
    packed = int(packed_cores)
    if cpt <= 0:
        cpt = max(1, math.isqrt(packed))
        if cpt * cpt < packed:
            cpt += 1
    n_tiles = (packed + cpt - 1) // cpt
    width, height = most_square_exact_dims(n_tiles)
    return cpt, n_tiles, width, height
