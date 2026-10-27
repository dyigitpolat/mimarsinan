"""[E85/W03, F3] The one pinned rule of a residency plan, on any plane: how many tiles each pass pins, and which of its
cores; a core holding a dynamic operand bank (N2: its payload is rewritten every inference) is never pinned, so its
tiles are written every period."""

from __future__ import annotations

from typing import FrozenSet, Optional, Sequence, Tuple

#: One packed core: its occupied cells and the physical tiles it occupies (``residency_tiles.PackedCore``).
PackedCore = Tuple[int, int]


def pinned_tiles(pass_tiles: Sequence[int], r: int, C: int, operand: Sequence[int] = ()) -> Tuple[int, ...]:
    """Pins per pass: none without a pinned budget (r = C, the stateless plan: every tile rewritten each period, also
    a phase packed beyond C); else the forced (R_q - r)^+, then the rest of the budget C - r in program order. [F3]
    Only a pass's static tiles are pinned (``operand``: per pass the tiles of its operand cores, never pinned): the
    forced pins are at most the static tiles (a step's phases hold at most r operand tiles, so they are never fewer)."""
    static = [int(t) - int(o) for t, o in zip(pass_tiles, operand)] if operand else [int(t) for t in pass_tiles]
    forced = [min(s, max(0, int(t) - int(r))) if int(r) < int(C) else 0 for t, s in zip(pass_tiles, static)]
    spare = max(0, int(C) - int(r) - sum(forced))
    out = []
    for s, k in zip(static, forced):
        extra = min(spare, s - k)
        spare -= extra
        out.append(k + extra)
    return tuple(out)


def pinned_cores(core_payloads: Sequence[int], k: int, core_tiles: Optional[Sequence[int]] = None, *,
                 operand_cores: FrozenSet[int] = frozenset()) -> FrozenSet[int]:
    """Cores of largest payload first (ties to the lower index) while their tiles stay within ``k``; an operand
    core (N2: its payload is rewritten every inference) is never pinned."""
    tiles = core_tiles if core_tiles is not None else [1] * len(core_payloads)
    chosen, used = [], 0
    for i in sorted((i for i in range(len(core_payloads)) if i not in operand_cores),
                    key=lambda i: (-int(core_payloads[i]), i)):
        if used + int(tiles[i]) <= int(k):
            chosen.append(i)
            used += int(tiles[i])
    return frozenset(chosen)


def operand_core_indices(producer: Sequence[int]) -> FrozenSet[int]:
    """[F3] The cores of a pass that hold operand-bank regions (``producer``: per core its producer cells)."""
    return frozenset(i for i, cells in enumerate(producer) if int(cells))


def pin_passes(pass_cores: Sequence[Sequence[PackedCore]], r: int, C: int,
               producer: Sequence[Sequence[int]] = ()) -> Tuple[FrozenSet[int], ...]:
    """The one pinned rule, on any plane: ``pinned_tiles`` of each pass's packed tiles, taken by ``pinned_cores``;
    [F3] with ``producer`` (per pass, each core's operand-bank cells; () where it holds none) its operand cores are
    never pinned and their tiles are never counted as pins."""
    operand = [operand_core_indices(producer[q]) if q < len(producer) else frozenset() for q in range(len(pass_cores))]
    counts = pinned_tiles([sum(int(t) for _, t in cores) for cores in pass_cores], int(r), int(C),
                          [sum(int(cores[i][1]) for i in ops) for cores, ops in zip(pass_cores, operand)])
    return tuple(pinned_cores([c for c, _ in cores], k, [t for _, t in cores], operand_cores=ops)
                 for cores, k, ops in zip(pass_cores, counts, operand))
