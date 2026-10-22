"""[U1 R4] The fused stage's channel positions of a pass's input slices, for a position-dependent encode (dither)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

#: A ``HybridStage`` (``hybrid_types`` imports the packers, which import this module's sibling).
HybridStage = Any
#: A stage's place in the program: (index in ``stages``, its level-unit index, or -1 for the stage itself). Positional,
#: so the memo stays valid through pickle and deepcopy, which give every stage a new identity.
StageKey = Tuple[int, int]
_CACHE = "_fused_position_cache"


def fused_input_positions(hybrid_mapping, stage) -> Optional[Dict[int, Optional[int]]]:
    """``{input slice index: offset}`` where the fused stage (under re-timing, the identity's level stage) encodes that
    slice's node at another offset than ``stage`` does, and None for each carried slice (replayed, never encoded); None
    when it encodes every external slice where ``stage`` does. Derived from the program's stages, memoized by key."""
    cache = getattr(hybrid_mapping, _CACHE, None)
    if cache is None:
        cache = _fused_positions(hybrid_mapping.stages)
        setattr(hybrid_mapping, _CACHE, cache)
    key = stage_key(hybrid_mapping.stages, stage)
    return None if key is None else cache.get(key)


def stage_key(stages: Sequence[HybridStage], stage) -> Optional[StageKey]:
    """``stage``'s positional key among ``stages`` and their level units; None when it is not one of them."""
    for index, candidate in enumerate(stages):
        if candidate is stage:
            return index, -1
        for unit, level_stage in enumerate(getattr(candidate, "retimed_level_stages", None) or ()):
            if level_stage is stage:
                return index, unit
    return None


def _fused_positions(stages: Sequence[HybridStage]) -> Dict[StageKey, Dict[int, Optional[int]]]:
    groups: Dict[Tuple[int, int], List[Tuple[StageKey, HybridStage]]] = defaultdict(list)
    by_segment: Dict[int, List[Tuple[int, HybridStage]]] = defaultdict(list)
    for index, stage in enumerate(stages):
        if stage.kind == "neural" and stage.schedule_segment_index is not None:
            by_segment[int(stage.schedule_segment_index)].append((index, stage))
    for segment, passes in by_segment.items():
        units = [((index, u), unit) for index, p in passes
                 for u, unit in enumerate(getattr(p, "retimed_level_stages", None) or ())]
        streaming = [((index, -1), p) for index, p in passes if not getattr(p, "retimed_level_stages", None)]
        for keyed, level in zip(units, _unit_levels([unit for _, unit in units])):
            groups[(segment, level)].append(keyed)
        groups[(segment, -1)].extend(streaming)
    out: Dict[StageKey, Dict[int, Optional[int]]] = {}
    for group in groups.values():
        if len(group) < 2:
            continue
        produced = {int(s.node_id) for _, st in group for s in st.output_map}
        layout = _layout([st for _, st in group], produced)
        for key, st in group:
            fused = {k: (None if int(s.node_id) in produced else layout[int(s.node_id)])
                     for k, s in enumerate(st.input_map)}
            if any(o is not None and o != int(s.offset) for o, s in zip(fused.values(), st.input_map)):
                out[key] = fused
    return out


def _unit_levels(units: Sequence[HybridStage]) -> List[int]:
    """Per level unit of one segment, its fused depth level: one past the deepest unit whose node it reads."""
    level_of_node: Dict[int, int] = {}
    levels = []
    for unit in units:
        level = 1 + max((level_of_node[int(s.node_id)] for s in unit.input_map if int(s.node_id) in level_of_node),
                        default=-1)
        levels.append(level)
        for s in unit.output_map:
            level_of_node[int(s.node_id)] = level
    return levels


def _layout(group: Sequence[HybridStage], produced: Set[int]) -> Dict[int, int]:
    """The input layout of the stage that executes ``group`` fused: the raw input first, then every external node in
    id order at its widest slice (``_remap_external_sources_to_segment_inputs``'s layout)."""
    sizes: Dict[int, int] = {}
    for st in group:
        for s in st.input_map:
            if int(s.node_id) not in produced:
                sizes[int(s.node_id)] = max(sizes.get(int(s.node_id), 0), int(s.size))
    layout, cursor = {-2: 0}, sizes.pop(-2, 0)
    for node in sorted(sizes):
        layout[node] = cursor
        cursor += sizes[node]
    return layout
