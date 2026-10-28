"""[U-F4] The stage attributes of a pipelined (wavefront) program and their one accessor each.

A wavefront group's first pass loads the group's layout: every core of it, the cores no instance of that pass uses
included (``schedule_dark_cores``: loaded, not evaluated), and it carries the group's description
(``schedule_wavefront``). Every later pass of the group runs wholly resident on that layout and holds only the layout
cores an instance of it uses, each with the head ordinal of the core it runs on (``schedule_core_ordinals``). All three
are declared-only: a program built without the decision carries none of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet, Optional, Tuple

from mimarsinan.mapping.platform.mapping_structure import PositionMultiplexingError

SCHEDULE_CORE_ORDINALS = "schedule_core_ordinals"
SCHEDULE_DARK_CORES = "schedule_dark_cores"
SCHEDULE_WAVEFRONT = "schedule_wavefront"


@dataclass(frozen=True)
class WavefrontHead:
    """A wavefront group as its first pass states it: its levels, its passes and the layout cores it loads."""

    levels: Tuple[int, ...]
    passes: int
    layout_cores: int


def core_ordinals(stage: Any, count: int) -> Tuple[int, ...]:
    """Per core of ``stage``, the ordinal of the core it runs on in its residency chain's head (itself by default)."""
    ordinals = getattr(stage, SCHEDULE_CORE_ORDINALS, None)
    return tuple(range(int(count))) if ordinals is None else tuple(int(o) for o in ordinals)


def dark_cores(stage: Any) -> FrozenSet[int]:
    """The cores ``stage`` loads without evaluating (a wavefront head's cores no instance of the pass uses)."""
    return frozenset(int(i) for i in (getattr(stage, SCHEDULE_DARK_CORES, None) or ()))


def wavefront_head(stage: Any) -> Optional[WavefrontHead]:
    return getattr(stage, SCHEDULE_WAVEFRONT, None)


def stage_groups(stages: Any) -> Tuple[Optional[int], ...]:
    """Per stage of a program, the stage index of its wavefront group's first pass (None outside every group)."""
    out: list = [None] * len(stages)
    for index, stage in enumerate(stages):
        head = wavefront_head(stage)
        if head is None:
            continue
        neural = [i for i in range(index, len(stages)) if getattr(stages[i], "kind", None) == "neural"]
        for i in neural[:int(head.passes)]:
            out[i] = index
    return tuple(out)


def program_groups(programs: Any) -> Tuple[Optional[int], ...]:
    """Per planned pass (``PassProgram``), the pass index of its wavefront group's first pass (None outside)."""
    out: list = []
    head = None
    for index, program in enumerate(programs):
        if getattr(program, "wavefront", False) and getattr(program, "core_ordinals", None) is None:
            head = index
        out.append(head if getattr(program, "wavefront", False) else None)
    return tuple(out)


def is_wavefront_program(stages: Any) -> bool:
    """Whether a built program holds a wavefront group (a backend that cannot run one refuses it by name)."""
    return any(wavefront_head(stage) is not None for stage in stages)


def refuse_wavefront_backend(stages: Any, backend: str) -> None:
    """A backend not composed with wavefront programs refuses one by name: each group's first pass loads layout cores
    no instance of that pass uses, and the backend runs (and censuses) every core it is handed."""
    if is_wavefront_program(stages):
        raise PositionMultiplexingError(
            f"pipelined_multiplexing: the {backend} backend is not composed with a wavefront program (a group's first "
            "pass loads cores none of its instances use; the backend would run them and count their activity); run "
            "it on the reference executors (HCM, value, TTFS) or without pipelined_multiplexing")
