"""[U-F4] The stage attributes of a pipelined (wavefront) program and their one accessor each.

A wavefront group's first pass loads the group's layout: every core of it, the cores no instance of that pass uses
included (``schedule_dark_cores``: loaded, not evaluated), and it carries the group's description
(``schedule_wavefront``). Every later pass of the group runs wholly resident on that layout and holds only the layout
cores an instance of it uses, each with the head ordinal of the core it runs on (``schedule_core_ordinals``). All three
are declared-only: a program built without the decision carries none of them.

[thru:w1] Under layer copies a core may hold several copies, and a pass may run an instance on one of them only: the
core runs wholly, and the pass declares the columns of its copies that hold no instance of the pass
(``schedule_idle_columns``: loaded, run on off axons, read by nothing). Declared-only as well.

[thru:w2] Under the batch pipeline a run serves B inputs, and a pass that serves several inputs is one stage per input
slot, all with the pass's index, each carrying its slot (``schedule_input_slot``: ``InputSlot(slot, slots)``). The
first stage of a group's first pass loads the layout and every other stage of the group runs resident on it, so a
group's stages are read by pass index (``stage_groups``). Input k of a batch is slot k mod B (``slot_rows``): a slot
stage reads and writes only its slot's rows. Declared-only as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional, Tuple

from mimarsinan.mapping.platform.mapping_structure import PositionMultiplexingError

SCHEDULE_CORE_ORDINALS = "schedule_core_ordinals"
SCHEDULE_DARK_CORES = "schedule_dark_cores"
SCHEDULE_WAVEFRONT = "schedule_wavefront"
SCHEDULE_IDLE_COLUMNS = "schedule_idle_columns"
SCHEDULE_INPUT_SLOT = "schedule_input_slot"


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


def idle_columns(stage: Any) -> Dict[int, FrozenSet[int]]:
    """[thru:w1] Per core of ``stage``, the columns it loads and runs on off axons (a copy no instance of the pass
    uses, beside one that runs)."""
    return column_sets(getattr(stage, SCHEDULE_IDLE_COLUMNS, None))


def column_sets(runs: Any) -> Dict[int, FrozenSet[int]]:
    """[thru:w1] Declared ``(core, first column, columns)`` runs as each core's set of columns."""
    out: Dict[int, set] = {}
    for core, first, count in runs or ():
        out.setdefault(int(core), set()).update(range(int(first), int(first) + int(count)))
    return {core: frozenset(columns) for core, columns in out.items()}


def wavefront_head(stage: Any) -> Optional[WavefrontHead]:
    return getattr(stage, SCHEDULE_WAVEFRONT, None)


@dataclass(frozen=True)
class InputSlot:
    """[thru:w2] The input slot a stage serves (``slot``, from 0) of a run's ``slots`` inputs."""

    slot: int
    slots: int


def input_slot(stage: Any) -> Optional[InputSlot]:
    """[thru:w2] The input slot ``stage`` serves (None: a stage of every input, as without the batch pipeline)."""
    return getattr(stage, SCHEDULE_INPUT_SLOT, None)


def program_slots(stages: Any) -> Optional[int]:
    """[thru:w2] The inputs a run of the program serves (None without the batch pipeline: one)."""
    return next((slot.slots for slot in map(input_slot, stages) if slot is not None), None)


def serves_slot(stage: Any, slot: int) -> bool:
    """[thru:w2] Whether ``stage`` runs for input slot ``slot`` (a stage without a slot runs for every input)."""
    own = input_slot(stage)
    return own is None or int(own.slot) == int(slot)


@dataclass(frozen=True)
class PassUnion:
    """[thru:w3] The stages of one pass of a program of input slots (one per slot) as the one pass they run as: each
    layout core once, by its head ordinal, as ``(member, core index)``; the head ordinals of a wholly resident pass
    (None where a member loads the layout, whose cores then come in its order); the cores no member evaluates (a
    loading head's dark cores less every core another slot's stage runs); the loading member (None: none loads)."""

    cores: Tuple[Tuple[int, int], ...]
    ordinals: Optional[Tuple[int, ...]]
    dark: FrozenSet[int]
    loader: Optional[int]


def pass_union(members: Any) -> PassUnion:
    """[thru:w3] One pass of several input slots' stages, each member ``(cores, head ordinals or None, dark cores,
    wholly resident)``: two slots' instances on one core in one pass are one evaluation of that core (the copies of a
    key take disjoint rows and columns), so the pass's cores are counted once each. A single member is itself."""
    members = list(members)
    if len(members) == 1:
        count, ordinals, dark, resident = members[0]
        return PassUnion(tuple((0, i) for i in range(int(count))), None if ordinals is None else tuple(ordinals),
                         frozenset(dark), None if resident else 0)
    loaders = [m for m, member in enumerate(members) if not member[3]]
    if len(loaders) > 1:
        raise PositionMultiplexingError(f"batch_pipeline_slots: members {loaders} of one pass each load cores; only a "
                                        "group's first stage loads its layout and every other stage runs resident")
    loader = loaders[0] if loaders else None
    used: Dict[int, Tuple[int, int]] = {}
    for m, (count, ordinals, dark, _resident) in enumerate(members):
        for i, o in enumerate(range(int(count)) if ordinals is None else ordinals):
            if not (m == loader and i in dark):  # a loading head's dark core is loaded, not evaluated, by it
                used.setdefault(int(o), (m, i))
    if loader is not None:
        count = int(members[loader][0])
        return PassUnion(tuple((loader, o) for o in range(count)), None,
                         frozenset(o for o in range(count) if o not in used), loader)
    order = sorted(used)
    return PassUnion(tuple(used[o] for o in order), tuple(order), frozenset(), None)


def slot_rows(slot: int, slots: int, batch: int) -> range:
    """[thru:w2] The rows of a batch of ``batch`` inputs that input slot ``slot`` of ``slots`` serves: input k is
    slot k mod ``slots`` of run k // ``slots``."""
    return range(int(slot), int(batch), int(slots))


def stage_groups(stages: Any) -> Tuple[Optional[int], ...]:
    """Per stage of a program, the stage index of its wavefront group's first pass (None outside every group).
    [thru:w2] Read by pass index: a group's stages are its segment's neural stages from its head on whose pass index
    lies in the head's passes, so a pass of several input slots (one stage each) is the group's once per slot."""
    out: list = [None] * len(stages)
    for index, stage in enumerate(stages):
        head = wavefront_head(stage)
        if head is None:
            continue
        if input_slot(stage) is None:
            neural = [i for i in range(index, len(stages)) if getattr(stages[i], "kind", None) == "neural"]
            for i in neural[:int(head.passes)]:
                out[i] = index
            continue
        first, segment = int(stage.schedule_pass_index), stage.schedule_segment_index
        for i in range(index, len(stages)):
            member = stages[i]
            if getattr(member, "kind", None) != "neural":
                continue
            if (member.schedule_segment_index != segment
                    or not first <= int(member.schedule_pass_index) < first + int(head.passes)):
                break
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
    no instance of that pass uses, and the backend runs (and censuses) every core it is handed. [thru:w2] A program of
    input slots likewise: its stages serve one input slot each, and the backend runs every stage on every input."""
    if program_slots(stages) is not None:
        raise PositionMultiplexingError(
            f"batch_pipeline_slots: the {backend} backend is not composed with a program of input slots (each stage "
            "serves the rows of one input slot; the backend would run every stage on every input); run it on the "
            "reference executors (HCM, value, TTFS) or without batch_pipeline_slots")
    if is_wavefront_program(stages):
        raise PositionMultiplexingError(
            f"pipelined_multiplexing: the {backend} backend is not composed with a wavefront program (a group's first "
            "pass loads cores none of its instances use; the backend would run them and count their activity); run "
            "it on the reference executors (HCM, value, TTFS) or without pipelined_multiplexing")
