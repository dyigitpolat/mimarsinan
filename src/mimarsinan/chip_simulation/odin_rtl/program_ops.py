"""The exporter's sequencer program, executed as testbench opcodes.

Every stage of schema v1 has exactly one translation here, and nothing else in
the harness builds an SPI address or an AER word.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from mimarsinan.chip_simulation.odin_rtl.stimulus import (
    NEURON_BYTES_PER_WORD,
    NEURON_WORDS,
    OP_AER,
    OP_SHADOW,
    OP_SPI_R,
    OP_SPI_W,
    OP_TAG,
    OP_WAIT,
    SHADOW_REGISTERS,
    SHADOW_SYN_SIGN_WORDS,
    SYNAPSE_BYTES_PER_WORD,
    SYNAPSE_WORDS,
    Op,
    StimulusError,
    all_neuron_tref_event,
    config_write_address,
    masked_byte_data,
    neuron_address,
    neuron_spike_event,
    shadow_register_id,
    shadow_syn_sign_id,
    synapse_address_field,
    word_bytes,
)
from mimarsinan.mapping.export.odin.program import (
    STAGE_BARRIER,
    STAGE_CLEAR,
    STAGE_CONFIG,
    STAGE_GATE,
    STAGE_INJECT,
    STAGE_READOUT,
    STAGE_TREF,
    plan_injection,
)
from mimarsinan.mapping.export.odin.registers import (
    SYN_SIGN_BASE_ADDR,
    config_register,
)
from mimarsinan.mapping.platform.event_order import logical_slot_of_row

#: A whole-word write leaves no byte masked (mask bit 1 = keep the old byte).
UNMASKED = 0x00

#: Programming one core end to end, in SPI transactions (plan F13).
FULL_PROGRAM_TRANSACTIONS = (
    NEURON_WORDS * NEURON_BYTES_PER_WORD + SYNAPSE_WORDS * SYNAPSE_BYTES_PER_WORD
)


def config_register_ops(core_index: int, register_writes: Sequence[Mapping[str, int]]
                        ) -> List[Op]:
    """The configuration-register writes of one CONFIG stage, in stage order."""
    return [
        Op(OP_SPI_W, (int(core_index),
                      config_write_address(int(write["address"])),
                      int(write["value"])))
        for write in register_writes
    ]


def neuron_image_ops(core_index: int, neuron_words: Sequence[int]) -> List[Op]:
    """Every byte of the neuron memory, unmasked — the memory has no reset."""
    if len(neuron_words) != NEURON_WORDS:
        raise StimulusError(
            f"the neuron image is {NEURON_WORDS} words, got {len(neuron_words)}")
    ops: List[Op] = []
    for word_addr, word in enumerate(neuron_words):
        for byte_addr, value in enumerate(word_bytes(int(word), NEURON_BYTES_PER_WORD)):
            ops.append(Op(OP_SPI_W, (
                int(core_index),
                neuron_address(word_addr, byte_addr, write=True),
                masked_byte_data(value, UNMASKED),
            )))
    return ops


def synapse_image_ops(core_index: int, synapse_words: Sequence[int]) -> List[Op]:
    """Every byte of the synapse memory, unmasked."""
    if len(synapse_words) != SYNAPSE_WORDS:
        raise StimulusError(
            f"the synapse image is {SYNAPSE_WORDS} words, got {len(synapse_words)}")
    ops: List[Op] = []
    for word_addr, word in enumerate(synapse_words):
        for byte_addr, value in enumerate(word_bytes(int(word), SYNAPSE_BYTES_PER_WORD)):
            ops.append(Op(OP_SPI_W, (
                int(core_index),
                synapse_address_field(word_addr, byte_addr, write=True),
                masked_byte_data(value, UNMASKED),
            )))
    return ops


def config_stage_ops(payload: Mapping[str, Any]) -> List[Op]:
    """One CONFIG stage: registers first (it asserts the gate), then both memories."""
    core_index = int(payload["core_index"])
    return (
        config_register_ops(core_index, payload["register_writes"])
        + neuron_image_ops(core_index, payload["neuron_words"])
        + synapse_image_ops(core_index, payload["synapse_words"])
    )


def gate_stage_ops(core_indices: Sequence[int], *, on: bool) -> List[Op]:
    """A GATE stage reaches every core: the gate is per-chip, the program is not."""
    address = config_write_address(config_register("SPI_GATE_ACTIVITY").address)
    return [Op(OP_SPI_W, (int(core), address, 1 if on else 0)) for core in core_indices]


def clear_stage_ops(payload: Mapping[str, Any]) -> List[Op]:
    """One CLEAR stage: the masked neuron-state byte writes, per used neuron."""
    core_index = int(payload["core_index"])
    return [
        Op(OP_SPI_W, (
            core_index,
            neuron_address(int(write["neuron"]), int(write["byte_addr"]), write=True),
            masked_byte_data(int(write["value"]), int(write["mask"])),
        ))
        for write in payload["byte_writes"]
    ]


def inject_ops(core_index: int, events: Sequence[Tuple[int, int]]) -> List[Op]:
    """``(row, multiplicity)`` deliveries as neuron-spike events, adjacency intact.

    One occurrence is one physical row event: the raw AER stream has no
    multiplicity field, so ``k`` occurrences of a slot are ``k`` back-to-back
    events on that row — which is exactly the adjacency the canonical order
    demands and the wire does not carry (plan §2.3, F8).
    """
    ops: List[Op] = []
    for row, multiplicity in events:
        if int(multiplicity) < 0:
            raise StimulusError(
                f"multiplicity must be non-negative, got {multiplicity} at row {row}")
        event = neuron_spike_event(int(row))
        for _occurrence in range(int(multiplicity)):
            ops.append(Op(OP_AER, (int(core_index), event)))
    return ops


def tref_stage_ops(payload: Mapping[str, Any], core_indices: Sequence[int]) -> List[Op]:
    """A TREF stage: the all-neurons time reference the schema declares."""
    scope = str(payload.get("scope"))
    if scope != "all":
        raise StimulusError(
            f"the testbench executes the all-neurons time reference only; the "
            f"program asks for scope={scope!r}")
    event = all_neuron_tref_event()
    return [Op(OP_AER, (int(core), event)) for core in core_indices]


def barrier_stage_ops(payload: Mapping[str, Any]) -> List[Op]:
    """A BARRIER stage is a plain wait of the program's own deterministic bound."""
    return [Op(OP_WAIT, (int(payload["cycles"]),))]


def tag_op(tag: int) -> Op:
    """Open a READOUT window: every capture until the next tag carries this one."""
    return Op(OP_TAG, (int(tag),))


def neuron_readback_ops(core_index: int, neuron_words: Sequence[int]) -> List[Op]:
    """Read back the FULL neuron memory and byte-compare against the image."""
    ops: List[Op] = []
    for word_addr, word in enumerate(neuron_words):
        for byte_addr, value in enumerate(word_bytes(int(word), NEURON_BYTES_PER_WORD)):
            ops.append(Op(OP_SPI_R, (
                int(core_index),
                neuron_address(word_addr, byte_addr, write=False),
                value,
            )))
    return ops


def synapse_readback_ops(core_index: int, synapse_words: Sequence[int]) -> List[Op]:
    """Read back the FULL synapse memory and byte-compare against the image."""
    ops: List[Op] = []
    for word_addr, word in enumerate(synapse_words):
        for byte_addr, value in enumerate(word_bytes(int(word), SYNAPSE_BYTES_PER_WORD)):
            ops.append(Op(OP_SPI_R, (
                int(core_index),
                synapse_address_field(word_addr, byte_addr, write=False),
                value,
            )))
    return ops


def shadow_ops(core_index: int, register_writes: Sequence[Mapping[str, int]],
               *, gate_on: bool) -> List[Op]:
    """Assert every configuration register hierarchically (no readback exists).

    SIMULATION ONLY. `doc/README.md` Sec.4: the registers "can be written
    through the SPI bus (no readback operation is available) and do not have a
    default reset value". The runtime therefore reprograms them every session
    and cannot verify them; the testbench taps them inside the DUT instead, and
    that tap is not a capability the deployed system has.
    """
    values: Dict[int, int] = {
        int(write["address"]): int(write["value"]) for write in register_writes
    }
    ops: List[Op] = []
    for name in SHADOW_REGISTERS:
        address = config_register(name).address
        if name == "SPI_GATE_ACTIVITY":
            expected = 1 if gate_on else 0
        elif address not in values:
            raise StimulusError(
                f"the CONFIG stage never writes {name!r} (address {address}), "
                f"so the testbench has nothing to assert it against — a "
                f"register with no reset value that nobody programs is exactly "
                f"the silent-garbage case the shadow gate exists to catch")
        else:
            expected = values[address]
        ops.append(Op(OP_SHADOW, (int(core_index), shadow_register_id(name), expected)))
    for word in range(SHADOW_SYN_SIGN_WORDS):
        address = SYN_SIGN_BASE_ADDR + word
        if address not in values:
            raise StimulusError(
                f"SYN_SIGN word {word} (address {address}) is never written")
        ops.append(Op(OP_SHADOW, (
            int(core_index), shadow_syn_sign_id(word), values[address])))
    return ops


def slot_rows_from_inject(payload: Mapping[str, Any]) -> Tuple[Dict[int, Tuple[int, ...]],
                                                               Tuple[int, ...]]:
    """``(slot -> emitting physical rows, bias slots)`` reconstructed from INJECT.

    The exported stage carries the always-on deliveries as concrete events and
    delegates the rest to the host as ``runtime_rows``; together they are the
    complete slot->row map the per-cycle plan needs, so the harness never
    re-derives the row-pair expansion.
    """
    slot_rows: Dict[int, List[int]] = {}
    bias: List[int] = []
    for row, _multiplicity in payload["events"]:
        slot = logical_slot_of_row(int(row))
        slot_rows.setdefault(slot, []).append(int(row))
        if slot not in bias:
            bias.append(slot)
    for slot, rows in payload["runtime_rows"]:
        if int(slot) in slot_rows:
            raise StimulusError(
                f"slot {slot} is both injected at export time and delegated to "
                f"the host; one of the two would deliver a duplicate event")
        slot_rows[int(slot)] = [int(row) for row in rows]
    return (
        {slot: tuple(sorted(rows)) for slot, rows in slot_rows.items()},
        tuple(sorted(bias)),
    )


def plan_cycle_injection(slot_rows: Mapping[int, Sequence[int]],
                         counts: Sequence[int]) -> Tuple[Tuple[int, int], ...]:
    """One cycle's per-slot counts -> the physical delivery, through the ONE planner."""
    return plan_injection(
        counts=counts,
        emitting_rows=lambda slot: tuple(slot_rows.get(int(slot), ())),
    )


def stages_of_kind(program: Any, kind: str) -> List[Mapping[str, Any]]:
    """Every stage of one kind, in program order."""
    return [stage["payload"] for stage in program.stages if stage["kind"] == kind]


__all__ = [
    "FULL_PROGRAM_TRANSACTIONS",
    "STAGE_BARRIER", "STAGE_CLEAR", "STAGE_CONFIG", "STAGE_GATE",
    "STAGE_INJECT", "STAGE_READOUT", "STAGE_TREF",
    "barrier_stage_ops", "clear_stage_ops", "config_register_ops",
    "config_stage_ops", "gate_stage_ops", "inject_ops", "neuron_image_ops",
    "neuron_readback_ops", "plan_cycle_injection", "shadow_ops",
    "slot_rows_from_inject", "stages_of_kind", "synapse_image_ops",
    "synapse_readback_ops", "tag_op", "tref_stage_ops",
]
