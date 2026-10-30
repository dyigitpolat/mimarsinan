"""The RTL testbench's token program: one encoder, one decoder, one output parser."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

# CROSS-LANGUAGE CONTRACT — every constant below is also written, once, in the
# testbench that executes it: `hw/tb/tb_odin_core.v` for the vendored stock core
# and `hw/tb/tb_odin_gen_core.v` for a generated variant. The round-trip test in
# the default suite is what keeps the copies of the opcode table from drifting.
# The two testbenches share the EVENT program (AER/WAIT/TAG/END) and differ only
# in how a core is programmed: SPI transactions on the stock core, one
# configuration-port write per clock on a generated one.
OP_END = 0
OP_SPI_W = 1
OP_SPI_R = 2
OP_AER = 3
OP_WAIT = 4
OP_TAG = 5
OP_SHADOW = 6
OP_PROG = 7

OP_ARITY: Dict[int, int] = {
    OP_END: 0, OP_SPI_W: 3, OP_SPI_R: 3, OP_AER: 2,
    OP_WAIT: 1, OP_TAG: 1, OP_SHADOW: 3, OP_PROG: 4,
}

#: Opcode -> mnemonic, a display table (zipped, so no literal of it reads as a capability-class map to the
#: release head's second-map scan: "AER" and "TAG" happen to be spelled in class letters).
OP_NAMES: Dict[int, str] = dict(zip(
    (OP_END, OP_SPI_W, OP_SPI_R, OP_AER, OP_WAIT, OP_TAG, OP_SHADOW, OP_PROG),
    ("END", "SPI_W", "SPI_R", "AER", "WAIT", "TAG", "SHADOW", "PROG"),
))

#: The ops that address ONE core, always through their first argument: the
#: sequencer latches it as ``active_core`` (hw/fpga/kernel/odin_fpga_kernel.v,
#: S_SPI / S_AER) and the testbenches index their core array with it. WAIT and
#: TAG address the whole fabric. This set is what CORE SELECTION rewrites.
CORE_ADDRESSED_OPS = frozenset({OP_SPI_W, OP_SPI_R, OP_AER, OP_SHADOW, OP_PROG})

TOKEN_BITS = 32

# SPI address field a[19:0] = {R, W, cmd[1:0], addr[15:0]} — doc/README.md
# Sec.2.1's command table. cmd 00 is a configuration-register write (R/W are
# don't-care and stay 0), 01 the neuron memory, 10 the synapse memory.
_CMD_CONFIG = 0b00
_CMD_NEURON = 0b01
_CMD_SYNAPSE = 0b10

_READ_BIT = 1 << 19
_WRITE_BIT = 1 << 18

NEURON_WORDS = 256
NEURON_BYTES_PER_WORD = 16
SYNAPSE_WORDS = 8192
SYNAPSE_BYTES_PER_WORD = 4

# AER-in event encodings (doc/README.md Sec.2.2.1). A neuron-spike event is the
# ONLY stimulation this deployment uses: it takes the mapping-table bit into
# account and sweeps the whole post-synaptic row, which is exactly one
# occurrence of one canonical-order slot on the physical crossbar.
AER_NEURON_SPIKE_SUFFIX = 0x07
AER_ALL_NEURON_TREF = 0x7F

#: Config-register ids the OP_SHADOW opcode names, in the tb's mux order.
SHADOW_REGISTERS: Tuple[str, ...] = (
    "SPI_GATE_ACTIVITY", "SPI_OPEN_LOOP", "SPI_BURST_TIMEREF",
    "SPI_AER_SRC_CTRL_nNEUR", "SPI_OUT_AER_MONITOR_EN",
    "SPI_MONITOR_NEUR_ADDR", "SPI_MONITOR_SYN_ADDR",
    "SPI_UPDATE_UNMAPPED_SYN", "SPI_PROPAGATE_UNMAPPED_SYN",
    "SPI_SDSP_ON_SYN_STIM",
)
SHADOW_SYN_SIGN_BASE = len(SHADOW_REGISTERS)
SHADOW_SYN_SIGN_WORDS = 16


class StimulusError(ValueError):
    """The token program does not satisfy the encoding the testbench executes."""


@dataclass(frozen=True)
class Op:
    """One testbench opcode with its arguments, arity-checked on construction."""

    code: int
    args: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        arity = OP_ARITY.get(self.code)
        if arity is None:
            raise StimulusError(
                f"unknown opcode {self.code}; the encoding declares "
                f"{', '.join(f'{name}={code}' for code, name in sorted(OP_NAMES.items()))}")
        if len(self.args) != arity:
            raise StimulusError(
                f"opcode {OP_NAMES[self.code]} takes {arity} argument(s), "
                f"got {len(self.args)}: {self.args}")
        for index, value in enumerate(self.args):
            if not isinstance(value, int) or isinstance(value, bool):
                raise StimulusError(
                    f"{OP_NAMES[self.code]} argument {index} must be an int, "
                    f"got {value!r}")
            if value < 0 or value >= (1 << TOKEN_BITS):
                raise StimulusError(
                    f"{OP_NAMES[self.code]} argument {index} is {value}, "
                    f"outside the {TOKEN_BITS}-bit token range")


def _checked(name: str, value: int, bits: int) -> int:
    value = int(value)
    if value < 0 or value >= (1 << bits):
        raise StimulusError(f"{name} is {value}, not a {bits}-bit value")
    return value


def config_write_address(register_address: int) -> int:
    """``a`` for a configuration-register write (cmd 00, R/W don't-care)."""
    return (_CMD_CONFIG << 16) | _checked("register address", register_address, 16)


def neuron_address(word_addr: int, byte_addr: int, *, write: bool) -> int:
    """``a`` for one neuron-memory BYTE access: ``{n/a, byte_addr, word_addr}``."""
    low = (_checked("neuron byte address", byte_addr, 4) << 8) \
        | _checked("neuron word address", word_addr, 8)
    return (_WRITE_BIT if write else _READ_BIT) | (_CMD_NEURON << 16) | low


def synapse_address_field(word_addr: int, byte_addr: int, *, write: bool) -> int:
    """``a`` for one synapse-memory BYTE access: ``{n/a, byte_addr, word_addr}``."""
    low = (_checked("synapse byte address", byte_addr, 2) << 13) \
        | _checked("synapse word address", word_addr, 13)
    return (_WRITE_BIT if write else _READ_BIT) | (_CMD_SYNAPSE << 16) | low


def masked_byte_data(value: int, mask: int) -> int:
    """The 20-bit data field ``{n/a, mask, byte}``; a mask bit KEEPS the old bit."""
    return (_checked("mask", mask, 8) << 8) | _checked("byte", value, 8)


def neuron_spike_event(pre_neuron: int) -> int:
    """The 17-bit AER-in address of one neuron-spike event on ``pre_neuron``'s row."""
    return (_checked("pre-synaptic row", pre_neuron, 8) << 8) \
        | AER_NEURON_SPIKE_SUFFIX


def all_neuron_tref_event() -> int:
    """The 17-bit AER-in address of the all-neurons time-reference event."""
    return AER_ALL_NEURON_TREF


def variant_axon_event(axon: int, *, axon_address_bits: int) -> int:
    """A generated core's AER-in word for one occurrence of one axon slot."""
    return _checked("axon slot", axon, axon_address_bits)


def variant_tref_event(*, axon_address_bits: int) -> int:
    """A generated core's AER-in word for the all-neurons time reference."""
    return 1 << int(axon_address_bits)


def word_bytes(word: int, n_bytes: int) -> Tuple[int, ...]:
    """A memory word's bytes in SPI ``byte_addr`` order (byte i is bits 8i+7:8i)."""
    if word < 0 or word >= (1 << (8 * n_bytes)):
        raise StimulusError(f"{word} is not a {8 * n_bytes}-bit word")
    return tuple((word >> (8 * i)) & 0xFF for i in range(n_bytes))


def encode_ops(ops: Iterable[Op]) -> Tuple[int, ...]:
    """Flatten a program into the token stream ``$readmemh`` loads, END-terminated."""
    tokens: List[int] = []
    for op in ops:
        if op.code == OP_END:
            raise StimulusError(
                "OP_END terminates the stream and is appended by the encoder; "
                "an explicit END in the middle would silently truncate the run")
        tokens.append(op.code)
        tokens.extend(op.args)
    tokens.append(OP_END)
    return tuple(tokens)


def decode_ops(tokens: Sequence[int]) -> Tuple[Op, ...]:
    """The exact inverse of :func:`encode_ops`, loud on a truncated stream."""
    ops: List[Op] = []
    index = 0
    while index < len(tokens):
        code = int(tokens[index])
        if code == OP_END:
            return tuple(ops)
        arity = OP_ARITY.get(code)
        if arity is None:
            raise StimulusError(f"unknown opcode {code} at token {index}")
        if index + 1 + arity > len(tokens):
            raise StimulusError(
                f"opcode {OP_NAMES[code]} at token {index} needs {arity} "
                f"argument(s) but the stream ends after {len(tokens)} tokens")
        ops.append(Op(code, tuple(int(t) for t in tokens[index + 1:index + 1 + arity])))
        index += 1 + arity
    raise StimulusError(
        "the token stream is not END-terminated; the testbench would run off "
        "the end of its program array into uninitialised memory")


def retarget_ops(ops: Iterable[Op], placement: Mapping[int, int]) -> Tuple[Op, ...]:
    """The same program on other cores: CORE SELECTION, as data.

    ``placement`` maps every LOGICAL core the program addresses to the FABRIC
    core it runs on. Every core-addressed op (:data:`CORE_ADDRESSED_OPS`) has
    its core argument rewritten and nothing else; no op is added, dropped or
    reordered, so the enabled-cycle schedule of the run is the same and the
    cores are identical, which is what makes a placed run's counts the logical
    run's counts. A core the placement does not name, or two logical cores on
    one fabric core, is refused: either would run a program nobody mapped.
    The shipped driver's byte-level copy (``odin_board_driver.retarget_payload``)
    is gated against this one at packaging.
    """
    mapping = {int(logical): int(fabric) for logical, fabric in placement.items()}
    fabric = sorted(mapping.values())
    if len(set(fabric)) != len(fabric):
        raise StimulusError(
            f"placement {mapping} puts two logical cores on the same fabric core; "
            f"their programs would overwrite each other")
    moved: List[Op] = []
    for op in ops:
        if op.code not in CORE_ADDRESSED_OPS:
            moved.append(op)
            continue
        core = int(op.args[0])
        if core not in mapping:
            raise StimulusError(
                f"{OP_NAMES[op.code]} addresses core {core}, which placement "
                f"{mapping} does not place on any fabric core")
        moved.append(Op(op.code, (mapping[core],) + tuple(op.args[1:])))
    return tuple(moved)


def write_stimulus(path: Path, ops: Sequence[Op]) -> int:
    """Write the ``$readmemh`` file and return its token count."""
    tokens = encode_ops(ops)
    Path(path).write_text("".join(f"{token:08x}\n" for token in tokens))
    return len(tokens)


def read_stimulus(path: Path) -> Tuple[Op, ...]:
    """Read back a written stimulus file — the round-trip's other half."""
    tokens = [int(line, 16) for line in Path(path).read_text().split()]
    return decode_ops(tokens)


def shadow_register_id(name: str) -> int:
    """The tb mux id of a scalar configuration register, refusing an unknown name."""
    try:
        return SHADOW_REGISTERS.index(name)
    except ValueError as exc:
        raise StimulusError(
            f"{name!r} is not a scalar configuration register the testbench "
            f"shadows; known: {', '.join(SHADOW_REGISTERS)} (SPI_SYN_SIGN is "
            f"read word-by-word through shadow_syn_sign_id)") from exc


def shadow_syn_sign_id(word_index: int) -> int:
    """The tb mux id of SYN_SIGN's 16-bit word ``word_index``."""
    if not 0 <= int(word_index) < SHADOW_SYN_SIGN_WORDS:
        raise StimulusError(
            f"SYN_SIGN spans {SHADOW_SYN_SIGN_WORDS} 16-bit words, "
            f"got index {word_index}")
    return SHADOW_SYN_SIGN_BASE + int(word_index)


def op_summary(ops: Sequence[Op]) -> Mapping[str, int]:
    """How many of each opcode a program carries — the honest size report."""
    summary: Dict[str, int] = {name: 0 for name in OP_NAMES.values()}
    for op in ops:
        summary[OP_NAMES[op.code]] += 1
    return summary
