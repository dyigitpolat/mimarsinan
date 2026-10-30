"""The stock-ODIN neuron-memory bit layout: fields, packing, and the state rewrite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

# CROSS-LANGUAGE CONTRACT (the one comment class the code cannot express):
# every bit range below is transcribed from the upstream RTL at
# ChFrenkel/ODIN @ 1781931 and is the SPECIFICATION, not a convenience.
#   * the LIF field ranges are the port map in src/neuron_core.v:142-156
#     (param_leak_str NEUR_STATE[7:1] ... state_caleak_cnt NEUR_STATE[85:81]),
#     restated in prose in doc/README.md Sec.3.3.2;
#   * the model-select LSB is src/neuron_core.v:97 (NEUR_STATE[0]);
#   * the disable MSB is src/neuron_core.v:136 (NEUR_STATE[127]);
#   * the word is 128 bits and the memory 256 words: src/neuron_core.v:80 and
#     doc/README.md Sec.3.1;
#   * SPI writes are BYTE-granular with a mask where 1 = masked = keep the old
#     byte: doc/README.md Sec.2.1, the 20-bit data-field table.

NEURON_WORD_BITS = 128
NEURON_WORD_BYTES = NEURON_WORD_BITS // 8
NEURON_MEMORY_WORDS = 256

PARAMETER_FIELD = "parameter"
STATE_FIELD = "state"


class OdinLayoutError(ValueError):
    """A value does not fit the stock-ODIN memory layout it was handed to."""


@dataclass(frozen=True)
class BitField:
    """One named bit range of a memory word, with its parameter/state kind."""

    name: str
    lsb: int
    width: int
    kind: str

    @property
    def mask(self) -> int:
        return ((1 << self.width) - 1) << self.lsb

    def encode(self, value: int) -> int:
        if isinstance(value, bool):
            value = int(value)
        if not isinstance(value, int):
            raise OdinLayoutError(
                f"field {self.name!r} takes an integer, got {value!r}")
        if value < 0 or value >= (1 << self.width):
            raise OdinLayoutError(
                f"field {self.name!r} is {self.width} bits wide "
                f"([0, {(1 << self.width) - 1}]); {value} does not fit")
        return value << self.lsb

    def decode(self, word: int) -> int:
        return (word >> self.lsb) & ((1 << self.width) - 1)


LIF_NEURON_WORD_FIELDS: Tuple[BitField, ...] = (
    BitField("lif_izh_sel", 0, 1, PARAMETER_FIELD),
    BitField("leak_str", 1, 7, PARAMETER_FIELD),
    BitField("leak_en", 8, 1, PARAMETER_FIELD),
    BitField("thr", 9, 8, PARAMETER_FIELD),
    BitField("ca_en", 17, 1, PARAMETER_FIELD),
    BitField("thetamem", 18, 8, PARAMETER_FIELD),
    BitField("ca_theta1", 26, 3, PARAMETER_FIELD),
    BitField("ca_theta2", 29, 3, PARAMETER_FIELD),
    BitField("ca_theta3", 32, 3, PARAMETER_FIELD),
    BitField("ca_leak", 35, 5, PARAMETER_FIELD),
    BitField("vmem", 70, 8, STATE_FIELD),
    BitField("calcium", 78, 3, STATE_FIELD),
    BitField("caleak_cnt", 81, 5, STATE_FIELD),
    BitField("neur_disable", 127, 1, PARAMETER_FIELD),
)

_FIELDS_BY_NAME: Dict[str, BitField] = {f.name: f for f in LIF_NEURON_WORD_FIELDS}


def lif_neuron_field(name: str) -> BitField:
    """The named LIF field, refusing an unknown name rather than returning None."""
    field = _FIELDS_BY_NAME.get(name)
    if field is None:
        raise OdinLayoutError(
            f"unknown LIF neuron field {name!r}; known fields: "
            f"{', '.join(sorted(_FIELDS_BY_NAME))}")
    return field


def pack_neuron_word(values: Mapping[str, int]) -> int:
    """Pack a COMPLETE field map into the 128-bit word; partial maps are refused."""
    missing = sorted(set(_FIELDS_BY_NAME) - set(values))
    if missing:
        raise OdinLayoutError(
            f"the neuron word needs every field; missing: {', '.join(missing)}")
    unknown = sorted(set(values) - set(_FIELDS_BY_NAME))
    if unknown:
        raise OdinLayoutError(
            f"unknown LIF neuron field(s): {', '.join(unknown)}")
    word = 0
    for field in LIF_NEURON_WORD_FIELDS:
        word |= field.encode(values[field.name])
    return word


def unpack_neuron_word(word: int) -> Dict[str, int]:
    """The inverse of :func:`pack_neuron_word` over every declared field."""
    if word < 0 or word >= (1 << NEURON_WORD_BITS):
        raise OdinLayoutError(f"{word} is not a {NEURON_WORD_BITS}-bit word")
    return {field.name: field.decode(word) for field in LIF_NEURON_WORD_FIELDS}


def neuron_word_to_bytes(word: int) -> Tuple[int, ...]:
    """The 16 bytes in SPI ``byte_addr`` order: byte i is bits ``[8i+7:8i]``."""
    if word < 0 or word >= (1 << NEURON_WORD_BITS):
        raise OdinLayoutError(f"{word} is not a {NEURON_WORD_BITS}-bit word")
    return tuple((word >> (8 * i)) & 0xFF for i in range(NEURON_WORD_BYTES))


def neuron_word_from_bytes(payload) -> int:
    """Reassemble a word from its SPI byte order."""
    payload = tuple(payload)
    if len(payload) != NEURON_WORD_BYTES:
        raise OdinLayoutError(
            f"a neuron word is {NEURON_WORD_BYTES} bytes, got {len(payload)}")
    word = 0
    for index, byte in enumerate(payload):
        if byte < 0 or byte > 0xFF:
            raise OdinLayoutError(f"byte {index} is {byte}, not a byte")
        word |= byte << (8 * index)
    return word


def neuron_state_bit_mask() -> int:
    """The union of every STATE field's bits — what a per-sample CLEAR rewrites."""
    mask = 0
    for field in LIF_NEURON_WORD_FIELDS:
        if field.kind == STATE_FIELD:
            mask |= field.mask
    return mask


@dataclass(frozen=True)
class MaskedByteWrite:
    """One SPI byte write: ``mask`` bits are KEPT from the old byte (1 = masked)."""

    byte_addr: int
    value: int
    mask: int


def masked_state_byte_writes(word: int) -> Tuple[MaskedByteWrite, ...]:
    """The masked byte writes that install ``word``'s STATE fields and nothing else.

    Derived from the field table, so a layout with different state bits produces
    different writes without a second place to edit.
    """
    state_mask = neuron_state_bit_mask()
    payload = neuron_word_to_bytes(word)
    writes = []
    for byte_addr in range(NEURON_WORD_BYTES):
        touched = (state_mask >> (8 * byte_addr)) & 0xFF
        if not touched:
            continue
        writes.append(MaskedByteWrite(
            byte_addr=byte_addr,
            value=payload[byte_addr] & touched,
            mask=(~touched) & 0xFF,
        ))
    return tuple(writes)
