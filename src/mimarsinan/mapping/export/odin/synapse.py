"""The stock-ODIN synapse memory: addressing and 4-bit nibble packing."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from mimarsinan.mapping.export.odin.layout import OdinLayoutError

# CROSS-LANGUAGE CONTRACT — transcribed from ChFrenkel/ODIN @ 1781931:
#   * the memory is 8192 words of 32 bits, 8 four-bit synapses per word:
#     src/synaptic_core.v:136-150 (SRAM_8192x32_wrapper) and doc/README.md Sec.3.1;
#   * a synapse nibble is {mapping_bit, w[2:0]}: src/synaptic_core.v:78-88 reads
#     SYNARRAY_RDATA[(i<<2)+3] as the mapping bit over SYNARRAY_RDATA[(i<<2)+2:0];
#   * the address split is word {pre_neur[7:0], post_neur[7:3]}, byte
#     post_neur[2:1], nibble post_neur[0] (0 = least-significant half-byte):
#     doc/README.md Sec.3.1, synapse-memory address table;
#   * SPI_SYN_SIGN (in `registers.py`, it is a config register) is indexed by the
#     PRE-synaptic address, so the weight sign is a property of the physical ROW:
#     src/synaptic_core.v:128 (SPI_SYN_SIGN >> CTRL_SYNARRAY_ADDR[12:5]).

SYNAPSE_WORD_BITS = 32
SYNAPSE_WORD_COUNT = 8192
SYNAPSE_NIBBLES_PER_WORD = 8
SYNAPSE_MAGNITUDE_BITS = 3
SYNAPSE_MAX_MAGNITUDE = (1 << SYNAPSE_MAGNITUDE_BITS) - 1
SYNAPSE_MAPPING_BIT = 3

CROSSBAR_ROWS = 256
CROSSBAR_COLUMNS = 256



def synapse_address(pre: int, post: int) -> Tuple[int, int, int]:
    """``(word_addr, byte_addr, nibble)`` of the synapse at row ``pre``, column ``post``."""
    if not 0 <= pre < CROSSBAR_ROWS:
        raise OdinLayoutError(
            f"pre-synaptic row {pre} outside the {CROSSBAR_ROWS}-row crossbar")
    if not 0 <= post < CROSSBAR_COLUMNS:
        raise OdinLayoutError(
            f"post-synaptic column {post} outside the "
            f"{CROSSBAR_COLUMNS}-column crossbar")
    return ((pre << 5) | (post >> 3), (post >> 1) & 0b11, post & 0b1)


def pack_synapse_nibble(*, magnitude: int, mapped: bool) -> int:
    """One 4-bit synapse: the mapping bit over the 3-bit unsigned magnitude."""
    if isinstance(magnitude, bool) or not isinstance(magnitude, int):
        raise OdinLayoutError(f"synapse magnitude must be an integer, got {magnitude!r}")
    if magnitude < 0 or magnitude > SYNAPSE_MAX_MAGNITUDE:
        raise OdinLayoutError(
            f"synapse magnitude {magnitude} outside the "
            f"{SYNAPSE_MAGNITUDE_BITS}-bit unsigned range "
            f"[0, {SYNAPSE_MAX_MAGNITUDE}]")
    return (int(bool(mapped)) << SYNAPSE_MAPPING_BIT) | magnitude


def unpack_synapse_nibble(nibble: int) -> Tuple[int, bool]:
    """``(magnitude, mapped)`` of one 4-bit synapse."""
    if not 0 <= nibble <= 0xF:
        raise OdinLayoutError(f"{nibble} is not a 4-bit synapse nibble")
    return (nibble & SYNAPSE_MAX_MAGNITUDE, bool((nibble >> SYNAPSE_MAPPING_BIT) & 1))


def _nibble_shift(byte_addr: int, nibble: int) -> int:
    return 4 * (2 * byte_addr + nibble)


def pack_synapse_words(
    magnitudes: Sequence[Sequence[int]], *, mapped: bool
) -> Tuple[int, ...]:
    """Pack a full ``256 x 256`` unsigned-magnitude crossbar into the 8192-word image."""
    if len(magnitudes) != CROSSBAR_ROWS:
        raise OdinLayoutError(
            f"the crossbar image needs {CROSSBAR_ROWS} rows, got {len(magnitudes)}")
    words = [0] * SYNAPSE_WORD_COUNT
    for pre, row in enumerate(magnitudes):
        if len(row) != CROSSBAR_COLUMNS:
            raise OdinLayoutError(
                f"row {pre} needs {CROSSBAR_COLUMNS} columns, got {len(row)}")
        for post, magnitude in enumerate(row):
            nibble_value = pack_synapse_nibble(
                magnitude=int(magnitude), mapped=mapped)
            if not nibble_value:
                continue
            word_addr, byte_addr, nibble = synapse_address(pre, post)
            words[word_addr] |= nibble_value << _nibble_shift(byte_addr, nibble)
    return tuple(words)


def unpack_synapse_words(
    words: Sequence[int],
) -> Tuple[List[List[int]], List[List[bool]]]:
    """The inverse of :func:`pack_synapse_words`: magnitudes and mapping bits."""
    if len(words) != SYNAPSE_WORD_COUNT:
        raise OdinLayoutError(
            f"the synapse image is {SYNAPSE_WORD_COUNT} words, got {len(words)}")
    magnitudes = [[0] * CROSSBAR_COLUMNS for _ in range(CROSSBAR_ROWS)]
    mapped = [[False] * CROSSBAR_COLUMNS for _ in range(CROSSBAR_ROWS)]
    for pre in range(CROSSBAR_ROWS):
        for post in range(CROSSBAR_COLUMNS):
            word_addr, byte_addr, nibble = synapse_address(pre, post)
            raw = (words[word_addr] >> _nibble_shift(byte_addr, nibble)) & 0xF
            magnitudes[pre][post], mapped[pre][post] = unpack_synapse_nibble(raw)
    return magnitudes, mapped
