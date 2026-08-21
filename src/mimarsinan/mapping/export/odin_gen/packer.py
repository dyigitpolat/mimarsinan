"""Packer tables for a generated core: the writes its configuration port takes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import numpy as np

from mimarsinan.mapping.export.odin.feasibility import check_weight_magnitudes
from mimarsinan.mapping.export.odin_gen.spec import CoreSpec, require_generatable

# CROSS-LANGUAGE CONTRACT — the four selectors of the generated core's
# configuration port, mirrored in `hw/gen/odin_gen_core.v.tmpl`.
PROG_SEL_REGISTER = 0
PROG_SEL_THRESHOLD = 1
PROG_SEL_MEMBRANE = 2
PROG_SEL_SYNAPSE = 3

#: The only configuration register the variant core carries: activity gating.
REGISTER_GATE = 0


class VariantPackError(ValueError):
    """A packed core does not fit the geometry its spec declares."""


@dataclass(frozen=True)
class ProgWrite:
    """One configuration-port write: selector, address, 32-bit data."""

    sel: int
    addr: int
    data: int


@dataclass(frozen=True)
class VariantCoreImage:
    """One generated core's complete programming payload, by stage."""

    core_index: int
    theta: int
    threshold_writes: Tuple[ProgWrite, ...]
    synapse_writes: Tuple[ProgWrite, ...]
    membrane_writes: Tuple[ProgWrite, ...]

    @property
    def total_writes(self) -> int:
        return (len(self.threshold_writes) + len(self.synapse_writes)
                + len(self.membrane_writes))


def gate_write(*, on: bool) -> ProgWrite:
    """The GATE register write: activity is gated while the memories are written."""
    return ProgWrite(PROG_SEL_REGISTER, REGISTER_GATE, int(bool(on)))


def encode_weight(value: int, *, weight_bits: int) -> int:
    """One synapse cell: two's complement in ``weight_bits`` bits."""
    limit = 1 << (weight_bits - 1)
    if not -limit <= int(value) < limit:
        raise VariantPackError(
            f"weight {value} does not fit a {weight_bits}-bit two's-complement "
            f"synapse cell (range [{-limit}, {limit - 1}]); re-quantize against "
            f"the declared width rather than truncating.")
    return int(value) & ((1 << weight_bits) - 1)


def decode_weight(cell: int, *, weight_bits: int) -> int:
    """The inverse of :func:`encode_weight`, so the packing round-trips."""
    mask = (1 << weight_bits) - 1
    raw = int(cell) & mask
    sign = 1 << (weight_bits - 1)
    return raw - (1 << weight_bits) if raw & sign else raw


def pack_synapse_words(matrix: Any, *, spec: CoreSpec) -> Tuple[int, ...]:
    """Pack a signed logical crossbar into the core's 32-bit synapse words.

    ``matrix`` is ``(slots, neurons)`` and may be SMALLER than the declared
    geometry: unmapped rows and columns hold zero, which is the no-op weight.
    Allocates fresh arrays — ``HardCore.get_core_matrix()`` hands back a shared
    memoized grid callers must never mutate.
    """
    values = np.asarray(matrix)
    if values.ndim != 2:
        raise VariantPackError(
            f"the logical crossbar must be 2-D (slots x neurons), got shape "
            f"{values.shape}")
    rounded = np.rint(values.astype(np.float64))
    if not np.allclose(values.astype(np.float64), rounded, atol=0.0):
        raise VariantPackError(
            "the logical crossbar must be integral before packing; quantization "
            "delivers integer weights and a silent truncation would change the "
            "deployed physics")
    signed = rounded.astype(np.int64)
    slots, neurons = int(signed.shape[0]), int(signed.shape[1])
    if slots > spec.max_axons or neurons > spec.max_neurons:
        raise VariantPackError(
            f"a {slots}x{neurons} logical grid does not fit the declared "
            f"{spec.max_axons}x{spec.max_neurons} generated core")
    words = [0] * spec.synapse_depth
    cells = spec.cells_per_word
    for slot in range(slots):
        base = slot * spec.synapse_words_per_row
        for neuron in range(neurons):
            cell = encode_weight(
                int(signed[slot][neuron]), weight_bits=spec.weight_bits)
            if not cell:
                continue
            words[base + neuron // cells] |= cell << (
                (neuron % cells) * spec.weight_bits)
    return tuple(words)


def unpack_synapse_words(
    words: Sequence[int], *, spec: CoreSpec
) -> Tuple[Tuple[int, ...], ...]:
    """The inverse of :func:`pack_synapse_words` over the full declared geometry."""
    if len(words) != spec.synapse_depth:
        raise VariantPackError(
            f"the synapse image is {spec.synapse_depth} words, got {len(words)}")
    cells = spec.cells_per_word
    mask = (1 << spec.weight_bits) - 1
    grid: List[Tuple[int, ...]] = []
    for slot in range(spec.max_axons):
        base = slot * spec.synapse_words_per_row
        row = []
        for neuron in range(spec.max_neurons):
            raw = (words[base + neuron // cells]
                   >> ((neuron % cells) * spec.weight_bits)) & mask
            row.append(decode_weight(raw, weight_bits=spec.weight_bits))
        grid.append(tuple(row))
    return tuple(grid)


def silent_threshold(spec: CoreSpec) -> int:
    """The threshold an UNMAPPED neuron carries: the highest representable one.

    The arrays have no reset, so every neuron must be programmed; a neuron left
    at theta=0 would fire on the inclusive compare against its own zero
    membrane and inject spikes the mapping never produced.
    """
    return spec.membrane_high


def build_variant_core_image(
    core: Any, *, spec: CoreSpec, core_index: int, theta: int, membrane_init: int
) -> VariantCoreImage:
    """Pack one ``HardCore`` into the generated core's programming payload."""
    require_generatable(spec)
    logical_axons = int(core.axons_per_core)
    logical_neurons = int(core.neurons_per_core)
    if len(core.axon_sources) != logical_axons:
        raise VariantPackError(
            f"core {core_index}: {len(core.axon_sources)} axon sources for "
            f"{logical_axons} slots — the positional pairing of weight row a "
            f"with axon_sources[a] is what the whole chain rests on")
    if logical_axons > spec.max_axons or logical_neurons > spec.max_neurons:
        raise VariantPackError(
            f"core {core_index}: a {logical_axons}x{logical_neurons} mapping "
            f"does not fit the declared {spec.max_axons}x{spec.max_neurons} "
            f"generated core")
    matrix = np.asarray(core.get_core_matrix())
    check_weight_magnitudes(
        np.rint(matrix.astype(np.float64)).astype(np.int64),
        weight_bits=spec.weight_bits,
        weight_sign_granularity=spec.weight_sign_granularity,
        core_index=core_index,
    )
    used_neurons = logical_neurons - int(core.available_neurons or 0)
    silent = silent_threshold(spec)
    thresholds = tuple(
        ProgWrite(PROG_SEL_THRESHOLD, neuron,
                  int(theta) if neuron < used_neurons else silent)
        for neuron in range(spec.max_neurons)
    )
    membranes = tuple(
        ProgWrite(PROG_SEL_MEMBRANE, neuron,
                  _membrane_word(membrane_init if neuron < used_neurons else 0, spec))
        for neuron in range(spec.max_neurons)
    )
    synapses = tuple(
        ProgWrite(PROG_SEL_SYNAPSE, address, word)
        for address, word in enumerate(pack_synapse_words(matrix, spec=spec))
    )
    return VariantCoreImage(
        core_index=core_index, theta=int(theta),
        threshold_writes=thresholds, synapse_writes=synapses,
        membrane_writes=membranes,
    )


def _membrane_word(value: int, spec: CoreSpec) -> int:
    """The membrane state field, two's complement when the register is signed."""
    if not spec.membrane_low <= int(value) <= spec.membrane_high:
        raise VariantPackError(
            f"membrane init {value} is outside the register's interval "
            f"[{spec.membrane_low}, {spec.membrane_high}]")
    return int(value) & ((1 << spec.membrane_bits) - 1)
