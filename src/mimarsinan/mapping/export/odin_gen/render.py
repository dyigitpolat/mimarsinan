"""The ``CoreSpec`` -> Verilog substitution table: one place, checked both ways."""

from __future__ import annotations

from typing import Dict

from mimarsinan.mapping.export.odin_gen.spec import CoreSpec, require_generatable
from mimarsinan.mapping.export.odin_gen.templates import render

CORE_TEMPLATE = "odin_gen_core.v"
CORE_MODULE = "odin_gen_core"


def substitution_table(spec: CoreSpec) -> Dict[str, object]:
    """Every parameter the core template declares, derived from the spec alone."""
    require_generatable(spec)
    return {
        "AXONS": spec.max_axons,
        "NEURONS": spec.max_neurons,
        "MBITS": spec.membrane_bits,
        "MSIGNED": int(spec.membrane_signed),
        "WBITS": spec.weight_bits,
        "PER_EVENT": int(spec.per_event),
        "RESET_ZERO": int(spec.reset_zero),
        "CMP_INCL": int(spec.compare_inclusive),
        "ASSERT_NO_SAT": int(spec.asserts_no_saturation),
        "AW": spec.axon_address_bits,
        "NW": spec.neuron_address_bits,
        "NWM1": spec.neuron_address_bits - 1,
        "CELLS_PER_WORD_LOG2": _log2(spec.cells_per_word),
        "SYN_WORDS_PER_ROW": spec.synapse_words_per_row,
        "SYN_DEPTH": spec.synapse_depth,
        "SYN_ADDR_BITS": spec.synapse_address_bits,
        "V_LO": spec.membrane_low,
        "V_HI": spec.membrane_high,
    }


def render_core_rtl(spec: CoreSpec) -> str:
    """The variant core's Verilog, expanded from ``hw/gen`` for this spec."""
    return render(CORE_TEMPLATE, substitution_table(spec))


def core_filename(spec: CoreSpec) -> str:
    """The emitted file's name; the MODULE name is fixed so a testbench binds it."""
    return f"{CORE_MODULE}.v"


def spec_flags_word(spec: CoreSpec) -> int:
    """``SPEC_FLAGS`` as the generated core drives it — the testbench's check.

    Bit order is the concatenation in the template, LSB first:
    MSIGNED, PER_EVENT, RESET_ZERO, CMP_INCL, ASSERT_NO_SAT.
    """
    bits = (
        int(spec.membrane_signed),
        int(spec.per_event),
        int(spec.reset_zero),
        int(spec.compare_inclusive),
        int(spec.asserts_no_saturation),
    )
    word = 0
    for index, bit in enumerate(bits):
        word |= bit << index
    return word


def _log2(value: int) -> int:
    return int(value).bit_length() - 1
