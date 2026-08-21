"""The generated core's descriptor: the spec, the law, and the port conventions."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from mimarsinan.mapping.export.odin_gen.feasibility import SaturationBound
from mimarsinan.mapping.export.odin_gen.packer import (
    PROG_SEL_MEMBRANE,
    PROG_SEL_REGISTER,
    PROG_SEL_SYNAPSE,
    PROG_SEL_THRESHOLD,
    REGISTER_GATE,
)
from mimarsinan.mapping.export.odin_gen.render import CORE_MODULE, spec_flags_word
from mimarsinan.mapping.export.odin_gen.spec import CoreSpec

DESCRIPTOR_SCHEMA_VERSION = 1


def build_descriptor(
    spec: CoreSpec, *, files: Sequence[str], vendored: bool,
    saturation_bounds: Tuple[SaturationBound, ...] = (),
) -> Dict[str, Any]:
    """Everything a consumer needs to build, program and read this core.

    Echoes only values the spec or a gate already established: the descriptor
    is evidence, not a second declaration.
    """
    return {
        "schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "spec_key": spec.spec_key(),
        "module": None if vendored else CORE_MODULE,
        "vendored": bool(vendored),
        "files": list(files),
        "geometry": {
            "core_type": spec.core_type(),
            "weight_bits": spec.weight_bits,
            "weight_sign_granularity": spec.weight_sign_granularity,
            "physical_row_factor": spec.physical_row_factor,
            "axon_address_bits": spec.axon_address_bits,
            "neuron_address_bits": spec.neuron_address_bits,
            "synapse_words_per_row": spec.synapse_words_per_row,
            "synapse_depth": spec.synapse_depth,
            "cells_per_word": spec.cells_per_word,
        },
        "soma_law": {
            "firing_mode": spec.soma_law.firing_mode,
            "thresholding_mode": spec.soma_law.thresholding_mode,
            "firing_granularity": spec.soma_law.firing_granularity,
            "membrane_arithmetic": spec.soma_law.membrane_arithmetic,
            "membrane_bits": spec.membrane_bits,
            "membrane_signed": spec.membrane_signed,
            "membrane_interval": [spec.membrane_low, spec.membrane_high],
            "bias_slot": spec.soma_law.bias_slot,
            "point_tag": spec.soma_law.point_tag(),
            "asserts_no_saturation": spec.asserts_no_saturation,
        },
        "programming": _programming_conventions(),
        "spec_flags": spec_flags_word(spec),
        "feasibility": {
            "saturation_bounds": [
                {"core_index": bound.core_index, "lowest": bound.lowest,
                 "highest": bound.highest}
                for bound in saturation_bounds
            ],
        },
    }


def _programming_conventions() -> Dict[str, Any]:
    """The direct synchronous write port, named once for every consumer."""
    return {
        "port": "direct_synchronous",
        "selectors": {
            "register": PROG_SEL_REGISTER,
            "threshold": PROG_SEL_THRESHOLD,
            "membrane": PROG_SEL_MEMBRANE,
            "synapse": PROG_SEL_SYNAPSE,
        },
        "registers": {"gate_activity": REGISTER_GATE},
        "aer_in": {
            "time_reference_bit": "AW",
            "axon_field": "AERIN_ADDR[AW-1:0]",
        },
        "aer_out": {"address": "the neuron that fired"},
    }
