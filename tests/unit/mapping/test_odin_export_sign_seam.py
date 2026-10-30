"""[U-H seam] Two declarations of one sign placement: the release head's per-core-type field and the ODIN fabric claim.

The release head declares a per-axon core type in PHYSICAL rows (``cores[i].weight_sign_granularity``): the mapper
splits every input into one row per sign (``rows_per_input == physical_row_expansion``) and the packed hard core
carries ``weight_sign_granularity = per_axon``. The ODIN backend declares the same fabric in LOGICAL slots (the
platform's ``weight_sign_granularity``): every slot is one signed row of the mapping, and its exporter places the row
pair itself (``RowPairExpansion``). Exporting a mapper-split core would split it twice, programming four rows per
input on a 256-row core sized for two. Until the exporter maps a mapper-split core one to one (an open item), the
export refuses such a core by name, before any image is built; a logical-slot core exports as before.
"""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.export.odin.exporter import OdinExportError, export_odin
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.softcore import HardCoreMapping
from mimarsinan.mapping.platform.platform_constraints import PER_AXON

from unit.mapping.test_odin_exporter import ODIN_LAW, _core


def _mapping(*, mapper_split: bool):
    core = _core(np.array([[2.0, 0.0], [0.0, -3.0]]), threshold=4.0,
                 sources=[SpikeSource(-2, 0, is_input=True), SpikeSource(-2, 1, is_input=True)])
    if mapper_split:
        core.weight_sign_granularity = PER_AXON   # what a per-axon core type's packed cores carry
    mapping = HardCoreMapping(chip_cores=[])
    mapping.cores = [core]
    mapping.output_sources = [SpikeSource(0, 0), SpikeSource(0, 1)]
    ChipLatency(mapping).calculate()
    return mapping


def _export(mapping):
    return export_odin(mapping, soma_law=ODIN_LAW, weight_bits=4, weight_sign_granularity="per_axon",
                       effective_max_axons=127, membrane_init=0)


def test_a_logical_slot_core_exports_as_before():
    assert len(_export(_mapping(mapper_split=False)).cores) == 1


def test_a_mapper_split_core_is_refused_by_name_before_any_image():
    with pytest.raises(OdinExportError, match=r"core 0.*cores\[i\]\.weight_sign_granularity") as refusal:
        _export(_mapping(mapper_split=True))
    assert "twice" in str(refusal.value)


def test_the_refusal_is_a_toolchain_refusal():
    from mimarsinan.common.refusal import ToolchainRefusal

    with pytest.raises(ToolchainRefusal):
        _export(_mapping(mapper_split=True))
