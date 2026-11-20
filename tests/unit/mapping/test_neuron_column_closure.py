"""[U-LIVE r2] The per-neuron column rule as a closure by construction (the conductor's rulings of 2026-11-13, (a) and
(e)): one function carries every value a core holds per neuron through every site that drops, selects or reorders
neurons, and a test over src's syntax trees (``neuron_column_sites``) fails when a site takes a per-neuron value at
some of its columns by itself, or when a value a core holds per neuron is not declared, so not carried.

``READERS`` are the reviewed functions that take a per-neuron value at columns and drop, select and reorder no core's
neurons (each reads or writes a core's own neurons by its own index, or builds a new core from the same columns as its
matrix), each with its read. A site that cuts a core's neurons (it stores a matrix cut along its neuron axis, or a
compaction's kept columns) calls the one function, or is reviewed in ``CUTS_REVIEWED``. ``NOT_PER_NEURON`` are the
reviewed attributes a census of built programs finds holding one entry per neuron of their core without being values
the core's neurons read (masks and index lists of a cut), each with its read; every other one must be declared per
neuron, so that ``take_neuron_columns`` carries it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import unit.mapping.neuron_column_sites as ncs

SRC = Path(__file__).resolve().parents[3] / "src"

#: Functions that take a per-neuron value at columns and drop, select and reorder no neuron of a core, each read.
READERS = {
    "mimarsinan.certification.program_check.check:_value_levels":
        "reads a placement's columns of a hard core's readout law (placement_value, the declared layout)",
    "mimarsinan.chip_simulation.lava_loihi.segment_runner:LavaSegmentMixin._run_neural_segment_scheduled":
        "reads a hard core's used neurons (a prefix of its own columns) of its bias and registers",
    "mimarsinan.chip_simulation.sanafe.net_synth.build:build_network_for_segment":
        "reads each used neuron of a hard core (its bias entry and its register of the core's column)",
    "mimarsinan.chip_simulation.sanafe.net_synth.spike_trains:apply_ttfs_preset_membranes":
        "reads each used neuron's bias entry of a hard core",
    "mimarsinan.chip_simulation.ttfs.segment_arrays:_segment_ttfs_arrays_from_mapping_uncached":
        "reads a hard core's used neurons of its bias",
    "mimarsinan.mapping.export.chip_export:hard_cores_to_chip":
        "exports each neuron of a hard core (its bias entry and register by its own index)",
    "mimarsinan.mapping.lanes.ir.reduce_cores:reduced_lane_cores":
        "builds a lane's reduction core: each new neuron sums the bias of the producer columns its matrix blocks "
        "read (the same columns); lanes fuse at mapping, before any per-neuron register or law exists, and a "
        "per-neuron register is refused by name (core_threshold)",
    "mimarsinan.mapping.packing.softcore.hard_core:HardCore.add_softcore":
        "packing: writes a soft core's bias over its own neuron range of the hard core (the declared per-neuron "
        "names by place_neuron_value, the same range)",
    "mimarsinan.mapping.pruning.certificate.seed_reference:apply_admitted_seed_zeroing":
        "zeroes the bias at a seed's admitted columns of its own core (a certificate's reference); no neuron dropped",
    "mimarsinan.models.spiking.hybrid.segment_cache:build_segment_tensors":
        "reads a hard core's used neurons (a prefix of its own columns) of its bias and registers",
}

#: Functions that store a core's kept columns or a column cut of its matrix without the one function, each read.
CUTS_REVIEWED = {
    "mimarsinan.mapping.packing.softcore.sign_split:split_soft_core":
        "re-describes a core by its physical rows (a mixed row in two signed rows): it keeps every column it holds",
    "mimarsinan.mapping.packing.softcore.soft_core:SoftCore.__init__": "a new soft core's descriptor starts empty",
}

#: Attributes a census finds holding one entry per neuron of their core that no neuron reads as its value, each read.
NOT_PER_NEURON = {
    "pruned_col_mask": "the cut's own mask: which neurons a compaction drops (it describes the matrix before the cut; "
                       "a length disagreement compacts nothing, compaction_masks_apply)",
    "pre_pruning_col_mask": "IR pruning's record of the mask it cut by (the columns before the cut)",
    "compact_keep_cols": "a compacted core's kept columns of its source matrix (the cut's own index)",
}


@pytest.fixture(scope="module")
def analysis() -> ncs.Analysis:
    return ncs.Analysis(SRC)


def test_every_site_taking_per_neuron_columns_is_the_one_function_or_a_reviewed_reader(analysis):
    found = analysis.sites()
    expected = {ncs.ONE_FUNCTION} | set(READERS)
    assert found == expected, (sorted(found - expected), sorted(expected - found))


def test_every_site_cutting_a_core_s_neurons_carries_its_per_neuron_values(analysis):
    found = analysis.uncarried_cuts()
    assert found == set(CUTS_REVIEWED), (sorted(found - set(CUTS_REVIEWED)), sorted(set(CUTS_REVIEWED) - found))


def test_the_takers_list_holds_every_function_that_takes_a_per_neuron_value_at_columns(analysis):
    assert analysis.takers() <= ncs.TAKERS, sorted(analysis.takers() - ncs.TAKERS)


def test_the_one_function_carries_what_each_holder_declares(analysis):
    assert analysis.one_function_reads_the_declarations()
    assert analysis.names == {"threshold", "hardware_bias", "readout_law", "pruned_row_magnitude", "lane_outputs",
                              "lane_reduced_instances", "lane_operand_rows"}


# ---------------------------------------------------------------------------------------------------------------------
# the detector's own test: the shapes a new site can take are red
# ---------------------------------------------------------------------------------------------------------------------
DECLARATION = ("from mimarsinan.mapping.support.value_layouts import layouts\n"
               "class Core:\n"
               "    VALUE_LAYOUTS = layouts(PER_NEURON=('threshold', 'hardware_bias', 'readout_law'))\n")

SHAPES = {
    # a compaction that cuts a register column by itself
    "direct": ("def cut(core, keep):\n    core.threshold = core.threshold[keep]\n", {"cut"}),
    # through a name bound from the value, handed on by a cast
    "bound": ("import numpy as np\n"
              "def cut(core, keep):\n"
              "    hb = np.asarray(getattr(core, 'hardware_bias'))\n"
              "    core.hardware_bias = hb[keep]\n", {"cut"}),
    # a law's own columns taken beside a cut
    "law": ("def cut(core, keep):\n    law = core.readout_law\n    core.readout_law = law.take_columns(keep)\n",
            {"cut"}),
    # a listed taker called with a register column
    "taker": ("from mimarsinan.mapping.support.neuron_thresholds import threshold_columns\n"
              "def window(core, lo, hi):\n    return threshold_columns(core.threshold, lo, hi)\n", {"window"}),
    # the one function's own form copied: a loop over the declared names
    "declared_loop": ("from mimarsinan.mapping.support.value_layouts import ValueLayout, declared_names\n"
                      "def cut(core, keep):\n"
                      "    for name in declared_names(type(core), ValueLayout.PER_NEURON):\n"
                      "        setattr(core, name, getattr(core, name)[keep])\n", {"cut"}),
    # the names spelled out in a loop
    "named_loop": ("def cut(core, keep):\n"
                   "    for name in ('threshold', 'hardware_bias'):\n"
                   "        value = getattr(core, name)\n"
                   "        setattr(core, name, value[keep])\n", {"cut"}),
    # the name held by a module constant, read by getattr
    "module_constant": ("BIAS = 'hardware_bias'\n"
                        "def cut(core, keep):\n    value = getattr(core, BIAS)\n    setattr(core, BIAS, value[keep])\n",
                        {"cut"}),
    # a store into part of a column (a range written)
    "range_write": ("def place(hard, soft, lo, hi):\n    hard.threshold[lo:hi] = soft.threshold\n", {"place"}),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_detector_finds_each_shape_of_a_new_site(shape):
    text, expected = SHAPES[shape]
    analysis = ncs.Analysis(SRC, sources={"mimarsinan.newsite.decl": DECLARATION, "mimarsinan.newsite.module": text})
    assert analysis.sites() == {f"mimarsinan.newsite.module:{name}" for name in expected}


CUT_SHAPES = {
    # the defect's own shape: a column compaction that cuts the matrix and leaves the registers whole
    "matrix": ("def cut(core, keep):\n    core.core_matrix = core.core_matrix[:, keep]\n", {"cut"}),
    # through a name bound from the matrix, cut, then stored
    "bound": ("def cut(node, keep):\n    mat = node.core_matrix\n    mat = mat[:, keep]\n"
              "    node.core_matrix = mat\n", {"cut"}),
    # a compaction descriptor's kept columns
    "descriptor": ("def cut(core, cols):\n    core.compact_source_matrix = core.core_matrix\n"
                   "    core.compact_keep_cols = cols\n    core.core_matrix = None\n", {"cut"}),
    # rows only: no neuron cut
    "rows": ("def cut(core, rows):\n    core.core_matrix = core.core_matrix[rows, :]\n", set()),
    # the cut carried by the one function
    "carried": ("from mimarsinan.mapping.packing.softcore.compaction import take_neuron_columns\n"
                "def cut(core, keep):\n    take_neuron_columns(core, keep)\n"
                "    core.core_matrix = core.core_matrix[:, keep]\n", set()),
}


@pytest.mark.parametrize("shape", sorted(CUT_SHAPES))
def test_the_detector_finds_each_shape_of_a_cut_that_carries_nothing(shape):
    text, expected = CUT_SHAPES[shape]
    analysis = ncs.Analysis(SRC, sources={"mimarsinan.newsite.decl": DECLARATION, "mimarsinan.newsite.module": text})
    assert analysis.uncarried_cuts() == {f"mimarsinan.newsite.module:{name}" for name in expected}


def test_a_helper_that_cuts_a_value_handed_to_it_is_a_new_taker():
    text = ("def helper(values, keep):\n    return values[keep]\n"
            "def cut(core, keep):\n    core.threshold = helper(core.threshold, keep)\n")
    analysis = ncs.Analysis(SRC, sources={"mimarsinan.newsite.decl": DECLARATION, "mimarsinan.newsite.module": text})
    assert analysis.takers() == {"helper"} and not analysis.takers() <= ncs.TAKERS


def test_a_reader_of_a_whole_value_is_no_site():
    text = "import numpy as np\ndef total(core):\n    return float(np.sum(core.threshold)) + float(core.hardware_bias)\n"
    analysis = ncs.Analysis(SRC, sources={"mimarsinan.newsite.decl": DECLARATION, "mimarsinan.newsite.module": text})
    assert analysis.sites() == set()


# ---------------------------------------------------------------------------------------------------------------------
# the census: every value a core holds per neuron is declared, so carried
# ---------------------------------------------------------------------------------------------------------------------
def _per_neuron_entries(holder, neurons: int, axons: int):
    """The attributes of ``holder`` holding one entry per neuron (not as many as its axons): arrays, tensors and lists
    of numbers, and objects carrying their own columns (a tuple is a range or a shape, never a value per neuron)."""
    for name, value in vars(holder).items():
        if hasattr(value, "take_columns"):
            yield name
            continue
        if isinstance(value, torch.Tensor):
            size, ndim = int(value.numel()), value.dim()
        elif isinstance(value, np.ndarray):
            size, ndim = int(value.size), value.ndim
        elif isinstance(value, list) and value and all(isinstance(v, (bool, int, float, np.number)) for v in value):
            size, ndim = len(value), 1
        else:
            continue
        if ndim == 1 and size == neurons and size != axons and neurons > 1:
            yield name


def _census(ir, hybrid) -> dict:
    from mimarsinan.mapping.ir import NeuralCore
    from mimarsinan.mapping.ir.legacy_convert import neural_core_to_soft_core
    from mimarsinan.mapping.support.value_layouts import ValueLayout, declared_names

    found: dict = {}

    def visit(holder, neurons: int, axons: int) -> None:
        declared = {name for layout in ValueLayout for name in declared_names(type(holder), layout)}
        for name in _per_neuron_entries(holder, neurons, axons):
            if name not in declared and name not in NOT_PER_NEURON:
                found.setdefault(type(holder).__name__, set()).add(name)

    for node in ir.get_neural_cores():
        assert isinstance(node, NeuralCore)
        visit(node, int(node.get_output_count()), len(node.input_sources.flatten()))
        soft = neural_core_to_soft_core(node, ir)
        soft.threshold = node.threshold
        visit(soft, int(soft.get_output_count()), int(soft.get_input_count()))
    for stage in hybrid.stages:
        hcm = getattr(stage, "hard_core_mapping", None)
        for core in (hcm.cores if hcm is not None else ()):
            visit(core, int(core.neurons_per_core), int(core.axons_per_core))
    return found


@pytest.mark.parametrize("family", ("count", "timing"))
def test_every_array_a_core_holds_per_neuron_is_declared(family, monkeypatch):
    """Built programs of both families under per-neuron grids (registers per neuron, a bias per neuron): every value
    an IR node, a soft core or a hard core holds with one entry per neuron is declared by its class (per neuron, or per
    channel where its entries index the perceptron's channels), or reviewed."""
    import unit.spiking.host_gather_forms as hg
    import unit.spiking.position_fold_forms as pf

    import mimarsinan.transformations.normalization_aware_perceptron_quantization as napq_module

    class _PerNeuron(napq_module.NormalizationAwarePerceptronQuantization):
        def __init__(self, *args, **kwargs):
            kwargs["per_neuron"] = True
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(napq_module, "NormalizationAwarePerceptronQuantization", _PerNeuron)
    monkeypatch.setattr(hg, "NormalizationAwarePerceptronQuantization", _PerNeuron)
    if family == "count":
        program = pf.count_program("tree481", 1, "streamed_lif", "one", wq=True)
        ir, hybrid = program.ir, program.hybrid
    else:
        read = pf.timing_read("tree481", 1, "one", wq=True)
        from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping

        ir = read["ir"]
        hybrid = build_identity_hybrid_mapping(ir_graph=ir)
    assert any(np.ndim(n.threshold) == 1 for n in ir.get_neural_cores()), "the program holds per-neuron registers"
    assert _census(ir, hybrid) == {}


def test_the_census_finds_an_undeclared_per_neuron_value():
    from mimarsinan.mapping.packing.softcore import SoftCore

    core = SoftCore(core_matrix=np.ones((2, 3)), axon_sources=[SimpleNamespace()] * 2, id=0)
    core.gauges_per_neuron = np.arange(3.0)  # pyright: ignore[reportAttributeAccessIssue]
    assert list(_per_neuron_entries(core, 3, 2)) == ["gauges_per_neuron"]
