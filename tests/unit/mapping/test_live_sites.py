"""[U-LIVE] The one liveness predicate as a closure by construction: one function, a finite list of sites, and a test
over src's syntax trees that fails when a site decides an axon's liveness by itself, also where the decision is split
over two functions (``live_sites``: a site is defined by what it reads, closed over the call graph).

``SITES`` are the functions that call the predicate (``mapping/latency/upstream.py``): every scheduling rule reads an
axon's liveness through one of them. ``OUTSIDE`` are the reviewed readers of an axon's marks or a core's weights that
the schedule's closure reaches and that decide no axon's liveness for a schedule, each with its read. ``HANDOVER`` are
the reviewed functions outside the closure that read a mark or a weight and store an attribute the closure loads,
each with its read. A new function that reads a mark or a weight inside the schedule's closure, a new hand-over, or a
new caller of the predicate, is red until it is reviewed onto one of the lists.

The facts' two finite lists (``live_sites.MARK_READS``, ``live_sites.WEIGHT_READS``) are held beside the searches that
find every method of src returning a source's marks and every method or attribute handing over a core's matrix; a
name a search finds is either on its list or reviewed here (``MARK_RETURNS_REVIEWED``, ``MATRIX_RETURNS_REVIEWED``),
each with its read.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import unit.mapping.live_sites as ls

SRC = Path(__file__).resolve().parents[3] / "src"

#: Every caller of the one predicate, with the scheduling rule it serves.
SITES = {
    "mimarsinan.mapping.latency.chip:ChipLatency._reads":
        "ChipLatency: the sweep, the shift alignment, the consumer map and the per-neuron delays",
    "mimarsinan.mapping.latency.chip:ChipLatency.get_delay_for": "ChipLatency: an output source's direct signal",
    "mimarsinan.mapping.latency.chip:ChipLatency.calculate": "ChipLatency: the output sources' direct signals",
    "mimarsinan.mapping.latency.ir:IRLatency._get_depth": "IRLatency: the IR's tiers and the record's ir_max_latency",
    "mimarsinan.mapping.latency.depth_balancing:_live_axon_flags": "the relay planner",
    "mimarsinan.mapping.latency.cycle_windows:_segment_reads": "the window rule: the dead-read pass and ir_window_gaps",
    "mimarsinan.mapping.latency.cycle_windows:core_window_gaps": "the window rule: the per-cycle executor's assertion",
    "mimarsinan.mapping.latency.cycle_windows:carried_window_gaps": "the window rule: a pass's carried slices",
    "mimarsinan.mapping.packing.pass_timing:_Placed.read":
        "the pass projection: fused and own latencies, pins and delivery delays",
    "mimarsinan.mapping.packing.pass_timing:_reads_of": "the pass projection's delivery delays on a placed view",
    "mimarsinan.mapping.packing.pass_timing:late_stage_input_readers": "the stage-input timing rule (W33)",
    "mimarsinan.certification.program_check.unbounded.latency:identity_timing":
        "the static check's model of the schedule",
    "mimarsinan.certification.program_check.tiles:_facts": "the program check's live rows (H1-latency, R6)",
}

#: Readers of an axon's marks or a core's weights in the schedule's closure that decide no axon's liveness, each read.
OUTSIDE = {
    "mimarsinan.chip_simulation.odin_hacc.pass_build:pass_mapping":
        "builds a pass's hard-core mapping (its sources and weights); its schedule is ChipLatency's",
    "mimarsinan.chip_simulation.odin_rtl.reference:route_of_source": "routes a source's events by its kind",
    "mimarsinan.mapping.export.odin.feasibility:entry_event_bound": "a value bound of an entry core's events",
    "mimarsinan.mapping.export.odin.feasibility:propagate_emission_bounds": "value bounds of emissions",
    "mimarsinan.mapping.export.odin.images:bias_row_count": "counts the always-on rows of an exported image",
    "mimarsinan.mapping.export.odin.images:build_core_image": "writes a core's weights into an exported image",
    "mimarsinan.mapping.export.odin_gen.packer:build_variant_core_image": "writes a core's weights into an image",
    "mimarsinan.mapping.ir.legacy_convert:ir_graph_to_soft_core_mapping":
        "converts the IR to soft cores; copies each node's latency",
    "mimarsinan.mapping.ir.legacy_convert:ir_source_to_spike_source": "converts a source's kind",
    "mimarsinan.mapping.ir.legacy_convert:neural_core_to_soft_core": "converts a node's sources and weights",
    "mimarsinan.mapping.latency.unread:drop_unread":
        "the removal's reader index (a source that is not off reads its neuron; after the dead-read sweep every "
        "in-segment read that is not off is live) and the matrix's shape for the compaction masks",
    "mimarsinan.mapping.latency.unread:_remove_unread": "the removal's reader index (as drop_unread)",
    "mimarsinan.mapping.latency.depth_balancing:find_intra_segment_depth_gaps":
        "compares two depths (a gap above 1) over the edges the relay planner's predicate keeps; the node ids are "
        "the depths' keys",
    "mimarsinan.mapping.latency.depth_balancing:assert_relays_alive": "a relay's identity weight against its threshold",
    "mimarsinan.mapping.packing.neural_segment_packing:neural_segment_to_soft_core_mapping":
        "converts a segment to soft cores; copies each node's latency",
    "mimarsinan.mapping.packing.softcore.compaction:_ir_keep_indices": "keeps the always-on row of a compacted node",
    "mimarsinan.mapping.packing.softcore.fragments:build_real_fragment": "a fragment's sources; copies its latency",
    "mimarsinan.mapping.packing.softcore.hard_core:HardCore._softcore_placement": "places a soft core's weights",
    "mimarsinan.mapping.packing.softcore.hard_core:HardCore.add_softcore":
        "places a soft core's sources and weights; copies its latency",
    "mimarsinan.mapping.support.spike_source_columns:SpikeSourceColumns.of": "encodes sources' kinds as columns",
    "mimarsinan.mapping.support.spike_source_columns:SpikeSourceColumns.from_ir_sources":
        "encodes IR sources' kinds as columns (as ir_source_to_spike_source, source by source)",
    "mimarsinan.code_generation.cpp_chip_model_types:SpikeSource.__init__": "builds a source from its given kind",
}

#: Functions outside the schedule's closure that read a mark or a weight and store an attribute a function of the
#: closure loads, each read: every one writes the program the predicate reads (its weights, its wiring, its masks), and
#: none hands a schedule a decision on which axons count.
HANDOVER = {
    "mimarsinan.certification.randomized.program:_rekey_residency_classes":
        "re-keys the layout specs' residency classes from each node's residency key (its bank); writes "
        "layout_softcores, no wiring",
    "mimarsinan.mapping.export.chip_quantize:quantize_ir_graph":
        "writes the quantized weights, thresholds and scales of every core and bank (Weight Quantization's program)",
    "mimarsinan.mapping.ir.graph:IRGraph.__getattr__": "the empty defaults of an unpickled graph's missing fields",
    "mimarsinan.mapping.ir.graph:IRGraph.remove_nodes":
        "deletes nodes and rewires their readers to off: the program's wiring the predicate reads",
    "mimarsinan.mapping.lanes.ir.operand_rows:_banked": "inserts a bank's operand rows: weights and sources written",
    "mimarsinan.mapping.lanes.ir.operand_rows:_owned": "inserts an owned core's operand rows: weights and sources written",
    "mimarsinan.mapping.lanes.ir.reduction_banks:bank_ir_cores": "moves reduction cores' matrices into shared banks",
    "mimarsinan.mapping.mappers.operand.emission:emit_operand_product":
        "emits an operand product's write op and bank instances; its producers order the write op's hoist",
    "mimarsinan.mapping.packing.hybrid_build_pool:_build_single_pool":
        "builds the pool's stages (packing and placement); hands the banks to the packer",
    "mimarsinan.mapping.packing.hybrid_build_scheduled:_build_scheduled":
        "builds the scheduled stages (packing and placement); hands the banks to the packer",
    "mimarsinan.mapping.packing.hybrid_segment_helpers:_remap_external_sources_to_segment_inputs":
        "rewires a segment's external reads to its inputs: the program's wiring",
    "mimarsinan.mapping.packing.schedule_bank_clustered:dedup_resident_stage_matrices":
        "aliases equal resident matrices for storage; no value changes",
    "mimarsinan.mapping.packing.softcore.compaction:compact_soft_core_mapping":
        "compacts each soft core at its kept rows and columns: weights, sources and per-neuron values written",
    "mimarsinan.mapping.packing.softcore.hard_core_mapping:HardCoreMapping._finalize_sources":
        "remaps placed sources to (core, neuron) and pads unused axons off: the program's wiring",
    "mimarsinan.mapping.packing.softcore.hard_core_mapping:HardCoreMapping.map":
        "places soft cores on hard cores; copies the banks",
    "mimarsinan.mapping.packing.softcore.hard_core_mapping:HardCoreMapping.map_identity":
        "places each soft core on its own hard core; copies the banks",
    "mimarsinan.mapping.packing.softcore.sign_split:split_soft_core":
        "re-describes a core by its physical rows (a mixed row in two signed rows): weights and sources written",
    "mimarsinan.mapping.pruning.certificate.dyadic_grid:snap_ir_graph_to_dyadic_grid":
        "a certification twin with weights, biases and thresholds snapped to a dyadic grid",
    "mimarsinan.mapping.pruning.graph.constant_folding:apply_constant_folds":
        "writes the constant folds' deltas into a row or the bias",
    "mimarsinan.mapping.pruning.ir_pruning_compact:_attach_bank_metadata":
        "projects a bank's pruned rows and columns onto its nodes' masks",
    "mimarsinan.mapping.pruning.ir_pruning_compact:_compact_node":
        "IR pruning's compaction: drops a node's pruned columns and rows (weights, sources, per-neuron values)",
    "mimarsinan.mapping.pruning.ir_pruning_compact:_reset_post_compaction_masks":
        "re-derives a compacted node's masks at its new shape",
    "mimarsinan.mapping.pruning.ir_pruning_helpers:_attach_pre_compaction_metadata":
        "stores the pre-pruning heatmap and the pruned masks of owned nodes",
    "mimarsinan.mapping.pruning.ir_pruning_helpers:_rewire_sources":
        "rewires readers of pruned neurons to off and reindexes the kept ones: the program's wiring",
}

#: Methods the marks search finds that hand no source's marks over by name, each read.
MARK_RETURNS_REVIEWED = {
    "__reduce__": "a column source's pickle reducer: its constructor's arguments, read only by pickle",
    "absorb": "a lane walk's absorbed op: returns its output sources (off windows dropped), not marks",
    "block": "the program check's canonical fan-in of a node by source code (two IRs compared for the certificate); "
             "by name it would be every model's block",
    "of": "builds a source sequence's columnar form; its marks are read back through source_arrays (listed)",
}

#: Methods or attributes the matrix search finds that hand no core's matrix over by name, each read.
MATRIX_RETURNS_REVIEWED = {
    "__getattr__": "an unpickled graph's empty weight banks, reached only by loading weight_banks itself (listed)",
}

#: Listed names the searches do not find (they search methods and attributes), each read: so that a list holds no stale
#: entry and a search that finds less is red.
LISTED_UNSEARCHED = {
    "weight_banks": "the IR's store of shared weight banks (a dictionary of banks, each holding its core_matrix)",
    "source_arrays": "the predicate module's arrays of a source sequence's marks (a module function; also a columnar "
                     "sequence's method returning its stored kind columns)",
    "_source_arrays": "ChipLatency's module alias of source_arrays",
    "source_pairs": "the static check's producer node ids and indices of IR sources (a module function)",
}


@pytest.fixture(scope="module")
def analysis() -> ls.Analysis:
    return ls.Analysis(SRC)


def test_every_hand_over_through_an_attribute_is_reviewed(analysis):
    """A function outside the schedule's closure that reads a mark or a weight and stores an attribute the closure
    loads hands a decision over unless it is on the reviewed list: red for a new one, red for a stale entry."""
    found = analysis.handovers()
    assert found == set(HANDOVER), (sorted(found - set(HANDOVER)), sorted(set(HANDOVER) - found))


def test_the_weight_list_holds_every_name_a_core_matrix_is_handed_over_by(analysis):
    """Every method returning, and every attribute stored from, a core's matrix read through the list is on the list
    or reviewed: a new accessor of the weights under another name is red, and so is a listed name no search finds."""
    found = analysis.matrix_returners()
    assert found - ls.WEIGHT_READS == set(MATRIX_RETURNS_REVIEWED), (
        sorted(found - ls.WEIGHT_READS - set(MATRIX_RETURNS_REVIEWED)),
        sorted(set(MATRIX_RETURNS_REVIEWED) - (found - ls.WEIGHT_READS)))
    assert ls.WEIGHT_READS - found == set(LISTED_UNSEARCHED) & ls.WEIGHT_READS


def test_the_mark_list_holds_every_method_returning_a_source_s_marks(analysis):
    """Every method whose returned value is read from a source's marks is a listed mark read, a mark method, the
    predicate, or reviewed: a new hand-over of the marks under another name is red."""
    found = analysis.mark_returners() - ls.MARK_METHODS - ls.PREDICATE
    assert found - ls.MARK_READS == set(MARK_RETURNS_REVIEWED), (
        sorted(found - ls.MARK_READS - set(MARK_RETURNS_REVIEWED)),
        sorted(set(MARK_RETURNS_REVIEWED) - (found - ls.MARK_READS)))
    assert ls.MARK_READS - found == set(LISTED_UNSEARCHED) & ls.MARK_READS


def test_every_reader_of_a_mark_or_a_weight_in_the_schedule_is_reviewed(analysis):
    """A function inside the schedule's closure that reads an axon's mark or a core's weight decides liveness by
    itself unless it is on the reviewed list: red for a new one, red for a stale entry."""
    assert analysis.sites() == set(OUTSIDE), (sorted(analysis.sites() - set(OUTSIDE)),
                                             sorted(set(OUTSIDE) - analysis.sites()))


def test_the_predicate_has_exactly_the_listed_callers(analysis):
    callers = {n for n, u in analysis.units.items() if u.predicate and u.module != ls.PREDICATE_MODULE}
    assert callers == set(SITES), (sorted(callers - set(SITES)), sorted(set(SITES) - callers))


def test_no_listed_site_reads_a_mark_or_a_weight_itself(analysis):
    """A site reads an axon's liveness only through the predicate: no mark and no weight of its own."""
    for name in SITES:
        unit = analysis.units[name]
        assert not (unit.mark or unit.weight), name


def test_every_latency_engine_reaches_the_predicate(analysis):
    for name in ("mimarsinan.mapping.latency.chip:ChipLatency.calculate",
                 "mimarsinan.mapping.latency.ir:IRLatency.calculate",
                 "mimarsinan.mapping.latency.depth_balancing:insert_depth_balancing_relays",
                 "mimarsinan.mapping.latency.cycle_windows:silence_dead_segment_reads",
                 "mimarsinan.mapping.latency.cycle_windows:ir_window_gaps",
                 "mimarsinan.mapping.packing.pass_timing:_delay",
                 "mimarsinan.certification.program_check.unbounded.latency:identity_timing"):
        assert analysis.reaches_predicate(name), name


# ---------------------------------------------------------------------------------------------------------------------
# the detector's own test: the shapes a new site can take are red
# ---------------------------------------------------------------------------------------------------------------------
STUBS = {
    "mimarsinan.mapping.latency.core_graph": "def alap_latencies(s):\n    return s\n\n"
                                             "def longest_paths(s):\n    return s\n",
    "mimarsinan.mapping.latency.upstream": "def live_axons(m, off, outside):\n    return m\n\n"
                                           "def core_reads(core):\n    return core\n\n"
                                           "def source_arrays(sources):\n    return sources\n\n"
                                           "def iter_upstream_neural_ids(sources):\n    return sources\n",
}

SHAPES = {
    # a direct reader: the marks and the order in one function
    "direct": ("from mimarsinan.mapping.latency.core_graph import alap_latencies\n"
               "def order(cores):\n"
               "    return alap_latencies([[s.core_ for s in c.axon_sources if not s.is_off_] for c in cores])\n",
               {"order"}),
    # split: a helper reads the marks, the function that joins them reads the order
    "helper": ("from mimarsinan.mapping.latency.core_graph import alap_latencies\n"
               "def sources(core):\n"
               "    return [s.core_ for s in core.axon_sources if not s.is_input_]\n"
               "def order(cores):\n"
               "    return alap_latencies([sources(c) for c in cores])\n",
               {"sources"}),
    # split over siblings: the caller hands one helper's sources to another that reads the order
    "siblings": ("from mimarsinan.mapping.latency.core_graph import longest_paths\n"
                 "def sources(core):\n"
                 "    return [s.core_ for s in core.axon_sources if getattr(s, 'is_always_on_', False) is False]\n"
                 "def depth(srcs):\n"
                 "    return longest_paths(srcs)\n"
                 "def plan(cores):\n"
                 "    return depth([sources(c) for c in cores])\n",
                 {"sources"}),
    # a weight test in a method reached through self from a latency writer
    "method": ("import numpy as np\n"
               "class Engine:\n"
               "    def live(self, core):\n"
               "        return np.any(core.get_core_matrix() != 0, axis=1)\n"
               "    def run(self, cores):\n"
               "        for c in cores:\n"
               "            c.latency = int(self.live(c).sum())\n",
               {"Engine.live"}),
    # an IR read by its node index, in a function that writes a latency
    "index": ("def tiers(nodes):\n"
              "    for n in nodes:\n"
              "        n.latency = sum(1 for s in n.input_sources if s.node_id >= 0)\n",
              {"tiers"}),
    # an IR source's off method in a latency writer
    "ir_off": ("def tiers(nodes):\n"
               "    for n in nodes:\n"
               "        n.latency = len([s for s in n.input_sources if not s.is_off()])\n",
               {"tiers"}),
    # [r2] the shapes U-LIVE's check wrote on paper (check_evidence/probe/k2_sites): s1b, the decision split over two
    # functions joined by a hook loop over a tuple of functions and handed over through the mapping's state
    "s1b_tuple_and_state": ("from mimarsinan.mapping.latency.core_graph import alap_latencies\n"
                            "def annotate_sources(mapping):\n"
                            "    for core in mapping.cores:\n"
                            "        core.schedule_sources = [int(s.core_) for s in core.axon_sources\n"
                            "                                 if not (s.is_off_ or s.is_input_)]\n"
                            "def schedule_from_sources(mapping):\n"
                            "    return alap_latencies([core.schedule_sources for core in mapping.cores])\n"
                            "def run(mapping):\n"
                            "    for hook in (annotate_sources, schedule_from_sources):\n"
                            "        hook(mapping)\n",
                            {"annotate_sources"}),
    # s1b's tuple alone: a helper reached only through a tuple of functions, nothing handed over by an attribute
    "s1b_tuple_alone": ("from mimarsinan.mapping.latency.core_graph import alap_latencies\n"
                        "def wired_sources(mapping):\n"
                        "    return [[int(s.core_) for s in c.axon_sources if not s.is_off_] for c in mapping.cores]\n"
                        "def run(mapping):\n"
                        "    for read in (wired_sources,):\n"
                        "        sources = read(mapping)\n"
                        "    return alap_latencies(sources)\n",
                        {"wired_sources"}),
    # s1b's hand-over alone: no caller joins the two halves, the attribute does
    "s1b_state_alone": ("from mimarsinan.mapping.latency.core_graph import alap_latencies\n"
                        "def annotate_sources(mapping):\n"
                        "    for core in mapping.cores:\n"
                        "        core.schedule_sources = [int(s.core_) for s in core.axon_sources if not s.is_off_]\n"
                        "def schedule_from_sources(mapping):\n"
                        "    return alap_latencies([core.schedule_sources for core in mapping.cores])\n",
                        {"annotate_sources"}),
    # s2: the weights read through the IR graph's resolver, another name of the core's matrix
    "s2_weights_other_name": ("import numpy as np\n"
                              "def ir_tiers(ir_graph):\n"
                              "    tier = {}\n"
                              "    for node in ir_graph.nodes:\n"
                              "        flat = node.input_sources.flatten()\n"
                              "        rows = np.abs(ir_graph.resolve_core_matrix(node)).sum(axis=1) > 0\n"
                              "        producers = {int(s.node_id) for s, r in zip(flat, rows) if r}\n"
                              "        node.latency = 1 + max((tier.get(p, -1) for p in producers), default=-1)\n"
                              "        tier[node.id] = node.latency\n"
                              "    return tier\n",
                              {"ir_tiers"}),
    # s3: the class's own defect, wired sources read through the predicate module's array helper
    "s3_source_arrays": ("from mimarsinan.mapping.latency.core_graph import alap_latencies\n"
                         "from mimarsinan.mapping.latency.upstream import source_arrays\n"
                         "def wired_latencies(mapping):\n"
                         "    sources = []\n"
                         "    for core in mapping.cores:\n"
                         "        src, _neuron, off, inp, on = source_arrays(core.axon_sources)\n"
                         "        sources.append(sorted(set(src[~(off | inp | on)].tolist())))\n"
                         "    return alap_latencies(sources)\n",
                         {"wired_latencies"}),
    # the predicate module's own walk past the predicate (a function of upstream that is not the predicate)
    "s3_upstream_walk": ("from mimarsinan.mapping.latency.upstream import iter_upstream_neural_ids\n"
                         "def tiers(nodes):\n"
                         "    for n in nodes:\n"
                         "        n.latency = len(list(iter_upstream_neural_ids(n.input_sources.flatten())))\n",
                         {"tiers"}),
    # s2 by a function's name: a listed weight read imported and called
    "s2_function_by_name": ("import numpy as np\n"
                            "from mimarsinan.mapping.packing.softcore.matrix_placement import resolve_core_matrix\n"
                            "def stamp(mapping):\n"
                            "    for core in mapping.cores:\n"
                            "        w = resolve_core_matrix(None, core.placements, 4, 4, 'core')\n"
                            "        core.latency = int(np.any(w != 0))\n",
                            {"stamp"}),
    # s3 in its columnar form: a source sequence's own arrays, a method called on an object of no known type
    "s3_columnar_arrays": ("from mimarsinan.mapping.latency.core_graph import alap_latencies\n"
                           "def wired_latencies(mapping):\n"
                           "    sources = []\n"
                           "    for core in mapping.cores:\n"
                           "        src, _neuron, off, inp, on = core.axon_sources.source_arrays()\n"
                           "        sources.append(sorted(set(src[~(off | inp | on)].tolist())))\n"
                           "    return alap_latencies(sources)\n",
                           {"wired_latencies"}),
    # s4: node ids read into an array, off and outside decided by a later comparison
    "s4_index_array": ("import numpy as np\n"
                       "def ir_depths(ir_graph):\n"
                       "    depth = {}\n"
                       "    for node in ir_graph.nodes:\n"
                       "        ids = np.asarray([s.node_id for s in node.input_sources.flatten()])\n"
                       "        producers = ids[ids >= 0]\n"
                       "        node.latency = 1 + max((depth.get(int(p), 0) for p in producers), default=-1)\n"
                       "        depth[node.id] = node.latency\n"
                       "    return depth\n",
                       {"ir_depths"}),
    # s5: the core-graph order through a relative import
    "s5_relative_import": ("from ..latency.core_graph import alap_latencies\n"
                           "def wired_order(mapping):\n"
                           "    return alap_latencies([[int(s.core_) for s in c.axon_sources if not s.is_off_]\n"
                           "                           for c in mapping.cores])\n",
                           {"wired_order"}),
    # s6: the weights read by getattr with the attribute's name as a string
    "s6_getattr_matrix": ("import numpy as np\n"
                          "def stamp(mapping):\n"
                          "    for core in mapping.cores:\n"
                          "        w = getattr(core, 'core_matrix')\n"
                          "        core.latency = int(np.count_nonzero(np.any(np.asarray(w) != 0, axis=1)) > 0)\n",
                          {"stamp"}),
}

#: Where a new site is placed: inside the mapping package, so that a relative import reaches the latency engines.
NEW_MODULE = "mimarsinan.mapping.newsite.module"


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_detector_finds_each_shape_of_a_new_site(shape):
    text, expected = SHAPES[shape]
    sources = dict(STUBS)
    sources[NEW_MODULE] = text
    analysis = ls.Analysis(SRC, sources=sources)
    assert analysis.sites() | analysis.handovers() == {f"{NEW_MODULE}:{name}" for name in expected}


def test_a_reader_outside_any_schedule_is_no_site_and_a_new_predicate_caller_is_seen():
    sources = dict(STUBS)
    sources[NEW_MODULE] = (
        "from mimarsinan.mapping.latency.upstream import live_axons\n"
        "def value(core):\n"
        "    return [s for s in core.axon_sources if not s.is_off_]\n"
        "def live(m, off, outside):\n"
        "    return live_axons(m, off, outside)\n")
    analysis = ls.Analysis(SRC, sources=sources)
    assert analysis.sites() == set() and analysis.handovers() == set()
    assert {n for n, u in analysis.units.items() if u.predicate} == {f"{NEW_MODULE}:live"}
