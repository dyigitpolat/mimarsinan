"""[W15] One seeded program fault per condition the static check claims, each caught by name: the mutant fails its
condition with the named rule and Theorem 1's verdict, while the clean program establishes it (test_clean_programs).

``MUTANTS`` maps every statically checked condition to the tests that break it (test_condition_map reads it)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from mimarsinan.certification.program_check.report import FAILED
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.latency.chip import PINNED_LATENCIES
from mimarsinan.mapping.support.period_census import SCHEDULE_RESIDENT_CORES

from unit.certification.program_check import programs as P

#: Per named rule (condition, rule), the seeded faults that break it; ``test_condition_map`` holds every rule of
#: ``program_check.RULES`` to one, or to an entry of ``program_check.UNMUTATED`` with its reason.
MUTANTS = {
    ("tiles", "fan_in"): ("test_a_tile_swap", "test_a_bias_stamped_differently",
                          "test_a_hardware_bias_stamped_differently", "test_a_sign_row_flipped",
                          "test_the_static_part_of_an_operand_core_is_compared_as_any_core"),
    ("tiles", "output_source"): ("test_an_output_wire_swapped",),
    ("tiles", "coverage"): ("test_a_placement_record_dropped",),
    ("tiles", "unplaced_column"): ("test_a_placement_record_dropped",),
    ("tiles", "placement_extent"): ("test_a_placement_past_the_used_columns",),
    ("tiles", "foreign_node"): ("test_a_placement_of_a_node_the_identity_does_not_hold",),
    ("tiles", "output_width"): ("test_an_output_slice_wider_than_its_node",),
    ("tiles", "node_scales"): ("test_a_boundary_scale_of_the_program_changed",),
    ("tiles", "program_outputs"): ("test_the_network_reading_other_buffers",),
    ("tiles", "host_op"): ("test_a_host_stage_running_another_module",),
    ("tiles", "host_training"): ("test_a_host_module_left_in_training_mode",),
    ("tiles", "bank_content"): ("test_bank_content_that_is_not_the_grid", "test_a_bank_range_shifted_at_equal_extent",
                                "test_a_bank_range_one_axon_short_of_the_used_extent_fails_by_name"),
    ("tiles", "stale_memo"): ("test_a_stale_axon_span_cache", "test_a_stale_output_span_cache"),
    ("tiles", "fused_axons"): ("test_fused_axons_that_do_not_sum",),
    # [E6] the integrated units' forms (test_operand_forms, test_lane_forms, test_pruned_row_form,
    # test_residency_forms, test_sign_width_relay_forms).
    ("tiles", "payload_cells"): ("test_a_payload_placement_reading_another_heads_bank",
                                 "test_a_payload_row_reading_another_source"),
    ("tiles", "operand_binding"): ("test_a_binding_table_that_is_not_the_irs", "test_a_program_that_lost_its_binding_table"),
    ("tiles", "operand_on_fused_core"): ("test_an_operand_block_on_a_fused_core",),
    ("H9", "payload_unproduced"): ("test_a_payload_write_produced_after_the_stage_that_reads_it",),
    ("laws", "operand_source"): ("test_a_fused_operand_row_reading_another_source",
                                 "test_a_lane_law_naming_another_operand_row"),
    ("H8", "cell_width"): ("test_a_core_declared_narrower_than_its_codes",),
    ("H10", "undeclared_core"): ("test_a_pinned_core_no_declared_core_type_holds_is_named",),
    ("N-identity", "dropped_offset_row"): ("test_the_pre_n7_offset_multiplier_is_named_where_its_image_of_zero_is_not_zero",
                                           "test_a_reader_whose_adc_rounds_the_dropped_term",
                                           "test_a_reader_whose_value_zero_is_no_code_of_its_grid"),
    ("H6", "partial_sums"): ("test_a_dropped_partial_sum", "test_a_duplicated_partial_sum",
                             "test_a_permuted_coalescing_wire"),
    ("R2", "delivery_delay"): ("test_a_carry_delay_off_by_one",),
    ("R2", "delay_on_encoded_slice"): ("test_a_delivery_delay_on_a_host_encoded_slice",),
    ("R1", "pinned_latency"): ("test_a_wrong_pinned_latency",),
    ("R1", "pin_count"): ("test_a_read_core_that_is_never_walked",),
    ("laws", "stamped_law"): ("test_a_law_stamped_differently",),
    ("laws", "law_free_core"): ("test_a_law_free_core_in_an_integer_value_program",),
    ("H8", "accumulation_bound"): ("test_a_charge_bound_exceeded_in_float64", "test_a_charge_bound_exceeded_in_float32",
                                   "test_a_charge_bound_one_bit_short_under_synchronized_ttfs"),
    ("H8", "off_lattice"): ("test_an_off_lattice_weight", "test_a_payload_whose_values_are_no_codes",
                            "test_a_core_that_accumulates_its_bias_still_needs_it_integral"),
    ("H8", "input_range"): ("test_a_code_whose_range_no_law_or_grid_declares",),
    ("H8", "law_magnitude"): ("test_a_law_whose_magnitude_leaves_float64",),
    ("H8", "upload_precision"): ("test_a_weight_a_float32_upload_rounds", "test_a_payload_wider_than_the_float32_upload"),
    ("H9", "reads_unproduced"): ("test_a_pass_ordered_before_its_producer",),
    ("H9", "produced_twice"): ("test_a_node_produced_twice",),
    ("H9", "output_unproduced"): ("test_the_network_reading_a_node_no_stage_produces",),
    ("H9", "coalescing_group_cut"): ("test_a_coalescing_group_cut_across_passes",),
    ("H9", "stale_memo"): ("test_a_stale_consumer_count_memo",),
    ("H10", "pinned_slot_overwritten"): ("test_a_pinned_slot_overwritten",
                                         "test_a_fit_on_a_pool_too_small_for_its_loading_passes"),
    ("H10", "resident_content"): ("test_a_resident_pass_that_was_never_written",
                                  "test_a_multiplexed_pass_holding_other_weights_than_its_runs_copy",
                                  "test_a_resident_level_unit_holding_other_weights_than_its_heads"),
    ("H10", "no_programming_pass"): ("test_a_first_pass_marked_weight_resident",),
    ("H10", "pin_out_of_range"): ("test_a_pin_of_a_core_the_pass_does_not_hold",),
    ("H10", "undeclared_pool"): ("test_pinned_cores_without_a_declared_pool",),
    # [U-F4] a wavefront group's head and its later passes (``test_wavefront_forms``)
    ("H10", "wavefront_layout"): ("test_a_head_stating_another_layout_than_it_loads",
                                  "test_a_later_pass_that_reprograms_its_cores"),
    ("H10", "wavefront_dark_core"): ("test_a_dark_core_holding_a_placement",),
    ("H1-theta", "threshold"): ("test_a_threshold_stored_differently",),
    ("H1-theta", "activation_scale"): ("test_a_stored_singleton_of_the_core_changed",),
    ("H1-theta", "parameter_scale"): ("test_a_stored_singleton_of_the_core_changed",),
    ("H1-theta", "input_activation_scale"): ("test_a_stored_singleton_of_the_core_changed",),
    ("H1-latency", "co_resident_latencies"): ("test_co_resident_soft_cores_at_two_latencies",),
    ("R3", "carry_scope"): ("test_a_carried_slice_scoped_to_another_segment",),
    ("R3", "mixed_segments"): ("test_a_core_holding_neurons_of_two_identity_segments",),
    ("R3", "stale_memo"): ("test_a_stale_carried_output_memo",),
    ("R4", "phase_channel"): ("test_a_dithered_slice_the_encoder_refuses",
                              "test_a_channel_table_keyed_off_the_identity_under_a_dithered_encode",
                              "test_a_reindex_fault_in_the_deployments_channel_table"),
    ("R4", "slice_kind"): ("test_the_level_flag_on_a_pass_reading_a_host_train",),
    ("R4", "multi_home_unit"): ("test_a_multi_level_windowed_pass_run_streamed_is_a_multi_home_unit",),
    ("R5", "unit_kind"): ("test_a_single_level_pass_run_as_a_plain_stage",),
    ("R5", "level_flag"): ("test_the_level_flag_cleared_on_the_level_units_of_a_windowed_program",),
    ("H0", "level_unit"): ("test_level_units_under_streamed_lif",),
    ("H0", "streamed_windowed_pass"): ("test_a_multi_level_windowed_stage_run_streamed",),
    ("H0", "unlevelled_windowed_stage"): ("test_a_windowed_identity_left_unlevelled",),
    ("H0", "level_flag"): ("test_the_level_flag_on_a_streamed_pass_with_delayed_slices",),
    ("H5-collapse", "transfer"): ("test_a_streamed_program_collapsing_its_carried_slices",),
    ("H5-collapse", "wire_delivery"): ("test_the_level_flag_on_a_streamed_pass_with_delayed_slices",),
    ("H5-collapse", "shifted_count"): ("test_a_delay_on_a_windowed_plain_stage",),
    ("H7", "fused_fan_in"): ("test_a_fused_fan_in_on_a_per_event_core",),
    ("H7", "co_residency"): ("test_co_resident_soft_cores_on_a_per_event_core",),
    ("H7", "split_neuron"): ("test_a_split_neuron_on_a_per_event_core",),
    ("H7", "level_unit"): ("test_level_units_on_a_per_event_core",),
    ("R6", "unwalked_core"): ("test_a_read_core_that_is_never_walked",),
    ("R6", "core_cycle"): ("test_cores_that_read_each_other",),
    ("N-identity", "dropped_live_row"): ("test_compaction_that_drops_a_live_row",),
    ("N-identity", "dropped_read_column"): ("test_a_dead_dropped_column_is_the_networks_and_a_live_one_is_named",),
    ("N-identity", "dropped_output"): ("test_compaction_that_drops_a_live_network_output",),
    ("H8-staircase", "off_lattice"): ("test_an_off_lattice_threshold",),
    # [E1 F1] test_ir_tiling: each fault built into the deployment's IR before its program, so only ir-tiling sees it.
    ("ir-tiling", "unkeyed_neuron"): ("test_a_neural_node_with_no_perceptron_identity",),
    ("ir-tiling", "duplicate_neuron"): ("test_two_tiles_holding_one_neuron",),
    ("ir-tiling", "partial_sum_group"): ("test_a_tile_holding_a_share_of_a_fan_in",),
    ("ir-tiling", "missing_neuron"): ("test_a_tile_keyed_on_another_slice_of_its_perceptron",
                                      "test_a_column_one_ir_compacts_and_the_other_keeps"),
    ("ir-tiling", "extra_neuron"): ("test_a_tile_keyed_on_another_slice_of_its_perceptron",),
    ("ir-tiling", "fan_in"): ("test_a_tile_whose_weight_the_tiling_changed", "test_a_tile_reading_another_source",
                              "test_a_tile_whose_bias_row_the_tiling_changed",
                              "test_a_tile_whose_compaction_drops_a_live_row"),
    ("ir-tiling", "bias"): ("test_a_tile_whose_hardware_bias_the_tiling_changed",),
    ("ir-tiling", "threshold"): ("test_the_tiles_whose_threshold_the_tiling_changed",),
    ("ir-tiling", "neuron_scale"): ("test_the_tiles_whose_scale_the_tiling_changed",),
    ("ir-tiling", "boundary_grid"): ("test_a_tile_whose_boundary_grid_the_tiling_changed",),
    ("ir-tiling", "law"): ("test_a_tile_whose_readout_law_the_tiling_changed",
                           "test_a_tile_whose_identity_adc_clamps_below_its_accumulator_bound"),
    # [E1 fix round] test_ir_tiling_executed: what the identity programs execute, the compaction they apply, the
    # reference, and the channel tables on the reindexed branch, each beside its sampled run.
    ("ir-tiling", "executed_latency"): (
        "test_a_tiling_that_splits_a_producers_consumers_moves_its_executed_latency",
        "test_per_tile_dead_row_masks_that_move_an_executed_latency"),
    ("ir-tiling", "timing_derivation"): ("test_an_identity_builder_whose_timing_departs_from_the_core_graph",),
    ("ir-tiling", "reference_geometry"): (
        "test_the_deployments_own_tiled_ir_given_as_the_reference",
        "test_an_ir_at_another_declared_geometry_given_as_the_reference",
        "test_a_reference_holding_one_host_op_name_twice", "test_a_reference_holding_an_unkeyed_neural_node",
        "test_a_reference_holding_a_coalescing_group"),
    ("ir-tiling", "activation"): ("test_a_tile_whose_activation_kind_the_tiling_changed",),
    ("ir-tiling", "host_op"): ("test_a_host_op_running_another_module",),
    ("ir-tiling", "program_outputs"): ("test_the_network_reading_other_neurons",),
    ("ir-tiling", "phase_channel"): ("test_a_channel_table_keyed_on_tile_columns_under_a_dithered_encode",
                                     "test_a_reindex_fault_confined_to_the_unbounded_identity_program"),
}


def _failed(report, condition: str, rule: str) -> None:
    result = report.results[condition]
    assert result.status == FAILED, (condition, result.status, result.note)
    assert rule in {v.rule for v in result.violations}, [v.text() for v in result.violations]
    theorem = next(s.theorem for s in report.specs if s.condition == condition)
    verdict = report.theorem1 if theorem == "theorem1" else report.proposition2
    assert verdict.startswith("failed") and condition in verdict, verdict


def _units(program):
    for stage in P.neural(program):
        yield from (list(getattr(stage, "retimed_level_stages", None) or ()) or [stage])


def _two_core_unit(program):
    for unit in _units(program):
        cores = unit.hard_core_mapping.cores
        rows = unit.hard_core_mapping.soft_core_placements_per_hard_core
        for a in range(len(cores)):
            for b in range(a + 1, len(cores)):
                if (cores[a].core_matrix.shape == cores[b].core_matrix.shape and rows[a] and rows[b]
                        and rows[a][0]["ir_node_id"] != rows[b][0]["ir_node_id"]):
                    return unit, a, b
    raise AssertionError("no unit holds two same-shape cores of different nodes")


def test_a_tile_swap():
    program, ir = P.streamed_multipass()
    bad = P.clone(program)
    unit, a, b = _two_core_unit(bad)
    cores = unit.hard_core_mapping.cores
    cores[a].core_matrix, cores[b].core_matrix = cores[b].core_matrix.copy(), cores[a].core_matrix.copy()
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "tiles", "fan_in")


def _fused(program):
    for unit in _units(program):
        for h, core in enumerate(unit.hard_core_mapping.cores):
            if core.fused_component_axons and len(core.fused_component_axons) > 1:
                return unit, h, core
    raise AssertionError("no fused core")


def test_a_dropped_partial_sum():
    program, ir = P.coalesced()
    bad = P.clone(program)
    _, _, core = _fused(bad)
    first = int(core.fused_component_axons[0])
    core.core_matrix[first:, :] = 0.0
    report = P.run(bad, ir, P.decl("streamed_lif"))
    _failed(report, "H6", "partial_sums")
    _failed(report, "tiles", "fan_in")


def test_a_duplicated_partial_sum():
    program, ir = P.coalesced()
    bad = P.clone(program)
    _, _, core = _fused(bad)
    first = int(core.fused_component_axons[0])
    width = min(first, int(core.axons_per_core) - first)
    core.core_matrix[:width, :] *= 2.0
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "H6", "partial_sums")


def test_a_permuted_coalescing_wire():
    program, ir = P.coalesced()
    bad = P.clone(program)
    _, _, core = _fused(bad)
    grid = core.core_matrix
    used = int(core.axons_per_core) - int(core.available_axons)
    rows = [a for a in range(used) if np.any(grid[a] != 0) and not core.axon_sources[a].is_off_]
    a, b = next((x, y) for x in rows for y in rows if x < y and not np.array_equal(grid[x], grid[y]))
    core.axon_sources[a], core.axon_sources[b] = core.axon_sources[b], core.axon_sources[a]
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "H6", "partial_sums")


def test_fused_axons_that_do_not_sum():
    program, ir = P.coalesced()
    bad = copy.deepcopy(program)
    _, _, core = _fused(bad)
    core.fused_component_axons = list(core.fused_component_axons) + [1]
    report = P.run(bad, ir, P.decl("streamed_lif"))
    _failed(report, "tiles", "fused_axons")       # a metadata rule: no executor reads the field, the fit does
    assert report.results["H6"].status != FAILED


def _delayed(program):
    return next(s for s in P.neural(program) if getattr(s, "replay_delays", None))


def test_a_carry_delay_off_by_one():
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    stage = _delayed(bad)
    k = next(iter(stage.replay_delays))
    delays = dict(stage.replay_delays)
    delays[k] += 1
    stage.replay_delays = delays
    report = P.run(bad, ir, P.decl("streamed_lif"))
    _failed(report, "R2", "delivery_delay")
    assert report.results["H5-skew"].status == FAILED


def test_a_wrong_pinned_latency():
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    hcm = next(s.hard_core_mapping for s in P.neural(bad) if getattr(s.hard_core_mapping, PINNED_LATENCIES, None))
    pins = list(getattr(hcm, PINNED_LATENCIES))
    pins[0] += 1
    setattr(hcm, PINNED_LATENCIES, pins)
    report = P.run(bad, ir, P.decl("streamed_lif"))
    _failed(report, "R1", "pinned_latency")
    assert report.results["H5-skew"].status == FAILED


def _on_row(program):
    for unit in _units(program):
        for core in unit.hard_core_mapping.cores:
            for a, src in enumerate(core.axon_sources):
                if src.is_always_on_ and np.any(core.core_matrix[a] != 0):
                    return core, a
    raise AssertionError("no always-on row with a bias")


def test_a_bias_stamped_differently():
    program, ir = P.streamed_multipass()
    bad = P.clone(program)
    core, a = _on_row(bad)
    column = int(np.flatnonzero(core.core_matrix[a])[0])
    core.core_matrix[a, column] += 1.0
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "tiles", "fan_in")


def test_a_hardware_bias_stamped_differently():
    program, ir = P.resident_chain()
    bad = copy.deepcopy(program)
    core = P.neural(bad)[0].hard_core_mapping.cores[0]
    core.hardware_bias = np.array(core.hardware_bias, dtype=np.float64) + np.eye(1, len(core.hardware_bias))[0]
    _failed(P.run(bad, ir, P.decl("synchronized_lif")), "tiles", "fan_in")


def test_an_output_wire_swapped():
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    hcm = P.neural(bad)[0].hard_core_mapping
    out = list(np.asarray(hcm.output_sources).flatten())
    out[0], out[1] = out[1], out[0]
    hcm.output_sources = np.array(out, dtype=object)
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "tiles", "output_source")


def test_a_law_stamped_differently():
    from mimarsinan.mapping.ir.core_law import CoreLaw

    program, ir = P.value_program()
    bad = copy.deepcopy(program)
    core = next(c for s in P.neural(bad) for c in s.hard_core_mapping.cores if getattr(c, "readout_law", None))
    law = core.readout_law
    bias = np.array(law.bias)
    bias[0] += 1
    core.readout_law = CoreLaw(input_step=law.input_step, m0=law.m0, shift=law.shift, bias=bias, lo=law.lo,
                               hi=law.hi, output_step=law.output_step, adc_shift=law.adc_shift,
                               adc_levels=law.adc_levels)
    _failed(P.run(bad, ir, P.decl("value_integer", accumulation={"value_census": "float64"})), "laws",
            "stamped_law")


def _scaled_ir(scale: float):
    from unit.mapping.residency.executed import vehicles as V

    ir = P.integer_ir(V.random_rep(0))
    node = ir.get_neural_cores()[0]
    matrix = np.array(node.core_matrix, dtype=np.float64)
    matrix[0, 0] = scale
    node.core_matrix = matrix
    return V.build(ir, 3, None), ir


def test_a_charge_bound_exceeded_in_float64():
    program, ir = _scaled_ir(2.0 ** 50)
    _failed(P.run(program, ir, P.decl("streamed_lif")), "H8", "accumulation_bound")


def test_a_charge_bound_exceeded_in_float32():
    program, ir = _scaled_ir(2.0 ** 22)
    report = P.run(program, ir, P.decl("streamed_lif", accumulation={"hcm": "float32"}))
    _failed(report, "H8", "accumulation_bound")
    assert P.run(program, ir, P.decl("streamed_lif")).results["H8"].status != FAILED


def test_an_off_lattice_weight():
    program, ir = _scaled_ir(0.5)
    _failed(P.run(program, ir, P.decl("streamed_lif")), "H8", "off_lattice")


def test_a_pass_ordered_before_its_producer():
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    first = next(i for i, s in enumerate(bad.stages) if s.kind == "neural")
    second = next(i for i, s in enumerate(bad.stages) if i > first and s.kind == "neural")
    bad.stages[first], bad.stages[second] = bad.stages[second], bad.stages[first]
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "H9", "reads_unproduced")


def test_a_node_produced_twice():
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    last = max(i for i, s in enumerate(bad.stages) if s.kind == "neural")
    bad.stages.append(copy.deepcopy(bad.stages[last]))
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "H9", "produced_twice")


def test_a_pinned_slot_overwritten():
    program, ir = P.residency_planned()
    bad = copy.deepcopy(program)
    passes = [s for s in P.neural(bad) if getattr(s, SCHEDULE_RESIDENT_CORES, None) is not None]
    assert passes, "the residency plan pins cores"
    for stage in passes:
        setattr(stage, SCHEDULE_RESIDENT_CORES, frozenset(range(len(stage.hard_core_mapping.cores))))
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "H10", "pinned_slot_overwritten")


def test_a_resident_pass_that_was_never_written():
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    later = [s for s in P.neural(bad) if s.schedule_pass_index]
    later[0].schedule_weights_resident = True
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "H10", "resident_content")


def test_a_threshold_stored_differently():
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    core = P.neural(bad)[0].hard_core_mapping.cores[0]
    core.threshold = float(core.threshold) + 1.0
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "H1-theta", "threshold")


def test_co_resident_soft_cores_at_two_latencies(monkeypatch):
    """The projection's refusal by name switched off (``_core_latencies`` keeps the first resident's latency): the
    token family co-packs instances the fused segment runs at two latencies, which the check names."""
    import mimarsinan.mapping.packing.pass_timing as pt
    from unit.mapping.residency.executed import vehicles as V

    def lenient(pl, fused, label):
        return [min((fused[n] for n in pl.nodes(h)), default=0) for h in range(len(pl.cores))]

    monkeypatch.setattr(pt, "_core_latencies", lenient)
    for count in (2, 3, 4, 6):
        ir = P.integer_ir(V.token_repr(deep=(8,), tokens=2, deep_tokens=(0,)))
        program = V.build(ir, count, None)
        report = P.run(program, ir, P.decl("streamed_lif"))
        if report.results["H1-latency"].violation_count:
            _failed(report, "H1-latency", "co_resident_latencies")
            return
    raise AssertionError("no budget co-packed the two latencies")


def test_a_carried_slice_scoped_to_another_segment():
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    stage = _delayed(bad)
    stage.schedule_segment_index = 99
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "R3", "carry_scope")


def test_a_dithered_slice_the_encoder_refuses():
    """[E1] A program without its channel table (a pickle from before E1, another builder): the dithered encoder
    refuses every slice by name, and R4 names the program for it."""
    from unit.mapping.residency.executed import vehicles as V

    ir = P.integer_ir(V.split_level_repr())
    program = V.build(ir, 1, None)
    clean = P.run(program, ir, P.decl("streamed_lif", dithered_encode=True))
    assert clean.results["R4"].status != FAILED and clean.results["R4"].checked
    bad = copy.deepcopy(program)
    bad.node_phase_channels = {}
    _failed(P.run(bad, ir, P.decl("streamed_lif", dithered_encode=True)), "R4", "phase_channel")


def test_a_single_level_pass_run_as_a_plain_stage():
    for seed in range(12):
        program, ir = P.windowed_multipass(seed, 3)
        bad = copy.deepcopy(program)
        single = [s for s in P.neural(bad) if len(getattr(s, "retimed_level_stages", None) or ()) == 1
                  and s.schedule_segment_index is not None]
        if not single:
            continue
        single[0].retimed_level_stages = None
        report = P.run(bad, ir, P.decl("windowed_lif"), identity=P.identity_of(ir, retimed=True))
        _failed(report, "R5", "unit_kind")
        return
    raise AssertionError("no windowed pass ran a single level unit")


def test_level_units_under_streamed_lif():
    program, ir = P.windowed_multipass()
    _failed(P.run(program, ir, P.decl("streamed_lif")), "H0", "level_unit")


def test_a_multi_level_windowed_stage_run_streamed():
    for seed in range(12):
        program, ir = P.windowed_multipass(seed, 3)
        bad = copy.deepcopy(program)
        multi = [s for s in P.neural(bad) if len(getattr(s, "retimed_level_stages", None) or ()) >= 2]
        if not multi:
            continue
        multi[0].retimed_level_stages = None
        report = P.run(bad, ir, P.decl("windowed_lif"), identity=P.identity_of(ir, retimed=True))
        _failed(report, "H0", "streamed_windowed_pass")
        return
    raise AssertionError("no windowed pass ran two level units")


def test_a_count_semantics_carrying_rasters_is_named_though_its_counts_are_kept():
    """Synchronized LIF under VERBATIM keeps its counts (the hypothesis table), yet the check names the transfer: a
    right program that fails, since the check is sufficient, not necessary (condition map section 5)."""
    program, ir = P.streamed_multipass()
    _failed(P.run(program, ir, P.decl("synchronized_lif", pass_transfer="verbatim")), "H5-collapse", "transfer")


def test_a_fused_fan_in_on_a_per_event_core():
    program, ir = P.coalesced()
    report = P.run(program, ir, P.decl("streamed_lif", firing_granularity="per_event"))
    _failed(report, "H7", "fused_fan_in")
    clean = P.run(program, ir, P.decl("streamed_lif"))
    assert clean.results["H7"].status == "vacuous" and clean.established("theorem1")


def test_a_read_core_that_is_never_walked():
    """A pin list one short of the pass's cores: ``ChipLatency`` pins the others and leaves the last core at its
    stored latency, here none, so every walk skips it and its readers read zeros."""
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    hcm = next(s.hard_core_mapping for s in P.neural(bad) if getattr(s.hard_core_mapping, PINNED_LATENCIES, None))
    last = len(hcm.cores) - 1
    setattr(hcm, PINNED_LATENCIES, list(getattr(hcm, PINNED_LATENCIES))[:last])
    hcm.cores[last].latency = None
    report = P.run(bad, ir, P.decl("streamed_lif"))
    reads_last = any(int(s.core_) == last for s in np.asarray(hcm.output_sources).flatten()) or any(
        int(src.core_) == last and not src.is_input_ for c in hcm.cores for src in c.axon_sources)
    assert reads_last
    _failed(report, "R6", "unwalked_core")
    _failed(report, "R1", "pin_count")


def test_cores_that_read_each_other():
    program, ir = P.streamed_multipass()
    bad = P.clone(program)
    unit, a, b = _two_core_unit(bad)
    cores = unit.hard_core_mapping.cores
    for x, y in ((a, b), (b, a)):
        row = int(cores[x].axons_per_core) - int(cores[x].available_axons) - 1
        cores[x].axon_sources[row] = SpikeSource(y, 0, is_input=False, is_off=False)
        cores[x].core_matrix[row, :] = 1.0
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "R6", "core_cycle")


def test_compaction_that_drops_a_live_row():
    from unit.mapping.residency.executed import vehicles as V

    ir = P.integer_ir(V.random_rep(0))
    node = ir.get_neural_cores()[2]           # its row 0 reads a kept neural column (node 1's), so it is live
    rows, cols = node.get_input_count(), node.get_output_count()
    node.pruned_row_mask = [r == 0 for r in range(rows)]
    node.pruned_col_mask = [False] * cols
    matrix = np.array(node.core_matrix, dtype=np.float64)
    matrix[0, :] = 3.0
    node.core_matrix = matrix
    program = V.build(ir, 3, None)
    _failed(P.run(program, ir, P.decl("synchronized_lif")), "N-identity", "dropped_live_row")


def test_a_dropped_row_read_from_a_host_op_is_left_to_the_sampled_run():
    """A dropped row whose source is a host op's output: whether that output is zero is the host op's property."""
    from mimarsinan.certification.program_check.report import NOT_STATIC
    from unit.mapping.residency.executed import vehicles as V

    ir = P.integer_ir(V.random_rep(0))
    node = ir.get_neural_cores()[0]           # reads the encoder, a host op
    node.pruned_row_mask = [r == 0 for r in range(node.get_input_count())]
    node.pruned_col_mask = [False] * node.get_output_count()
    matrix = np.array(node.core_matrix, dtype=np.float64)
    matrix[0, :] = 3.0
    node.core_matrix = matrix
    report = P.run(V.build(ir, 3, None), ir, P.decl("synchronized_lif"))
    assert report.results["N-identity"].status == NOT_STATIC and report.proposition2.startswith("conditional")


def test_an_off_lattice_threshold():
    from unit.mapping.residency.executed import vehicles as V

    ir = P.integer_ir(V.random_rep(0))
    for node in ir.get_neural_cores():
        node.threshold = float(node.threshold) + 0.5
    program = V.build(ir, 3, None)
    _failed(P.run(program, ir, P.decl("synchronized_lif")), "H8-staircase", "off_lattice")


@pytest.mark.parametrize("element,attribute", [("hard_core", "zz_unit_core_field"), ("placement", "lane_owner")])
def test_an_unknown_program_form_fails_by_name(element, attribute):
    """Unregistered, the form fails by name; registered with a predicate that reads it, the predicate decides."""
    from mimarsinan.certification.program_check import ProgramForm, register_program_form
    from mimarsinan.certification.program_check.forms import unregister_program_form

    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    stage = P.neural(bad)[0]
    if element == "hard_core":
        setattr(stage.hard_core_mapping.cores[0], attribute, ("operand",))
    else:
        stage.hard_core_mapping.soft_core_placements_per_hard_core[0][0][attribute] = 3
    report = P.run(bad, ir, P.decl("streamed_lif"))
    assert report.theorem1.startswith("failed: unknown program form"), report.theorem1
    assert f"{element}.{attribute}" in report.theorem1
    register_program_form(ProgramForm(f"test_{attribute}", element, frozenset({attribute}), condition="tiles",
                                      predicate=lambda program, delta, iota, declaration: []))
    try:
        assert P.run(bad, ir, P.decl("streamed_lif")).established("theorem1")
    finally:
        unregister_program_form(f"test_{attribute}")
