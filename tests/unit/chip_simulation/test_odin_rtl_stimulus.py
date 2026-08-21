"""[ODIN5] The RTL testbench's token program: encoding, drain order, translation.

These are the FAST half of plan §7 rows 15-18: everything that can be pinned
without a simulator lives in the default suite, so a drift in the opcode table,
an SPI address field, or the canonical drain order fails in seconds rather than
only under the named RTL runner.
"""

import numpy as np
import pytest

from mimarsinan.chip_simulation.odin_rtl.cosim import FIRST_TAG, build_cosim_ops
from mimarsinan.chip_simulation.odin_rtl.program_ops import (
    UNMASKED,
    barrier_stage_ops,
    clear_stage_ops,
    config_stage_ops,
    gate_stage_ops,
    inject_ops,
    neuron_readback_ops,
    plan_cycle_injection,
    shadow_ops,
    slot_rows_from_inject,
    stages_of_kind,
    tref_stage_ops,
)
from mimarsinan.chip_simulation.odin_rtl.reference import (
    ReferenceTraceError,
    gather_axon_counts,
    simulate_cycles,
)
from mimarsinan.chip_simulation.odin_rtl.stimulus import (
    OP_AER,
    OP_SHADOW,
    OP_SPI_R,
    OP_SPI_W,
    OP_TAG,
    OP_WAIT,
    Op,
    StimulusError,
    all_neuron_tref_event,
    config_write_address,
    decode_ops,
    encode_ops,
    masked_byte_data,
    neuron_address,
    neuron_spike_event,
    op_summary,
    read_stimulus,
    shadow_register_id,
    shadow_syn_sign_id,
    synapse_address_field,
    word_bytes,
    write_stimulus,
)
from mimarsinan.chip_simulation.odin_rtl.capture import (
    TestbenchFailure as _TestbenchFailure,
)
from mimarsinan.chip_simulation.odin_rtl.capture import parse_capture
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.export.odin.exporter import export_odin
from mimarsinan.mapping.export.odin.program import STAGE_CONFIG, STAGE_INJECT
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping

ODIN_LAW = SomaLaw.resolve({
    "firing_mode": "Novena",
    "thresholding_mode": "<=",
    "firing_granularity": "per_event",
    "membrane_bits": 8,
})


def _core(matrix, *, threshold, sources):
    core = HardCore(
        axons_per_core=matrix.shape[0], neurons_per_core=matrix.shape[1],
        has_bias_capability=False,
    )
    core.core_matrix = np.asarray(matrix, dtype=np.float64)
    core.axon_sources = list(sources)
    core.threshold = threshold
    core.available_axons = 0
    core.available_neurons = 0
    return core


def _two_core_mapping():
    first = _core(
        np.array([[2.0, -3.0], [0.0, 1.0], [1.0, 1.0]]),
        threshold=4.0,
        sources=[
            SpikeSource(-2, 0, is_input=True),
            SpikeSource(-2, 1, is_input=True),
            SpikeSource(-3, 0, is_always_on=True),
        ],
    )
    second = _core(
        np.array([[3.0, 0.0], [-2.0, 5.0]]),
        threshold=6.0,
        sources=[SpikeSource(0, 0), SpikeSource(0, 1)],
    )
    mapping = HardCoreMapping(chip_cores=[])
    mapping.cores = [first, second]
    mapping.output_sources = [SpikeSource(1, 0), SpikeSource(1, 1)]
    ChipLatency(mapping).calculate()
    return mapping


def _export(mapping=None):
    return export_odin(
        mapping if mapping is not None else _two_core_mapping(),
        soma_law=ODIN_LAW, weight_bits=4, weight_sign_granularity="per_axon",
        effective_max_axons=127, membrane_init=0,
    )


class TestTheTokenEncoding:
    def test_a_program_round_trips_through_the_one_encoder_and_decoder(self):
        ops = [
            Op(OP_SPI_W, (0, 0x51234, 0x000FF)),
            Op(OP_AER, (1, neuron_spike_event(9))),
            Op(OP_WAIT, (4321,)),
            Op(OP_TAG, (7,)),
            Op(OP_SHADOW, (0, 3, 1)),
            Op(OP_SPI_R, (1, 0x91234, 0xAB)),
        ]
        assert decode_ops(encode_ops(ops)) == tuple(ops)

    def test_the_stream_is_end_terminated_so_the_tb_never_runs_off_its_array(self):
        tokens = encode_ops([Op(OP_WAIT, (1,))])
        assert tokens[-1] == 0
        with pytest.raises(StimulusError, match="END-terminated"):
            decode_ops(tokens[:-1])

    def test_a_truncated_argument_list_is_refused_by_name(self):
        with pytest.raises(StimulusError, match="argument"):
            decode_ops([OP_SPI_W, 0, 1])

    def test_an_unknown_opcode_is_refused_rather_than_skipped(self):
        with pytest.raises(StimulusError, match="unknown opcode"):
            decode_ops([99, 0])

    def test_the_wrong_arity_is_refused_at_construction(self):
        with pytest.raises(StimulusError, match="takes 3 argument"):
            Op(OP_SPI_W, (0, 1))

    def test_a_file_round_trips(self, tmp_path):
        ops = [Op(OP_TAG, (3,)), Op(OP_AER, (0, neuron_spike_event(255)))]
        path = tmp_path / "stim.hex"
        assert write_stimulus(path, ops) == len(encode_ops(ops))
        assert read_stimulus(path) == tuple(ops)


class TestTheSpiAddressFields:
    """doc/README.md Sec.2.1: a = {R, W, cmd[1:0], addr[15:0]}."""

    def test_a_config_write_uses_command_zero_with_neither_r_nor_w(self):
        assert config_write_address(24) == 24
        assert config_write_address(0xFFFF) == 0xFFFF

    def test_a_neuron_write_sets_w_and_command_one(self):
        assert neuron_address(0x2A, 0xD, write=True) == (1 << 18) | (0b01 << 16) \
            | (0xD << 8) | 0x2A

    def test_a_neuron_read_sets_r_and_command_one(self):
        assert neuron_address(0x2A, 0xD, write=False) == (1 << 19) | (0b01 << 16) \
            | (0xD << 8) | 0x2A

    def test_a_synapse_access_carries_the_two_bit_byte_address_at_bit_thirteen(self):
        assert synapse_address_field(0x1234, 0b10, write=True) == \
            (1 << 18) | (0b10 << 16) | (0b10 << 13) | 0x1234

    def test_the_data_field_is_mask_over_byte_and_a_mask_bit_keeps_the_old_bit(self):
        assert masked_byte_data(0xA5, 0x0F) == 0x0FA5
        assert masked_byte_data(0x12, UNMASKED) == 0x0012

    def test_out_of_range_fields_are_refused_rather_than_truncated(self):
        with pytest.raises(StimulusError, match="neuron word address"):
            neuron_address(256, 0, write=True)
        with pytest.raises(StimulusError, match="synapse byte address"):
            synapse_address_field(0, 4, write=True)
        with pytest.raises(StimulusError, match="mask"):
            masked_byte_data(0, 256)

    def test_a_word_splits_into_spi_byte_order_least_significant_first(self):
        assert word_bytes(0x0102, 16)[:3] == (0x02, 0x01, 0x00)


class TestTheAerEventEncoding:
    def test_a_neuron_spike_event_names_its_row_and_the_0x07_suffix(self):
        assert neuron_spike_event(0) == 0x0007
        assert neuron_spike_event(255) == 0xFF07

    def test_the_all_neurons_time_reference_is_the_0x7f_event(self):
        assert all_neuron_tref_event() == 0x7F

    def test_a_row_outside_the_crossbar_is_refused(self):
        with pytest.raises(StimulusError, match="pre-synaptic row"):
            neuron_spike_event(256)


class TestTheDrainOrderReachesTheWire:
    """Plan §2.3: ascending slots, one slot's multiplicity ADJACENT.

    The raw AER stream has no multiplicity field, so k occurrences are k
    back-to-back events on the same physical row; anything else is a different
    physics (the θ=5, w=[+3,−3] counterexample).
    """

    def test_a_multiplicity_becomes_that_many_adjacent_row_events(self):
        ops = inject_ops(0, [(4, 3)])
        assert ops == [Op(OP_AER, (0, neuron_spike_event(4)))] * 3

    def test_slots_are_drained_ascending_with_each_slots_events_together(self):
        slot_rows = {0: (0, 1), 1: (2,), 2: (4,)}
        events = plan_cycle_injection(slot_rows, [2, 0, 3])
        assert events == ((0, 2), (1, 2), (4, 3))
        rows = [op.args[1] for op in inject_ops(0, events)]
        assert rows == [neuron_spike_event(0)] * 2 + [neuron_spike_event(1)] * 2 \
            + [neuron_spike_event(4)] * 3

    def test_a_slot_with_no_emitting_row_contributes_nothing(self):
        assert plan_cycle_injection({0: ()}, [5]) == ()

    def test_a_negative_multiplicity_is_refused_not_dropped(self):
        with pytest.raises(StimulusError, match="non-negative"):
            inject_ops(0, [(0, -1)])

    def test_the_slot_row_map_is_reconstructed_from_the_exported_inject_stage(self):
        payload = stages_of_kind(_export().program, STAGE_INJECT)[0]
        slot_rows, bias = slot_rows_from_inject(payload)
        assert bias == (2,)
        assert slot_rows == {0: (0, 1), 1: (2,), 2: (4,)}

    def test_a_slot_both_injected_and_delegated_is_refused(self):
        with pytest.raises(StimulusError, match="both injected"):
            slot_rows_from_inject({"events": [[0, 1]], "runtime_rows": [[0, [0]]]})


class TestTheProgramTranslation:
    def test_a_config_stage_writes_every_register_then_every_memory_byte(self):
        payload = stages_of_kind(_export().program, STAGE_CONFIG)[0]
        ops = config_stage_ops(payload)
        assert len(ops) == len(payload["register_writes"]) + 256 * 16 + 8192 * 4
        assert all(op.code == OP_SPI_W for op in ops)

    def test_a_gate_stage_reaches_every_core(self):
        ops = gate_stage_ops([0, 1, 2], on=True)
        assert [op.args[0] for op in ops] == [0, 1, 2]
        assert {op.args[2] for op in ops} == {1}
        assert {op.args[2] for op in gate_stage_ops([0], on=False)} == {0}

    def test_a_clear_stage_becomes_masked_neuron_state_byte_writes(self):
        export = _export()
        clears = [s for s in export.program.stages if s["kind"] == "CLEAR"]
        ops = clear_stage_ops(clears[0]["payload"])
        assert ops
        assert {op.args[2] >> 8 for op in ops} != {UNMASKED}

    def test_a_barrier_stage_becomes_a_plain_wait_of_its_own_bound(self):
        assert barrier_stage_ops({"cycles": 1234}) == [Op(OP_WAIT, (1234,))]

    def test_a_single_neuron_time_reference_is_refused_by_name(self):
        with pytest.raises(StimulusError, match="all-neurons"):
            tref_stage_ops({"scope": "one"}, [0])

    def test_the_readback_expectation_is_the_exported_image_byte_for_byte(self):
        image = _export().cores[0]
        ops = neuron_readback_ops(0, image.neuron_words)
        assert len(ops) == 256 * 16
        assert ops[0].args[2] == image.neuron_words[0] & 0xFF
        assert ops[1].args[2] == (image.neuron_words[0] >> 8) & 0xFF

    def test_the_shadow_expectations_cover_every_register_and_the_sign_vector(self):
        image = _export().cores[0]
        writes = [{"address": w.address, "value": w.value}
                  for w in image.register_writes]
        ops = shadow_ops(0, writes, gate_on=False)
        assert len(ops) == 10 + 16
        assert ops[shadow_register_id("SPI_GATE_ACTIVITY")].args[2] == 0
        assert ops[shadow_register_id("SPI_OPEN_LOOP")].args[2] == 1
        assert ops[shadow_register_id("SPI_PROPAGATE_UNMAPPED_SYN")].args[2] == 1
        assert ops[shadow_syn_sign_id(0)].args[2] == image.syn_sign & 0xFFFF

    def test_an_unwritten_register_is_refused_rather_than_shadowed_against_zero(self):
        with pytest.raises(StimulusError, match="never writes"):
            shadow_ops(0, [{"address": 0, "value": 1}], gate_on=True)

    def test_an_unknown_shadow_register_name_is_refused(self):
        with pytest.raises(StimulusError, match="not a scalar configuration"):
            shadow_register_id("SPI_BANANA")


class TestTheCosimPlan:
    def _plan(self, **kwargs):
        export = _export()
        cycles = [{0: (1, 1, 1), 1: (0, 0)}, {0: (1, 0, 1), 1: (1, 1)}]
        return export, build_cosim_ops(
            export, [cycles, cycles], latencies=(0, 1), **kwargs)

    def test_the_weights_are_programmed_once_and_the_second_sample_only_clears(self):
        export, plan = self._plan()
        first_tag = next(i for i, op in enumerate(plan.ops) if op.code == OP_TAG)
        second_tag = next(
            i for i, op in enumerate(plan.ops)
            if op.code == OP_TAG and op.args[0] == plan.tag_of(1, 0))
        between = plan.ops[first_tag:second_tag]
        writes = [op for op in between if op.code == OP_SPI_W]
        assert writes, "the second sample must at least re-CLEAR the membranes"
        gate_address = config_write_address(0)
        for op in writes:
            if op.args[1] == gate_address:
                continue
            command = (op.args[1] >> 16) & 0b11
            assert command == 0b01, "no synapse word is rewritten between samples"
            byte_addr = (op.args[1] >> 8) & 0xF
            assert byte_addr in (8, 9, 10), (
                "only the neuron word's STATE bytes are rewritten; a parameter "
                "byte would be reprogramming the network, not clearing it")

    def test_a_core_is_not_injected_before_its_own_latency(self):
        _export_, plan = self._plan()
        tag = plan.tag_of(0, 0)
        index = plan.ops.index(Op(OP_TAG, (tag,)))
        window = []
        for op in plan.ops[index + 1:]:
            if op.code == OP_TAG:
                break
            window.append(op)
        assert all(op.args[0] == 0 for op in window if op.code == OP_AER
                   and op.args[1] != all_neuron_tref_event())

    def test_the_tag_encodes_the_sample_and_the_cycle_reversibly(self):
        _export_, plan = self._plan()
        assert plan.tag_of(0, 0) == FIRST_TAG
        for sample in range(plan.samples):
            for cycle in range(plan.cycles_per_sample):
                assert plan.decode_tag(plan.tag_of(sample, cycle)) == (sample, cycle)

    def test_the_barrier_bound_comes_from_the_exported_program(self):
        export, plan = self._plan()
        barriers = [s["payload"]["cycles"] for s in export.program.stages
                    if s["kind"] == "BARRIER"]
        assert plan.barrier_cycles == max(barriers)

    def test_ragged_samples_are_refused(self):
        export = _export()
        with pytest.raises(StimulusError, match="same number of cycles"):
            build_cosim_ops(
                export,
                [[{0: (1, 1, 1), 1: (0, 0)}], []],
                latencies=(0, 1))

    def test_a_missing_core_plan_is_refused_rather_than_injected_empty(self):
        export = _export()
        with pytest.raises(StimulusError, match="no slot counts"):
            build_cosim_ops(export, [[{0: (1, 1, 1)}]], latencies=(0, 0))

    def test_the_program_summary_counts_every_opcode(self):
        _export_, plan = self._plan()
        summary = op_summary(plan.ops)
        assert summary["SPI_W"] > 0 and summary["TAG"] == 4 and summary["WAIT"] == 4
        assert sum(summary.values()) == len(plan.ops)


class TestTheCaptureProtocol:
    def test_the_line_protocol_parses_into_events_and_statistics(self):
        result = parse_capture(
            "EV 0 5 100 3\nEV 1 2 101 3\nEV 0 5 300 4\n"
            "TAGAT 3 90\nBARRIER 3 95 500\n"
            "RBSTAT 16 0\nSHSTAT 26 0\nDONE 1000 3\n")
        assert len(result.events) == 3
        assert result.counts_by_tag()[(3, 0, 5)] == 1
        assert result.cycles == 1000
        assert result.tag_opened == ((3, 90),)
        assert result.barriers[0].bound == 500
        assert result.drain_overruns() == ()

    def test_an_event_after_the_barriers_deadline_is_reported_as_an_overrun(self):
        result = parse_capture(
            "BARRIER 3 95 100\nEV 0 5 500 3\nRBSTAT 0 0\nSHSTAT 0 0\nDONE 600 1\n")
        assert result.drain_overruns() == ((3, 500, 195),)

    def test_a_fatal_line_raises_instead_of_returning_a_verdict(self):
        with pytest.raises(_TestbenchFailure, match="aborted"):
            parse_capture("FATAL aer_ack_timeout core=0 addr=7\n")

    def test_a_run_without_a_done_line_raises(self):
        with pytest.raises(_TestbenchFailure, match="DONE"):
            parse_capture("EV 0 1 2 3\n")


class TestTheCycleReference:
    def test_the_gather_mirrors_the_nevresim_span_walk(self):
        mapping = _two_core_mapping()
        gathered = gather_axon_counts(mapping, [[2, 0], [0, 0]], [1, 3])
        assert gathered[0] == (1, 3, 1)     # two inputs, then the always-on tail
        assert gathered[1] == (2, 0)        # core 0's previous outputs

    def test_an_off_source_delivers_nothing(self):
        mapping = _two_core_mapping()
        mapping.cores[1].axon_sources = [
            SpikeSource(0, 0, is_off=True), SpikeSource(0, 1)]
        assert gather_axon_counts(mapping, [[9, 4], [0, 0]], [0, 0])[1] == (0, 4)

    def test_a_source_outside_the_geometry_it_reads_is_refused_by_name(self):
        mapping = _two_core_mapping()
        mapping.cores[1].axon_sources = [SpikeSource(0, 9), SpikeSource(0, 1)]
        with pytest.raises(ReferenceTraceError, match="neuron 9"):
            gather_axon_counts(mapping, [[0, 0], [0, 0]], [0, 0])
        mapping.cores[1].axon_sources = [SpikeSource(7, 0), SpikeSource(0, 1)]
        with pytest.raises(ReferenceTraceError, match="reads core 7"):
            gather_axon_counts(mapping, [[0, 0], [0, 0]], [0, 0])

    def test_an_input_line_outside_the_raster_is_refused_rather_than_zeroed(self):
        mapping = _two_core_mapping()
        with pytest.raises(ReferenceTraceError, match="input line"):
            gather_axon_counts(mapping, [[0, 0], [0, 0]], [1])

    def test_a_core_does_not_compute_before_its_own_latency(self):
        mapping = _two_core_mapping()
        trace = simulate_cycles(
            mapping, soma_law=ODIN_LAW, input_counts=[[1, 1]] * 3,
            simulation_length=3)
        assert trace.latencies == (0, 1)
        assert trace.outputs[0][1] == (0, 0)

    def test_the_window_counts_use_each_cores_own_latency_shifted_window(self):
        mapping = _two_core_mapping()
        trace = simulate_cycles(
            mapping, soma_law=ODIN_LAW, input_counts=[[1, 1]] * 4,
            simulation_length=3)
        windows = trace.window_counts()
        assert len(windows) == 2
        manual = [0, 0]
        for cycle in range(1, 4):
            for neuron in range(2):
                manual[neuron] += trace.outputs[cycle][1][neuron]
        assert list(windows[1]) == manual
