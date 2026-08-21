"""[ODIN4] The sequencer program: one versioned schema, ONE emitter and ONE decoder.

Plan Sec.4 ("Sequencer program format") and Sec.7 row 14. The cosim testbench (P5) and the
XRT runtime (P7) READ what the exporter writes, so the round trip is the contract:
`decode_program(emit_program(p)) == p`, byte-for-byte on the JSON document.
"""

import json

import pytest

from mimarsinan.mapping.export.odin.program import (
    AEROUT_HANDSHAKE_CYCLES,
    NEURON_SWEEP_CYCLES,
    SCHEDULER_FIFO_DEPTH,
    SCHEDULER_PUSH_CYCLES,
    SEQUENCER_SCHEMA_VERSION,
    STAGE_BARRIER,
    STAGE_CLEAR,
    STAGE_CONFIG,
    STAGE_INJECT,
    STAGE_KINDS,
    STAGE_READOUT,
    STAGE_TREF,
    SequencerProgram,
    SequencerProgramError,
    barrier_stage,
    clear_stage,
    config_stage,
    decode_program,
    drain_bound_cycles,
    emit_program,
    inject_stage,
    plan_injection,
    readout_stage,
    tref_stage,
)
from mimarsinan.mapping.export.odin.registers import RegisterWrite


class TestTheSchemaIsVersionedAndClosed:
    def test_the_schema_version_is_one(self):
        assert SEQUENCER_SCHEMA_VERSION == 1

    def test_the_six_stage_kinds_are_the_declared_ones(self):
        assert STAGE_KINDS == (
            STAGE_CONFIG, STAGE_CLEAR, STAGE_INJECT,
            STAGE_TREF, STAGE_BARRIER, STAGE_READOUT,
        )

    def test_an_unknown_stage_kind_is_refused_at_construction(self):
        with pytest.raises(SequencerProgramError, match="RESET"):
            SequencerProgram(stages=({"kind": "RESET", "payload": {}},))

    def test_a_foreign_schema_version_is_refused_by_the_decoder(self):
        doc = emit_program(SequencerProgram(stages=(tref_stage(scope="all"),)))
        doc["schema_version"] = 2
        with pytest.raises(SequencerProgramError, match="schema_version"):
            decode_program(doc)


def _program():
    return SequencerProgram(stages=(
        config_stage(
            core_index=0,
            register_writes=(RegisterWrite(address=0, value=1),
                             RegisterWrite(address=1, value=1)),
            neuron_words=(0xABCDEF, 0x1234),
            synapse_words=(0, 15, 0),
        ),
        clear_stage(core_index=0, byte_writes=(
            {"neuron": 0, "byte_addr": 9, "value": 0, "mask": 0},
        )),
        inject_stage(core_index=0, events=((0, 2), (5, 1))),
        tref_stage(scope="all"),
        barrier_stage(cycles=1234),
        readout_stage(core_index=0, neurons=(0, 1, 2)),
    ))


class TestTheGoldenRoundTrip:
    def test_emit_then_decode_reproduces_the_program(self):
        program = _program()
        assert decode_program(emit_program(program)) == program

    def test_the_document_is_json_serializable_and_stable(self):
        doc = emit_program(_program())
        text = json.dumps(doc, sort_keys=True)
        assert json.dumps(emit_program(decode_program(json.loads(text))),
                          sort_keys=True) == text

    def test_the_document_carries_the_version_and_the_stage_order(self):
        doc = emit_program(_program())
        assert doc["schema_version"] == SEQUENCER_SCHEMA_VERSION
        assert [s["kind"] for s in doc["stages"]] == list(STAGE_KINDS)

    def test_a_truncated_document_is_refused(self):
        with pytest.raises(SequencerProgramError, match="stages"):
            decode_program({"schema_version": SEQUENCER_SCHEMA_VERSION})


class TestTheClearStageIsPerSample:
    def test_clear_carries_masked_neuron_state_byte_writes(self):
        stage = clear_stage(core_index=1, byte_writes=(
            {"neuron": 7, "byte_addr": 8, "value": 0, "mask": 0b00111111},
        ))
        assert stage["kind"] == STAGE_CLEAR
        assert stage["payload"]["core_index"] == 1
        assert stage["payload"]["byte_writes"][0]["mask"] == 0b00111111

    def test_a_byte_write_missing_a_field_is_refused(self):
        with pytest.raises(SequencerProgramError, match="mask"):
            clear_stage(core_index=0, byte_writes=({"neuron": 0, "byte_addr": 8,
                                                    "value": 0},))


class TestTheInjectStageCarriesSlotMultiplicityPairs:
    def test_injection_is_drained_ascending_with_multiplicity_adjacent(self):
        # counts per LOGICAL slot -> emitting PHYSICAL rows, ascending.
        events = plan_injection(
            counts=[2, 0, 1],
            emitting_rows=lambda slot: {0: (0,), 1: (), 2: (4, 5)}[slot],
        )
        assert events == ((0, 2), (4, 1), (5, 1))

    def test_a_slot_that_emits_nothing_contributes_no_events(self):
        events = plan_injection(
            counts=[3, 3], emitting_rows=lambda slot: () if slot == 0 else (2,)
        )
        assert events == ((2, 3),)

    def test_a_zero_count_slot_is_dropped_by_the_canonical_drain(self):
        events = plan_injection(counts=[0, 0, 0], emitting_rows=lambda s: (2 * s,))
        assert events == ()

    def test_a_negative_multiplicity_is_refused(self):
        with pytest.raises(ValueError, match="multiplicity"):
            inject_stage(core_index=0, events=((0, -1),))


class TestTheBarrierCarriesADeterministicDrainBound:
    def test_the_constants_are_the_upstream_ones(self):
        assert SCHEDULER_PUSH_CYCLES == 1
        assert NEURON_SWEEP_CYCLES == 512
        assert SCHEDULER_FIFO_DEPTH == 32

    def test_the_closed_form_is_the_documented_one(self):
        bound = drain_bound_cycles(injected_events=10, emitted_spike_bound=3)
        assert bound == (
            (10 + SCHEDULER_FIFO_DEPTH) * (SCHEDULER_PUSH_CYCLES + NEURON_SWEEP_CYCLES)
            + 3 * AEROUT_HANDSHAKE_CYCLES
        )

    def test_the_bound_is_monotone_in_both_arguments(self):
        base = drain_bound_cycles(injected_events=10, emitted_spike_bound=3)
        assert drain_bound_cycles(injected_events=11, emitted_spike_bound=3) > base
        assert drain_bound_cycles(injected_events=10, emitted_spike_bound=4) > base

    def test_an_empty_injection_still_reserves_the_queue_depth(self):
        assert drain_bound_cycles(injected_events=0, emitted_spike_bound=0) == (
            SCHEDULER_FIFO_DEPTH * (SCHEDULER_PUSH_CYCLES + NEURON_SWEEP_CYCLES)
        )

    def test_negative_inputs_are_refused(self):
        with pytest.raises(SequencerProgramError, match="non-negative"):
            drain_bound_cycles(injected_events=-1, emitted_spike_bound=0)

    def test_the_barrier_stage_carries_the_cycle_count_field(self):
        stage = barrier_stage(cycles=drain_bound_cycles(
            injected_events=4, emitted_spike_bound=2))
        assert stage["kind"] == STAGE_BARRIER
        assert stage["payload"]["cycles"] == 36 * 513 + 2 * AEROUT_HANDSHAKE_CYCLES
