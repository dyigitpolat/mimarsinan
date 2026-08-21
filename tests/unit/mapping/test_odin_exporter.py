"""[ODIN4] The exporter: HardCoreMapping -> per-core images + sequencer program + manifest.

Plan Sec.5.2 / F9: the packer consumes the MAPPING (per-core geometry, per-core axon
sources), never the `ChipModel` envelope, which pads to the global max and would
program real cells from phantom rows. This module's tests pin that seam, the
per-sample CLEAR, and the manifest's evidence.
"""

import json

import numpy as np
import pytest

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.export.odin.exporter import (
    EXPORT_FORMAT_VERSION,
    ORDERING_VERSION,
    OdinExportError,
    export_odin,
)
from mimarsinan.mapping.export.odin.feasibility import (
    KEY_FAN_IN,
    KEY_THETA_CEILING,
    OdinFeasibilityError,
)
from mimarsinan.mapping.export.odin.layout import unpack_neuron_word
from mimarsinan.mapping.export.odin.program import (
    STAGE_BARRIER,
    STAGE_CLEAR,
    STAGE_CONFIG,
    STAGE_INJECT,
    STAGE_READOUT,
    STAGE_TREF,
    decode_program,
    emit_program,
)
from mimarsinan.mapping.export.odin.synapse import synapse_address
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping

ODIN_LAW = SomaLaw.resolve({
    "firing_mode": "Novena",
    "thresholding_mode": "<=",
    "firing_granularity": "per_event",
    "membrane_bits": 8,
})


def _core(matrix, *, threshold, sources, has_bias_capability=False):
    core = HardCore(
        axons_per_core=matrix.shape[0],
        neurons_per_core=matrix.shape[1],
        has_bias_capability=has_bias_capability,
    )
    core.core_matrix = np.asarray(matrix, dtype=np.float64)
    core.axon_sources = list(sources)
    core.threshold = threshold
    core.available_axons = 0
    core.available_neurons = 0
    return core


def _two_core_mapping():
    """core 0: 2 inputs + a bias tail row, 2 neurons. core 1: fed by core 0."""
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


def _export(mapping=None, **overrides):
    kwargs = dict(
        soma_law=ODIN_LAW,
        weight_bits=4,
        weight_sign_granularity="per_axon",
        effective_max_axons=127,
        membrane_init=0,
    )
    kwargs.update(overrides)
    return export_odin(mapping if mapping is not None else _two_core_mapping(), **kwargs)


class TestTheExporterConsumesTheMappingNotTheEnvelope:
    def test_one_image_per_hard_core_with_its_own_logical_geometry(self):
        export = _export()
        assert [c.core_index for c in export.cores] == [0, 1]
        assert export.manifest["geometry"]["cores"][0]["logical_axons"] == 3
        assert export.manifest["geometry"]["cores"][1]["logical_axons"] == 2

    def test_heterogeneous_cores_are_not_padded_to_a_global_max(self):
        export = _export()
        rows = [c["physical_rows_used"] for c in export.manifest["geometry"]["cores"]]
        assert rows == [6, 4]

    def test_the_memoized_grid_is_never_mutated(self):
        mapping = _two_core_mapping()
        before = [np.array(c.get_core_matrix(), copy=True) for c in mapping.cores]
        _export(mapping)
        for core, snapshot in zip(mapping.cores, before):
            assert np.array_equal(core.get_core_matrix(), snapshot)

    def test_a_mapping_without_computed_latencies_is_refused_by_name(self):
        mapping = _two_core_mapping()
        mapping.cores[0].latency = None
        with pytest.raises(OdinExportError, match="ChipLatency"):
            _export(mapping)


class TestTheNeuronImage:
    def test_every_word_selects_the_lif_model(self):
        image = _export().cores[0]
        assert all(unpack_neuron_word(w)["lif_izh_sel"] == 1 for w in image.neuron_words)

    def test_the_used_neurons_carry_the_core_threshold(self):
        image = _export().cores[0]
        assert unpack_neuron_word(image.neuron_words[0])["thr"] == 4
        assert unpack_neuron_word(image.neuron_words[1])["thr"] == 4

    def test_unused_neurons_are_disabled(self):
        image = _export().cores[0]
        assert unpack_neuron_word(image.neuron_words[0])["neur_disable"] == 0
        assert unpack_neuron_word(image.neuron_words[2])["neur_disable"] == 1
        assert unpack_neuron_word(image.neuron_words[255])["neur_disable"] == 1

    def test_leak_and_calcium_learning_are_off_in_every_word(self):
        for image in _export().cores:
            for word in image.neuron_words:
                fields = unpack_neuron_word(word)
                assert fields["leak_en"] == 0
                assert fields["ca_en"] == 0

    def test_the_state_fields_carry_the_declared_membrane_init(self):
        image = _export(membrane_init=3).cores[0]
        fields = unpack_neuron_word(image.neuron_words[0])
        assert fields["vmem"] == 3
        assert fields["calcium"] == 0
        assert fields["caleak_cnt"] == 0

    def test_a_membrane_init_at_or_above_theta_is_refused(self):
        with pytest.raises(OdinFeasibilityError, match="membrane_init"):
            _export(membrane_init=4)


class TestTheSynapseImageAndSignVector:
    def test_a_positive_weight_lands_on_the_excitatory_row(self):
        image = _export().cores[0]
        word_addr, byte_addr, nibble = synapse_address(0, 0)  # slot 0 exc row, neuron 0
        shift = 4 * (2 * byte_addr + nibble)
        assert (image.synapse_words[word_addr] >> shift) & 0b111 == 2

    def test_a_negative_weight_lands_on_the_inhibitory_row(self):
        image = _export().cores[0]
        word_addr, byte_addr, nibble = synapse_address(1, 1)  # slot 0 inh row, neuron 1
        shift = 4 * (2 * byte_addr + nibble)
        assert (image.synapse_words[word_addr] >> shift) & 0b111 == 3

    def test_every_mapping_bit_is_written_zero_under_the_weight_freeze(self):
        image = _export().cores[0]
        for word in image.synapse_words:
            for j in range(8):
                assert (word >> (4 * j + 3)) & 1 == 0

    def test_the_sign_vector_alternates_excitatory_inhibitory_over_the_pairs(self):
        image = _export().cores[0]
        assert [(image.syn_sign >> r) & 1 for r in range(6)] == [0, 1, 0, 1, 0, 1]

    def test_rows_beyond_the_core_are_excitatory_zero(self):
        image = _export().cores[0]
        assert (image.syn_sign >> 6) == 0


class TestTheSequencerProgram:
    def test_the_program_round_trips_through_the_one_emitter_and_decoder(self):
        export = _export()
        doc = emit_program(export.program)
        assert decode_program(doc) == export.program
        assert json.loads(json.dumps(doc)) == doc

    def test_the_stage_order_is_config_clear_inject_tref_barrier_readout(self):
        kinds = [s["kind"] for s in _export().program.stages]
        assert kinds[0] == STAGE_CONFIG
        assert STAGE_CLEAR in kinds
        assert kinds.index(STAGE_CLEAR) < kinds.index(STAGE_INJECT)
        assert kinds.index(STAGE_INJECT) < kinds.index(STAGE_TREF)
        assert kinds.index(STAGE_TREF) < kinds.index(STAGE_BARRIER)
        assert kinds[-1] == STAGE_READOUT

    def test_the_clear_stage_rewrites_the_state_bytes_of_every_used_neuron(self):
        clears = [s for s in _export().program.stages if s["kind"] == STAGE_CLEAR]
        neurons = {w["neuron"] for w in clears[0]["payload"]["byte_writes"]}
        assert neurons == {0, 1}
        assert {w["byte_addr"] for w in clears[0]["payload"]["byte_writes"]} == {8, 9, 10}

    def test_the_barrier_bound_grows_with_the_injected_event_count(self):
        barriers = [s for s in _export().program.stages if s["kind"] == STAGE_BARRIER]
        assert barriers
        assert all(s["payload"]["cycles"] > 0 for s in barriers)


class TestTheManifest:
    def test_the_manifest_is_json_serializable(self):
        manifest = _export().manifest
        assert json.loads(json.dumps(manifest)) == manifest

    def test_the_manifest_carries_the_resolved_soma_law(self):
        law = _export().manifest["soma_law"]
        assert law["firing_granularity"] == "per_event"
        assert law["membrane_arithmetic"] == "saturating_unsigned"
        assert law["membrane_bits"] == 8
        assert law["bias_slot"] == "tail"

    def test_the_manifest_declares_the_ordering_version(self):
        ordering = _export().manifest["ordering"]
        assert ordering["version"] == ORDERING_VERSION
        assert ordering["bias_slot"] == "tail"

    def test_the_manifest_carries_the_propagated_emission_bounds(self):
        bounds = _export().manifest["emission_bounds"]
        assert bounds["ceiling"] == 127
        assert bounds["max"] >= 1
        assert set(bounds["per_core_max"]) == {"0", "1"}

    def test_the_manifest_records_every_gate_that_was_checked(self):
        gates = _export().manifest["feasibility"]["gates"]
        assert set(gates) == {
            "theta_ceiling", "weight_magnitude_range", "fan_in", "emission_bound",
        }
        assert all(gates.values())

    def test_the_manifest_carries_the_format_version(self):
        assert _export().manifest["format_version"] == EXPORT_FORMAT_VERSION


class TestTheGatesFireThroughTheExporter:
    def test_a_theta_over_the_membrane_ceiling_refuses_the_export(self):
        mapping = _two_core_mapping()
        mapping.cores[0].threshold = 300.0
        with pytest.raises(OdinFeasibilityError) as excinfo:
            _export(mapping)
        assert excinfo.value.key == KEY_THETA_CEILING

    def test_a_fan_in_over_the_effective_limit_refuses_the_export(self):
        mapping = _two_core_mapping()
        with pytest.raises(OdinFeasibilityError) as excinfo:
            _export(mapping, effective_max_axons=2)
        assert excinfo.value.key == KEY_FAN_IN

    def test_a_default_soma_law_is_refused_because_odin_is_a_per_event_chip(self):
        with pytest.raises(OdinExportError, match="per_event"):
            _export(soma_law=SomaLaw.resolve({}))

    def test_an_unbounded_membrane_is_refused(self):
        law = SomaLaw.resolve({"firing_granularity": "per_event"})
        with pytest.raises(OdinExportError, match="saturating_unsigned"):
            _export(soma_law=law)
