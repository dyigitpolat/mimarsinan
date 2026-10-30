"""[ODIN6] Packer tables and the generated core's feasibility gates.

The tables are what the configuration port executes, so they are gated the way
the stock exporter's images are: the packing round-trips over the WHOLE declared
geometry, every neuron is programmed (an unwritten threshold register would fire
on its own zero membrane), and each refusal names the declaration at fault.

The no-saturation bound is the sync-fire law's whole contract: it proves,
statically over the mapped weights, that neither rail of the two's-complement
register is reachable — which is the only condition under which that law holds
the number the unbounded accumulator holds.
"""

import numpy as np
import pytest

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.export.odin.feasibility import OdinFeasibilityError
from mimarsinan.mapping.export.odin_gen.feasibility import (
    KEY_NO_SATURATION,
    KEY_VARIANT_THETA,
    check_variant_theta,
    core_saturation_bound,
    require_no_saturation,
)
from mimarsinan.mapping.export.odin_gen.packer import (
    PROG_SEL_MEMBRANE,
    PROG_SEL_REGISTER,
    PROG_SEL_SYNAPSE,
    PROG_SEL_THRESHOLD,
    VariantPackError,
    build_variant_core_image,
    decode_weight,
    encode_weight,
    gate_write,
    pack_synapse_words,
    silent_threshold,
    unpack_synapse_words,
)
from mimarsinan.mapping.export.odin_gen.spec import CoreSpec
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping

_LIF = {"spiking_family": "lif", "spiking_variant": "streamed",
        "firing_mode": "Novena", "thresholding_mode": "<="}
_PER_EVENT = SomaLaw.resolve({**_LIF, "firing_granularity": "per_event",
                              "membrane_bits": 8})
_SYNC_FIRE = SomaLaw.resolve({**_LIF, "membrane_bits": 16,
                              "membrane_signed": True})


def _spec(law=_PER_EVENT, axons=16, neurons=16):
    return CoreSpec.project(
        {"max_axons": axons, "max_neurons": neurons, "count": 1,
         "has_bias": False},
        soma_law=law, weight_bits=4, weight_sign_granularity="per_synapse")


def _core(matrix, *, threshold, available_neurons=0):
    values = np.asarray(matrix, dtype=np.float64)
    core = HardCore(
        axons_per_core=values.shape[0], neurons_per_core=values.shape[1],
        has_bias_capability=False)
    core.core_matrix = values
    core.axon_sources = [
        SpikeSource(-2, i, is_input=True) for i in range(values.shape[0])]
    core.threshold = float(threshold)
    core.available_axons = 0
    core.available_neurons = available_neurons
    return core


def _mapping(cores):
    mapping = HardCoreMapping(chip_cores=[])
    mapping.cores = list(cores)
    mapping.output_sources = np.asarray([SpikeSource(0, 0)], dtype=object)
    ChipLatency(mapping).calculate()
    return mapping


class TestTheSynapseCellSignsItself:
    @pytest.mark.parametrize("value", list(range(-8, 8)))
    def test_the_two_s_complement_encoding_round_trips(self, value):
        assert decode_weight(encode_weight(value, weight_bits=4),
                             weight_bits=4) == value

    def test_a_weight_outside_the_cell_is_refused_not_truncated(self):
        with pytest.raises(VariantPackError, match=r"\[-8, 7\]"):
            encode_weight(8, weight_bits=4)


class TestTheSynapseImageRoundTrips:
    def test_a_full_grid_survives_pack_and_unpack(self):
        spec = _spec()
        rng = np.random.default_rng(11)
        grid = rng.integers(-8, 8, size=(16, 16)).astype(np.float64)
        words = pack_synapse_words(grid, spec=spec)
        assert len(words) == spec.synapse_depth
        restored = unpack_synapse_words(words, spec=spec)
        assert np.array_equal(np.asarray(restored, dtype=np.int64),
                              grid.astype(np.int64))

    def test_a_smaller_mapping_pads_with_the_no_op_weight(self):
        spec = _spec(axons=32, neurons=32)
        grid = np.full((4, 4), 3.0)
        restored = np.asarray(
            unpack_synapse_words(pack_synapse_words(grid, spec=spec), spec=spec))
        assert np.array_equal(restored[:4, :4], np.full((4, 4), 3))
        assert not restored[4:, :].any()
        assert not restored[:, 4:].any()

    def test_a_grid_over_the_declared_geometry_is_refused(self):
        with pytest.raises(VariantPackError, match="does not fit"):
            pack_synapse_words(np.zeros((32, 16)), spec=_spec())

    def test_a_fractional_grid_is_refused_rather_than_rounded(self):
        with pytest.raises(VariantPackError, match="integral"):
            pack_synapse_words(np.full((4, 4), 1.5), spec=_spec())

    def test_the_grid_is_not_mutated(self):
        grid = np.full((4, 4), -2.0)
        pack_synapse_words(grid, spec=_spec())
        assert np.array_equal(grid, np.full((4, 4), -2.0))


class TestTheCoreImageProgramsEveryNeuron:
    def test_every_declared_neuron_gets_a_threshold_and_a_membrane(self):
        spec = _spec(axons=16, neurons=32)
        image = build_variant_core_image(
            _core(np.zeros((16, 16)), threshold=4), spec=spec, core_index=0,
            theta=4, membrane_init=0)
        assert len(image.threshold_writes) == spec.max_neurons
        assert len(image.membrane_writes) == spec.max_neurons
        assert len(image.synapse_writes) == spec.synapse_depth
        assert {w.sel for w in image.threshold_writes} == {PROG_SEL_THRESHOLD}
        assert {w.sel for w in image.membrane_writes} == {PROG_SEL_MEMBRANE}
        assert {w.sel for w in image.synapse_writes} == {PROG_SEL_SYNAPSE}

    def test_an_unmapped_neuron_is_silenced_by_the_highest_threshold(self):
        """A neuron left at theta=0 fires on the inclusive compare against its
        own zero membrane and injects spikes the mapping never produced."""
        spec = _spec(axons=16, neurons=16)
        image = build_variant_core_image(
            _core(np.zeros((16, 16)), threshold=4, available_neurons=12),
            spec=spec, core_index=0, theta=4, membrane_init=0)
        data = [write.data for write in image.threshold_writes]
        assert data[:4] == [4] * 4
        assert data[4:] == [silent_threshold(spec)] * 12
        assert silent_threshold(spec) == spec.membrane_high

    def test_a_signed_membrane_init_is_stored_in_two_s_complement(self):
        spec = _spec(law=_SYNC_FIRE, axons=16, neurons=16)
        image = build_variant_core_image(
            _core(np.zeros((16, 16)), threshold=4), spec=spec, core_index=0,
            theta=4, membrane_init=-3)
        assert image.membrane_writes[0].data == (1 << 16) - 3

    def test_a_membrane_init_outside_the_register_is_refused(self):
        with pytest.raises(VariantPackError, match="interval"):
            build_variant_core_image(
                _core(np.zeros((16, 16)), threshold=4), spec=_spec(),
                core_index=0, theta=4, membrane_init=999)

    def test_a_weight_outside_the_declared_width_is_refused_by_key(self):
        with pytest.raises(OdinFeasibilityError):
            build_variant_core_image(
                _core(np.full((16, 16), 9.0), threshold=4), spec=_spec(),
                core_index=0, theta=4, membrane_init=0)

    def test_a_mapping_over_the_declared_geometry_is_refused(self):
        with pytest.raises(VariantPackError, match="does not fit"):
            build_variant_core_image(
                _core(np.zeros((32, 32)), threshold=4), spec=_spec(),
                core_index=0, theta=4, membrane_init=0)

    def test_the_gate_register_is_the_only_configuration_register(self):
        assert gate_write(on=True).sel == PROG_SEL_REGISTER
        assert gate_write(on=True).data == 1
        assert gate_write(on=False).data == 0


class TestTheThresholdShareTheMembraneRegister:
    def test_a_theta_inside_the_register_is_admitted(self):
        assert check_variant_theta(
            300, spec=_spec(law=_SYNC_FIRE), core_index=0) == 300

    def test_a_theta_over_the_register_is_refused_by_key(self):
        with pytest.raises(OdinFeasibilityError) as exc:
            check_variant_theta(300, spec=_spec(), core_index=0)
        assert exc.value.key == KEY_VARIANT_THETA
        assert "255" in str(exc.value)

    def test_a_fractional_theta_is_refused_rather_than_snapped(self):
        with pytest.raises(OdinFeasibilityError, match="integral"):
            check_variant_theta(4.5, spec=_spec(), core_index=0)

    def test_a_zero_theta_is_refused(self):
        with pytest.raises(OdinFeasibilityError):
            check_variant_theta(0, spec=_spec(), core_index=0)


class TestTheNoSaturationBound:
    def _sync_mapping(self, matrix, theta=5.0):
        return _mapping([_core(matrix, threshold=theta)])

    def test_it_is_inert_for_a_law_whose_saturation_is_the_substrate(self):
        mapping = self._sync_mapping(np.full((16, 16), 7.0), theta=4.0)
        assert require_no_saturation(
            mapping, spec=_spec(), thetas={0: 4}, cycles=8,
            membrane_init=0) == ()

    def test_the_peak_is_theta_minus_one_plus_the_positive_drive(self):
        matrix = np.zeros((16, 16))
        matrix[0][0] = 7.0
        matrix[1][0] = 7.0
        bound = core_saturation_bound(
            _core(matrix, threshold=5.0), spec=_spec(law=_SYNC_FIRE),
            core_index=0, theta=5, cycles=4, membrane_init=0)
        assert bound.highest == 5 - 1 + 14

    def test_the_floor_is_the_negative_drive_over_every_cycle(self):
        matrix = np.zeros((16, 16))
        matrix[0][0] = -7.0
        bound = core_saturation_bound(
            _core(matrix, threshold=5.0), spec=_spec(law=_SYNC_FIRE),
            core_index=0, theta=5, cycles=4, membrane_init=0)
        assert bound.lowest == -28

    def test_a_bound_inside_the_register_passes(self):
        matrix = np.zeros((16, 16))
        matrix[0][0] = 7.0
        matrix[1][0] = -7.0
        spec = _spec(law=_SYNC_FIRE)
        bounds = require_no_saturation(
            self._sync_mapping(matrix), spec=spec, thetas={0: 5}, cycles=64,
            membrane_init=0)
        assert len(bounds) == 1 and bounds[0].inside(spec)

    def test_a_reachable_rail_is_refused_by_key_with_its_remedies(self):
        spec = CoreSpec.project(
            {"max_axons": 512, "max_neurons": 16, "count": 1, "has_bias": False},
            soma_law=SomaLaw.resolve({**_LIF, "membrane_bits": 8,
                                      "membrane_signed": True}),
            weight_bits=4, weight_sign_granularity="per_synapse")
        matrix = np.zeros((512, 16))
        matrix[:, 0] = 7.0
        with pytest.raises(OdinFeasibilityError) as exc:
            require_no_saturation(
                self._sync_mapping(matrix, theta=5.0), spec=spec,
                thetas={0: 5}, cycles=4, membrane_init=0)
        assert exc.value.key == KEY_NO_SATURATION
        assert "membrane_bits" in str(exc.value)
        assert "never clamped" not in str(exc.value)

    def test_the_bound_is_strict_so_the_rail_itself_is_refused(self):
        spec = CoreSpec.project(
            {"max_axons": 32, "max_neurons": 16, "count": 1, "has_bias": False},
            soma_law=SomaLaw.resolve({**_LIF, "membrane_bits": 8,
                                      "membrane_signed": True}),
            weight_bits=4, weight_sign_granularity="per_synapse")
        matrix = np.zeros((32, 16))
        # theta-1 + 18*7 = 127 exactly: representable, and still refused.
        matrix[:18, 0] = 7.0
        with pytest.raises(OdinFeasibilityError):
            require_no_saturation(
                self._sync_mapping(matrix, theta=2.0), spec=spec,
                thetas={0: 2}, cycles=1, membrane_init=0)

    def test_only_the_mapped_neurons_are_bounded(self):
        matrix = np.zeros((16, 16))
        matrix[0][15] = 7.0
        bound = core_saturation_bound(
            _core(matrix, threshold=5.0, available_neurons=15),
            spec=_spec(law=_SYNC_FIRE), core_index=0, theta=5, cycles=4,
            membrane_init=0)
        assert bound.highest == 4
