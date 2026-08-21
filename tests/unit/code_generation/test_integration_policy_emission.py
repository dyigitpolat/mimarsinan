"""[ODIN P3] the integration-policy axis at the ONE emit site.

The axis is a defaulted C++ template parameter, so the default point's emitted
program must be BYTE-IDENTICAL to the one that predates the axis — the
argument is appended only when it is not the default. The non-default point
names the event-serial fold in both strings (the chip's compute policy and the
execution's), which is what the ``compute_policy_t`` identity static_assert
inside nevresim requires, and switches the carry seam to the counted generator.
"""

from pathlib import Path

import pytest

from mimarsinan.chip_simulation.nevresim_policy_types import (
    NevresimPolicyTypeError,
    nevresim_integration_policy,
)
from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW, SomaLaw
from mimarsinan.code_generation.generate_main import (
    generate_main_function,
    generate_main_function_runtime,
    get_config,
    resolve_exec_policy,
)

_PER_EVENT_8 = SomaLaw.resolve({
    "spiking_family": "lif", "spiking_variant": "streamed",
    "firing_mode": "Novena", "firing_granularity": "per_event",
    "membrane_bits": 8,
})


def _spec(**overrides):
    kwargs = dict(
        spiking_mode="lif", firing_mode="Novena", thresholding_mode="<=",
        spike_gen_mode="SpikeTrain", weight_type="int",
        simulation_length=6, latency=1, output_count=2,
    )
    kwargs.update(overrides)
    return resolve_exec_policy(**kwargs)


class TestTheResolverMapsThePoint:
    def test_the_default_point_resolves_to_the_defaulted_template_argument(self):
        assert nevresim_integration_policy(None) == "WholeVectorIntegrate"
        assert nevresim_integration_policy(DEFAULT_SOMA_LAW) == "WholeVectorIntegrate"

    def test_the_per_event_point_carries_its_register_width(self):
        assert nevresim_integration_policy(_PER_EVENT_8) == "EventSerialIntegrate<8>"
        wide = SomaLaw.resolve({
            "spiking_family": "lif", "spiking_variant": "streamed",
            "firing_mode": "Novena", "firing_granularity": "per_event",
            "membrane_bits": 16,
        })
        assert nevresim_integration_policy(wide) == "EventSerialIntegrate<16>"

    def test_an_unbounded_per_event_point_refuses_by_name(self):
        law = SomaLaw(
            firing_mode="Novena", thresholding_mode="<=",
            firing_granularity="per_event", membrane_arithmetic="unbounded",
            membrane_bits=0,
        )
        with pytest.raises(NevresimPolicyTypeError, match="membrane_bits"):
            nevresim_integration_policy(law)

    def test_a_saturating_per_cycle_point_refuses_by_name(self):
        law = SomaLaw(
            firing_mode="Default", thresholding_mode="<=",
            firing_granularity="per_cycle",
            membrane_arithmetic="saturating_unsigned", membrane_bits=8,
        )
        with pytest.raises(NevresimPolicyTypeError, match="per_cycle"):
            nevresim_integration_policy(law)


class TestTheDefaultPointEmitsNothingNew:
    def test_the_compute_policy_string_is_unchanged(self):
        assert _spec().compute_policy == (
            "SpikingCompute<LIFirePolicy<ZeroReset, InclusiveCompare>>")

    def test_the_exec_decl_string_is_unchanged(self):
        assert _spec().exec_decl == (
            "using exec = SpikingExecution<6, 1, 2, SpikeTrainSpikeGenerator, "
            "int, LIFirePolicy<ZeroReset, InclusiveCompare>>;")

    @pytest.mark.parametrize(
        "emit", [generate_main_function, generate_main_function_runtime])
    def test_the_emitted_main_cpp_is_byte_identical_to_the_pre_axes_program(
        self, emit, tmp_path,
    ):
        """The A/B the default-off claim rests on: a program emitted with the
        resolved default point is the same BYTES as one emitted by a caller
        that never heard of the axis."""
        written = []
        # ONE output directory: the generated path is embedded in the program
        # text, so the A/B must differ in the soma point and nothing else.
        for soma_law in (None, DEFAULT_SOMA_LAW):
            emit(str(tmp_path), 4, 2, 6, 1,
                 simulation_config=get_config(
                     "Uniform", "Novena", "int", "lif", threshold_type="int",
                     thresholding_mode="<=", soma_law=soma_law))
            written.append((tmp_path / "main" / "main.cpp").read_bytes())
        assert written[0] == written[1]
        assert b"IntegrationPolicy" not in written[0]
        assert b"EventSerialIntegrate" not in written[0]


class TestThePointNamesTheFoldInBothStrings:
    def test_both_emitted_types_name_the_same_integration_policy(self):
        spec = _spec(integration_policy="EventSerialIntegrate<8>")
        assert spec.compute_policy == (
            "SpikingCompute<LIFirePolicy<ZeroReset, InclusiveCompare>, "
            "EventSerialIntegrate<8>>")
        assert spec.exec_decl.endswith(
            "LIFirePolicy<ZeroReset, InclusiveCompare>, "
            "EventSerialIntegrate<8>>;")

    def test_the_carry_seam_switches_to_the_counted_generator(self):
        """A per-event producer emits COUNTS; a binarizing loader would deliver
        1 where the producing segment fired 3."""
        spec = _spec(integration_policy="EventSerialIntegrate<8>")
        assert "CountedSpikeTrainSpikeGenerator" in spec.exec_decl

    def test_a_value_mode_input_keeps_its_own_generator(self):
        spec = _spec(spike_gen_mode="Uniform",
                     integration_policy="EventSerialIntegrate<8>")
        assert "UniformSpikeGenerator" in spec.exec_decl
        assert "Counted" not in spec.exec_decl.split(",")[3]

    def test_the_emitted_program_names_the_policy(self, tmp_path):
        generate_main_function_runtime(
            str(tmp_path), 4, 2, 6, 1,
            simulation_config=get_config(
                "SpikeTrain", "Novena", "int", "lif", threshold_type="int",
                thresholding_mode="<=", soma_law=_PER_EVENT_8))
        text = Path(tmp_path / "main" / "main.cpp").read_text()
        assert text.count("EventSerialIntegrate<8>") == 2
        assert "CountedSpikeTrainSpikeGenerator" in text
