"""[E4] Core initialization is measured on Loihi, so it may not stay zero.

Every pass resets its cores' neuron state (that is what ``segment_cores``
counts), and both the energy and the wall of that reset were absent from every
profile. The owner's rule: a component that is declared or derivable may not be
left at zero; one that is neither stays absent and says so.

Loihi turned out to be better than derivable. The first pass at this bounded a
reset by a full compartment sweep at Table 2's inactive-neuron-update figure
(1024 x 52 pJ = 53.2 nJ). Frady 2020 then supplied the real thing: its Table 2
carries a RESET column, measured per query, at two system sizes — and the two
agree on the paper's own claim that reset energy per chip is constant. The
measurement is ~1.7x below the sweep bound, and the published 230 us reset wall
outright refutes the sweep's timing (it would put one chip's serial reset at
695 us, three times the whole system's measured wall).

TrueNorth publishes no per-neuron update figure and no reset measurement, so its
core init has neither and is not invented here.
"""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.platform_physics import get_platform_physics

#: Frady 2020 Table 2, Reset column: 3.06 mJ per query on the 768-chip
#: 1M-pattern configuration, 0.19 mJ on the 76,800-pattern one (48 chips x
#: 1600 patterns). 128 neuromorphic cores per Loihi chip.
RESET_MJ_1M, CHIPS_1M = 3.06, 768
RESET_MJ_SMALL, CHIPS_SMALL = 0.19, 48
CORES_PER_CHIP = 128
#: Sec 5.4: "the reset time of 230us on Pohoiki Springs".
RESET_WALL_US = 230.0


def _per_core_j(reset_mj, chips):
    return reset_mj * 1e-3 / chips / CORES_PER_CHIP


class TestTheMeasuredReset:
    def test_the_band_spans_the_two_published_configurations(self):
        band = get_platform_physics("loihi").band("e_core_init")
        assert band.low == pytest.approx(
            _per_core_j(RESET_MJ_SMALL, CHIPS_SMALL), rel=1e-3)
        assert band.nominal == pytest.approx(
            _per_core_j(RESET_MJ_1M, CHIPS_1M), rel=1e-3)

    def test_the_two_configurations_agree_on_the_per_chip_constant(self):
        """The paper claims reset energy per chip is constant per query; the
        two rows check that claim against each other, which is what makes this
        a decomposition rather than a guess."""
        big = RESET_MJ_1M / CHIPS_1M
        small = RESET_MJ_SMALL / CHIPS_SMALL
        assert abs(big - small) / small < 0.01

    def test_the_wall_is_the_published_reset_serialized_over_a_chips_cores(self):
        band = get_platform_physics("loihi").band("t_core_init")
        assert band.nominal == pytest.approx(
            RESET_WALL_US * 1e-6 / CORES_PER_CHIP, rel=1e-3)

    def test_the_measurement_undercuts_the_compartment_sweep_bound(self):
        """The bound this replaced: 1024 compartments at Table 2's inactive
        neuron update. Keeping the looser number would have overcharged every
        multi-pass program by ~1.7x on a cost that is now measured."""
        sweep_bound_j = 1024 * 52.0e-12
        assert get_platform_physics("loihi").band("e_core_init").high < sweep_bound_j

    def test_the_wall_band_admits_a_concurrent_clear(self):
        """The published wall is a whole-system figure whose intra-chip
        parallelism is unstated: a chip clearing its cores concurrently pays
        far less per core, so the low corner must not claim the serial share."""
        band = get_platform_physics("loihi").band("t_core_init")
        assert band.low < band.nominal

    @pytest.mark.parametrize("key", ["e_core_init", "t_core_init"])
    def test_the_derivation_is_written_down_and_marked_derived(self, key):
        value = get_platform_physics("loihi").constants[key]
        assert value.evidence_kind == "derived"
        assert "frady2020" in value.citation
        assert "upper bound" in value.derivation.lower()


class TestWhatIsNotMeasurableStaysAbsent:
    def test_truenorth_invents_no_core_reset_cost(self):
        physics = get_platform_physics("truenorth")
        assert not physics.has("e_neuron_update")
        assert not physics.has("e_core_init")


class TestItReachesThePrice:
    def test_a_multi_pass_program_pays_core_init_per_pass(self):
        """The multiplicand is cores over EVERY pass, so a program that cuts
        into more passes pays more init — the reason the term matters at all."""
        from mimarsinan.deployment_record.cost.absolute import price_absolute
        from mimarsinan.deployment_record.quantities.spec import (
            Quantities,
            QuantityValue,
        )

        def _priced(segment_cores):
            quantities = Quantities({
                key: QuantityValue(float(value), "static") for key, value in {
                    "synaptic_events": 1e6, "host_macs": 0, "cores_physical": 128,
                    "latency_steps": 32, "timesteps": 4, "neurons_used": 1000,
                    "segment_cores": segment_cores,
                }.items()
            })
            return {t.name: t.value for t in
                    price_absolute(quantities, get_platform_physics("loihi")).terms}

        one_pass, two_pass = _priced(4), _priced(8)
        four_more_cores_j = 4 * _per_core_j(RESET_MJ_1M, CHIPS_1M) * 1e3

        # The reset's own switching energy, isolated in the dynamic term.
        assert two_pass["energy_dynamic_mj"] - one_pass["energy_dynamic_mj"] == (
            pytest.approx(four_more_cores_j, rel=1e-3))
        # And its wall, in the steady-state headline the same event feeds.
        assert two_pass["e2e_latency_s"] - one_pass["e2e_latency_s"] == pytest.approx(
            4 * RESET_WALL_US * 1e-6 / CORES_PER_CHIP, rel=1e-3)
        # The headline moves by MORE than the switching energy: a longer wall
        # leaks longer, so static power follows the reset time too.
        assert (two_pass["energy_per_inference_mj"]
                - one_pass["energy_per_inference_mj"]) > four_more_cores_j

    def test_core_init_is_not_amortized_with_the_program_load(self):
        """A reset is paid every time a pass runs, so it belongs in the
        per-inference headline — the plane the wall was always charged on.
        Amortizing it with the program load made the same event per-load in
        energy and per-inference in time."""
        from mimarsinan.deployment_record.cost.absolute import price_absolute
        from mimarsinan.deployment_record.quantities.spec import (
            Quantities,
            QuantityValue,
        )

        quantities = Quantities({
            key: QuantityValue(float(value), "static") for key, value in {
                "synaptic_events": 1e6, "host_macs": 0, "cores_physical": 128,
                "latency_steps": 32, "timesteps": 4, "neurons_used": 1000,
                "segment_cores": 4,
            }.items()
        })
        pricing = price_absolute(quantities, get_platform_physics("loihi"))
        terms = {t.name: t for t in pricing.terms}
        assert "e_core_init" in terms["energy_per_inference_mj"].source
        assert "energy_programming_mj" not in terms or (
            "e_core_init" not in terms["energy_programming_mj"].source)
