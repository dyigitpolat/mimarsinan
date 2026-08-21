"""[ODIN4] The feasibility gates: typed, keyed, and each naming its remediation.

Plan Sec.5.2 / Sec.7 row 12. Every gate is exercised on a CONSTRUCTED violation —
a fixture built to break exactly that predicate — and asserted to raise its own key,
so a gate cannot be satisfied by another gate firing first.
"""

import numpy as np
import pytest

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.export.odin.feasibility import (
    EMISSION_CEILING,
    KEY_EMISSION_BOUND,
    KEY_FAN_IN,
    KEY_THETA_CEILING,
    KEY_WEIGHT_MAGNITUDE_RANGE,
    OdinFeasibilityError,
    check_fan_in,
    check_theta_ceiling,
    check_weight_magnitudes,
    emission_bound_of,
    propagate_emission_bounds,
)
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping


# --------------------------------------------------------------------------
# E1. theta ceiling: 1 <= theta <= 2**membrane_bits - 1
# --------------------------------------------------------------------------

class TestTheThetaCeilingGate:
    def test_the_legal_band_passes_at_both_ends(self):
        check_theta_ceiling(1, membrane_bits=8, core_index=0)
        check_theta_ceiling(255, membrane_bits=8, core_index=0)

    def test_theta_above_the_ceiling_is_refused_with_the_value_and_the_ceiling(self):
        with pytest.raises(OdinFeasibilityError) as excinfo:
            check_theta_ceiling(256, membrane_bits=8, core_index=3)
        assert excinfo.value.key == KEY_THETA_CEILING
        assert "256" in str(excinfo.value)
        assert "255" in str(excinfo.value)
        assert "core 3" in str(excinfo.value)

    def test_the_refusal_names_the_adaptation_ladder_remediation(self):
        with pytest.raises(OdinFeasibilityError, match="adaptation"):
            check_theta_ceiling(400, membrane_bits=8, core_index=0)

    def test_theta_below_one_is_refused_because_the_compare_would_always_fire(self):
        with pytest.raises(OdinFeasibilityError) as excinfo:
            check_theta_ceiling(0, membrane_bits=8, core_index=0)
        assert excinfo.value.key == KEY_THETA_CEILING

    def test_a_non_integral_theta_is_refused(self):
        with pytest.raises(OdinFeasibilityError) as excinfo:
            check_theta_ceiling(2.5, membrane_bits=8, core_index=0)
        assert excinfo.value.key == KEY_THETA_CEILING

    def test_an_undeclared_membrane_width_is_refused_not_waved_through(self):
        with pytest.raises(OdinFeasibilityError) as excinfo:
            check_theta_ceiling(4, membrane_bits=0, core_index=0)
        assert excinfo.value.key == KEY_THETA_CEILING

    def test_a_wider_membrane_raises_the_ceiling(self):
        check_theta_ceiling(4000, membrane_bits=16, core_index=0)
        with pytest.raises(OdinFeasibilityError):
            check_theta_ceiling(1 << 16, membrane_bits=16, core_index=0)


# --------------------------------------------------------------------------
# E2. q_min refusal under per-axon signs (the range is SYMMETRIC).
# --------------------------------------------------------------------------

class TestTheSymmetricMagnitudeGate:
    def test_the_symmetric_range_passes(self):
        check_weight_magnitudes(
            np.array([[7, -7], [0, 3]]),
            weight_bits=4, weight_sign_granularity="per_axon", core_index=0,
        )

    def test_q_min_is_refused_by_value_under_per_axon_signs(self):
        with pytest.raises(OdinFeasibilityError) as excinfo:
            check_weight_magnitudes(
                np.array([[-8, 1]]),
                weight_bits=4, weight_sign_granularity="per_axon", core_index=2,
            )
        assert excinfo.value.key == KEY_WEIGHT_MAGNITUDE_RANGE
        assert "-8" in str(excinfo.value)
        assert "core 2" in str(excinfo.value)

    def test_the_refusal_says_the_magnitude_field_is_unsigned(self):
        with pytest.raises(OdinFeasibilityError, match="magnitude"):
            check_weight_magnitudes(
                np.array([[-8]]),
                weight_bits=4, weight_sign_granularity="per_axon", core_index=0,
            )

    def test_per_synapse_signs_keep_the_asymmetric_framework_range(self):
        # A per-synapse substrate stores the sign in the cell, so q_min is
        # representable and must NOT be refused.
        check_weight_magnitudes(
            np.array([[-8, 7]]),
            weight_bits=4, weight_sign_granularity="per_synapse", core_index=0,
        )

    def test_a_magnitude_above_q_max_is_refused_under_either_granularity(self):
        for granularity in ("per_axon", "per_synapse"):
            with pytest.raises(OdinFeasibilityError) as excinfo:
                check_weight_magnitudes(
                    np.array([[8]]), weight_bits=4,
                    weight_sign_granularity=granularity, core_index=0,
                )
            assert excinfo.value.key == KEY_WEIGHT_MAGNITUDE_RANGE

    def test_an_unknown_granularity_is_refused_rather_than_defaulted(self):
        with pytest.raises(OdinFeasibilityError, match="per_axon"):
            check_weight_magnitudes(
                np.array([[1]]), weight_bits=4,
                weight_sign_granularity="per_core", core_index=0,
            )


# --------------------------------------------------------------------------
# E3. fan-in over the effective limit (should already have fired upstream).
# --------------------------------------------------------------------------

class TestTheFanInGate:
    def test_the_effective_limit_passes_at_the_boundary(self):
        check_fan_in(127, effective_max_axons=127, core_index=0)

    def test_one_axon_over_the_effective_limit_is_refused(self):
        with pytest.raises(OdinFeasibilityError) as excinfo:
            check_fan_in(128, effective_max_axons=127, core_index=5)
        assert excinfo.value.key == KEY_FAN_IN
        assert "128" in str(excinfo.value)
        assert "127" in str(excinfo.value)
        assert "core 5" in str(excinfo.value)

    def test_the_refusal_says_the_mapper_should_have_refused_first(self):
        with pytest.raises(OdinFeasibilityError, match="mapper"):
            check_fan_in(200, effective_max_axons=127, core_index=0)


# --------------------------------------------------------------------------
# E4. The propagated emission bound (plan Sec.2.2, revised per J3-5).
# --------------------------------------------------------------------------

def _core(matrix, *, threshold, sources):
    core = HardCore(
        axons_per_core=matrix.shape[0],
        neurons_per_core=matrix.shape[1],
        has_bias_capability=False,
    )
    core.core_matrix = np.asarray(matrix, dtype=np.float64)
    core.axon_sources = list(sources)
    core.threshold = threshold
    core.latency = 0
    core.available_axons = 0
    core.available_neurons = 0
    return core


def _mapping(cores, outputs):
    mapping = HardCoreMapping(chip_cores=[])
    mapping.cores = list(cores)
    mapping.output_sources = list(outputs)
    return mapping


def _in(index):
    return SpikeSource(-2, index, is_input=True, is_off=False)


def _from(core, neuron):
    return SpikeSource(core, neuron, is_input=False, is_off=False)


class TestTheEmissionBoundFormula:
    """e_out(n) = ceil((sum_a max(w,0) e_in(a) + (theta - 1)) / theta)."""

    @pytest.mark.parametrize(
        "positives, theta, expected",
        [
            # one unit-weight input at theta=1: every event crosses.
            (((1, 1),), 1, 1),
            # sum 3, theta 2, entering membrane theta-1=1 -> ceil(4/2) = 2.
            (((3, 1),), 2, 2),
            # sum 6 over two axons with multiplicity, theta 3 -> ceil(8/3) = 3.
            (((2, 2), (2, 1)), 3, 3),
            # nothing positive still admits the entering membrane: ceil(4/5)=1.
            ((), 5, 1),
        ],
    )
    def test_hand_computed_single_neuron_cases(self, positives, theta, expected):
        assert emission_bound_of(positives, theta=theta) == expected

    def test_a_negative_weight_never_raises_the_bound(self):
        assert emission_bound_of(((-9, 4), (2, 1)), theta=2) == emission_bound_of(
            ((2, 1),), theta=2
        )


class TestTheBoundPropagatesTopologically:
    def test_entries_are_seeded_at_one(self):
        # theta=1, w=1: a single entry event yields exactly one spike.
        core = _core(np.array([[1.0]]), threshold=1.0, sources=[_in(0)])
        bounds = propagate_emission_bounds(
            _mapping([core], [_from(0, 0)]), ceiling=EMISSION_CEILING
        )
        assert bounds[(0, 0)] == 1

    def test_a_two_layer_chain_multiplies_the_bound(self):
        # Layer 0: 4 unit inputs, theta=1 -> ceil(4/1) = 4 per neuron.
        first = _core(
            np.array([[1.0], [1.0], [1.0], [1.0]]),
            threshold=1.0,
            sources=[_in(i) for i in range(4)],
        )
        # Layer 1: one axon carrying 4 events, w=1, theta=1 -> 4.
        second = _core(
            np.array([[1.0]]), threshold=1.0, sources=[_from(0, 0)]
        )
        bounds = propagate_emission_bounds(
            _mapping([first, second], [_from(1, 0)]), ceiling=EMISSION_CEILING
        )
        assert bounds[(0, 0)] == 4
        assert bounds[(1, 0)] == 4

    def test_a_higher_threshold_damps_the_bound(self):
        first = _core(
            np.array([[1.0], [1.0], [1.0], [1.0]]),
            threshold=1.0,
            sources=[_in(i) for i in range(4)],
        )
        # 4 incoming events at w=1 with theta=4 -> ceil((4 + 3)/4) = 2.
        second = _core(
            np.array([[1.0]]), threshold=4.0, sources=[_from(0, 0)]
        )
        bounds = propagate_emission_bounds(
            _mapping([first, second], [_from(1, 0)]), ceiling=EMISSION_CEILING
        )
        assert bounds[(1, 0)] == 2

    def test_an_always_on_bias_row_is_seeded_at_one(self):
        core = _core(
            np.array([[1.0], [3.0]]),
            threshold=1.0,
            sources=[_in(0), SpikeSource(-3, 0, is_always_on=True)],
        )
        bounds = propagate_emission_bounds(
            _mapping([core], [_from(0, 0)]), ceiling=EMISSION_CEILING
        )
        assert bounds[(0, 0)] == 4

    def test_an_off_axon_contributes_nothing(self):
        core = _core(
            np.array([[1.0], [9.0]]),
            threshold=1.0,
            sources=[_in(0), SpikeSource(-1, 0, is_off=True)],
        )
        bounds = propagate_emission_bounds(
            _mapping([core], [_from(0, 0)]), ceiling=EMISSION_CEILING
        )
        assert bounds[(0, 0)] == 1


class TestTheBoundIsARefusal:
    def test_the_ceiling_is_the_count_currency_ceiling(self):
        assert EMISSION_CEILING == 127

    def test_a_bound_over_the_ceiling_refuses_with_the_offending_neuron(self):
        # 200 unit inputs at theta=1 -> 200 > 127.
        core = _core(
            np.ones((200, 1)),
            threshold=1.0,
            sources=[_in(i) for i in range(200)],
        )
        with pytest.raises(OdinFeasibilityError) as excinfo:
            propagate_emission_bounds(
                _mapping([core], [_from(0, 0)]), ceiling=EMISSION_CEILING
            )
        assert excinfo.value.key == KEY_EMISSION_BOUND
        assert "200" in str(excinfo.value)
        assert "127" in str(excinfo.value)
        assert "neuron 0" in str(excinfo.value)

    def test_the_refusal_names_scale_and_threshold_adaptation_never_a_clamp(self):
        core = _core(
            np.ones((200, 1)), threshold=1.0, sources=[_in(i) for i in range(200)]
        )
        with pytest.raises(OdinFeasibilityError, match="threshold") as excinfo:
            propagate_emission_bounds(
                _mapping([core], [_from(0, 0)]), ceiling=EMISSION_CEILING
            )
        assert "clamp" in str(excinfo.value)

    def test_a_lower_ceiling_refuses_a_mapping_the_default_ceiling_admits(self):
        core = _core(
            np.ones((10, 1)), threshold=1.0, sources=[_in(i) for i in range(10)]
        )
        propagate_emission_bounds(
            _mapping([core], [_from(0, 0)]), ceiling=EMISSION_CEILING
        )
        with pytest.raises(OdinFeasibilityError):
            propagate_emission_bounds(
                _mapping([core], [_from(0, 0)]), ceiling=4
            )

    def test_a_cycle_in_the_segment_is_refused_rather_than_looping(self):
        a = _core(np.array([[1.0]]), threshold=1.0, sources=[_from(1, 0)])
        b = _core(np.array([[1.0]]), threshold=1.0, sources=[_from(0, 0)])
        with pytest.raises(OdinFeasibilityError, match="cycle"):
            propagate_emission_bounds(
                _mapping([a, b], [_from(1, 0)]), ceiling=EMISSION_CEILING
            )

    def test_a_core_without_a_threshold_is_refused_not_defaulted(self):
        core = _core(np.array([[1.0]]), threshold=None, sources=[_in(0)])
        with pytest.raises(OdinFeasibilityError, match="threshold"):
            propagate_emission_bounds(
                _mapping([core], [_from(0, 0)]), ceiling=EMISSION_CEILING
            )
