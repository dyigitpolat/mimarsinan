"""Threshold grouping is a declared HARDWARE constraint, keyed on the threshold itself.

The policy and key live here; WIRING them into the layout finalizer, the spec adapters and
HardCore adoption is a separate migration unit (it changes packing and must be landed with the
config-registry entry and the parity evidence).

Some targets give every hardware core a single threshold register, so softcores sharing that
core must agree on their threshold. Others support independent thresholds per neuron column and
impose nothing. That is a property of the target, so it is DECLARED, and the grouping key is the
threshold value -- the constraint itself -- rather than a proxy for it.
"""

import pytest

from mimarsinan.mapping.platform.threshold_grouping import (
    ThresholdGroupingPolicy,
    resolve_threshold_grouping_policy,
    threshold_group_key,
)


class _Core:
    def __init__(self, threshold=1.0, perceptron_index=None):
        self.threshold = threshold
        self.perceptron_index = perceptron_index


class TestTheKeyIsTheConstraintNotAProxy:
    def test_same_threshold_groups_across_different_perceptrons(self):
        """The measured defect: 65 cores, one threshold value, split into 2 groups by provenance."""
        a = _Core(threshold=1.0, perceptron_index=0)
        b = _Core(threshold=1.0, perceptron_index=1)
        policy = ThresholdGroupingPolicy.SINGLE_THRESHOLD_PER_CORE
        assert threshold_group_key(a, policy=policy) == threshold_group_key(b, policy=policy)

    def test_different_thresholds_never_group(self):
        a = _Core(threshold=1.0)
        b = _Core(threshold=0.5)
        policy = ThresholdGroupingPolicy.SINGLE_THRESHOLD_PER_CORE
        assert threshold_group_key(a, policy=policy) != threshold_group_key(b, policy=policy)

    def test_missing_provenance_does_not_fragment(self):
        """Absent perceptron_index used to mean a UNIQUE group per core: 65 cores -> 65 groups."""
        cores = [_Core(threshold=1.0, perceptron_index=None) for _ in range(65)]
        policy = ThresholdGroupingPolicy.SINGLE_THRESHOLD_PER_CORE
        assert len({threshold_group_key(c, policy=policy) for c in cores}) == 1


class TestUnconstrainedIsANoOp:
    def test_every_core_shares_one_group_regardless_of_threshold(self):
        cores = [_Core(threshold=t) for t in (1.0, 0.5, 2.0, 7.25)]
        policy = ThresholdGroupingPolicy.UNCONSTRAINED
        assert len({threshold_group_key(c, policy=policy) for c in cores}) == 1

    def test_a_core_with_no_threshold_at_all_is_still_groupable(self):
        """Nothing is read when nothing is constrained."""
        class _Bare:
            pass
        assert threshold_group_key(_Bare(), policy=ThresholdGroupingPolicy.UNCONSTRAINED) is not None


class TestConstrainedPolicyFailsLoudRatherThanFragmenting:
    def test_absent_threshold_raises(self):
        class _Bare:
            pass
        with pytest.raises(ValueError, match="threshold"):
            threshold_group_key(_Bare(), policy=ThresholdGroupingPolicy.SINGLE_THRESHOLD_PER_CORE)

    def test_non_finite_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            threshold_group_key(
                _Core(threshold=float("nan")),
                policy=ThresholdGroupingPolicy.SINGLE_THRESHOLD_PER_CORE,
            )


class TestPolicyIsDeclaredByThePlatform:
    def test_default_constrains(self):
        """Conservative: a target does not silently lose a constraint it may rely on."""
        assert resolve_threshold_grouping_policy({}) is (
            ThresholdGroupingPolicy.SINGLE_THRESHOLD_PER_CORE
        )

    def test_a_target_supporting_per_neuron_thresholds_declares_it(self):
        policy = resolve_threshold_grouping_policy({"single_threshold_per_core": False})
        assert policy is ThresholdGroupingPolicy.UNCONSTRAINED


class TestGroupingCanOnlyEverMergeWhatTheHardwareAllows:
    def test_the_key_never_merges_distinct_thresholds_under_the_constraint(self):
        """The safety argument: the key IS the hardware predicate, so a merge is always legal."""
        policy = ThresholdGroupingPolicy.SINGLE_THRESHOLD_PER_CORE
        cores = [_Core(threshold=t) for t in (1.0, 1.0, 0.5, 0.5, 0.25)]
        buckets: dict = {}
        for c in cores:
            buckets.setdefault(threshold_group_key(c, policy=policy), []).append(c)
        for members in buckets.values():
            assert len({m.threshold for m in members}) == 1
