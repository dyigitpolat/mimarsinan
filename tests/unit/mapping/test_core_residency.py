"""A hardware core stores one of each singleton value; a merge that disagrees must fail loudly.

Adopting from the first softcore and ignoring the rest -- what this code did -- leaves the later
core computing against a value that is not its own, with no signal. These pin the loud behaviour,
and pin the SSOT so a newly added per-core value cannot quietly become a sixth way to corrupt.
"""

import numpy as np
import pytest
import torch

from mimarsinan.mapping.platform.core_residency import (
    ALL_SINGLETON_NAMES,
    CORE_SINGLETON_PROPERTIES,
    CoreResidencyViolation,
    adopt_or_check,
    residency_key,
)


class _Soft:
    _next = 0

    def __init__(self, **kw):
        _Soft._next += 1
        self.id = _Soft._next
        self.threshold = kw.get("threshold", 1.0)
        self.activation_scale = kw.get("activation_scale", torch.tensor(1.0))
        self.parameter_scale = kw.get("parameter_scale", torch.tensor(1.0))
        self.input_activation_scale = kw.get("input_activation_scale", torch.tensor(1.0))
        self.boundary_grid = kw.get("boundary_grid", None)


class _Hard:
    def __init__(self):
        for p in CORE_SINGLETON_PROPERTIES:
            setattr(self, p.name, None)


class TestFirstSoftcoreSetsTheValues:
    def test_an_empty_core_adopts(self):
        hard, soft = _Hard(), _Soft(threshold=0.5)
        adopt_or_check(hard, soft)
        assert hard.threshold == 0.5

    def test_an_agreeing_softcore_is_accepted(self):
        hard = _Hard()
        adopt_or_check(hard, _Soft(threshold=0.5))
        adopt_or_check(hard, _Soft(threshold=0.5))
        assert hard.threshold == 0.5


class TestDisagreementRaisesInsteadOfCorrupting:
    @pytest.mark.parametrize(
        "field,a,b",
        [
            ("threshold", 1.0, 0.5),
            ("activation_scale", torch.tensor(1.0), torch.tensor(2.0)),
            ("parameter_scale", torch.tensor(1.0), torch.tensor(0.25)),
            ("input_activation_scale", torch.tensor(1.0), torch.tensor(4.0)),
        ],
    )
    def test_every_singleton_property_is_guarded(self, field, a, b):
        hard = _Hard()
        adopt_or_check(hard, _Soft(**{field: a}))
        with pytest.raises(CoreResidencyViolation, match=field):
            adopt_or_check(hard, _Soft(**{field: b}))

    def test_the_message_names_both_values(self):
        hard = _Hard()
        adopt_or_check(hard, _Soft(threshold=1.0))
        with pytest.raises(CoreResidencyViolation) as excinfo:
            adopt_or_check(hard, _Soft(threshold=0.5))
        text = str(excinfo.value)
        assert "0.5" in text and "1.0" in text

    def test_nan_never_compares_equal_to_itself(self):
        """A NaN threshold is not a value two cores can be said to share."""
        hard = _Hard()
        adopt_or_check(hard, _Soft(threshold=float("nan")))
        with pytest.raises(CoreResidencyViolation):
            adopt_or_check(hard, _Soft(threshold=float("nan")))


class TestTheKeyMatchesTheGuard:
    """Whatever the key merges, the guard must accept -- otherwise packing raises on a legal plan."""

    def test_equal_keys_are_always_mergeable(self):
        a, b = _Soft(threshold=0.5), _Soft(threshold=0.5)
        assert residency_key(a) == residency_key(b)
        hard = _Hard()
        adopt_or_check(hard, a)
        adopt_or_check(hard, b)

    def test_a_difference_in_any_property_separates_the_key(self):
        base = _Soft()
        for field, other in (
            ("threshold", 0.5),
            ("activation_scale", torch.tensor(3.0)),
            ("parameter_scale", torch.tensor(3.0)),
            ("input_activation_scale", torch.tensor(3.0)),
        ):
            assert residency_key(base) != residency_key(_Soft(**{field: other})), field

    def test_relaxing_a_property_merges_only_along_that_axis(self):
        relaxed = ALL_SINGLETON_NAMES - {"threshold"}
        a, b = _Soft(threshold=1.0), _Soft(threshold=0.5)
        assert residency_key(a, constrained=relaxed) == residency_key(b, constrained=relaxed)
        c = _Soft(threshold=1.0, parameter_scale=torch.tensor(9.0))
        assert residency_key(a, constrained=relaxed) != residency_key(c, constrained=relaxed)

    def test_the_key_is_hashable_and_stable(self):
        a = _Soft()
        assert hash(residency_key(a)) == hash(residency_key(a))
        assert len({residency_key(_Soft()) for _ in range(5)}) == 1


class TestScaleSpellingsCompareByValue:
    def test_tensor_and_array_of_the_same_value_agree(self):
        hard = _Hard()
        adopt_or_check(hard, _Soft(activation_scale=torch.tensor(2.0)))
        adopt_or_check(hard, _Soft(activation_scale=np.float64(2.0)))

    def test_differing_shapes_never_agree(self):
        hard = _Hard()
        adopt_or_check(hard, _Soft(activation_scale=torch.tensor([1.0, 1.0])))
        with pytest.raises(CoreResidencyViolation):
            adopt_or_check(hard, _Soft(activation_scale=torch.tensor(1.0)))


class TestTheSSOTIsTheOnlyPlaceToAddAProperty:
    def test_latency_is_deliberately_not_a_singleton(self):
        """schedule_split aggregates latency as a max; it is not an equality constraint."""
        assert "latency" not in ALL_SINGLETON_NAMES

    def test_hardware_bias_is_deliberately_not_a_singleton(self):
        """add_softcore merges hardware_bias per neuron range rather than adopting one."""
        assert "hardware_bias" not in ALL_SINGLETON_NAMES

    def test_every_declared_property_carries_its_own_comparator(self):
        for prop in CORE_SINGLETON_PROPERTIES:
            assert callable(prop.compare), prop.name
