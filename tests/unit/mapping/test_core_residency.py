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
    Granularity,
    constrained_names,
    default_residency_policy,
    resolve_residency_policy,
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


class TestBankBackedCoresUseTheirBankScale:
    """A bank-backed core's effective parameter_scale lives on the BANK.

    export/chip_quantize reads it from there and re-expresses it as the core's threshold. Keying
    on the node's own field reports 1.0 for every sharer, merging cores that diverge on export --
    measured: threshold went 1 -> 2 distinct values across a real export once the banks carried
    distinct scales.
    """

    class _Bank:
        def __init__(self, scale):
            self.parameter_scale = scale

    def test_the_bank_scale_separates_cores_the_node_field_would_merge(self):
        a = _Soft(); a.weight_bank_id = 1
        b = _Soft(); b.weight_bank_id = 2
        banks = {1: self._Bank(torch.tensor(1.0)), 2: self._Bank(torch.tensor(4.0))}
        assert residency_key(a) == residency_key(b)                       # node fields agree
        assert residency_key(a, weight_banks=banks) != residency_key(b, weight_banks=banks)

    def test_cores_sharing_one_bank_stay_together(self):
        a = _Soft(); a.weight_bank_id = 7
        b = _Soft(); b.weight_bank_id = 7
        banks = {7: self._Bank(torch.tensor(3.0))}
        assert residency_key(a, weight_banks=banks) == residency_key(b, weight_banks=banks)

    def test_an_owned_core_is_unaffected(self):
        a = _Soft(parameter_scale=torch.tensor(2.0))
        a.weight_bank_id = None
        assert residency_key(a, weight_banks={}) == residency_key(a)


class TestGranularityIsDeclaredPerQuantity:
    """threshold and parameter_scale are alternative encodings, not one fixed pair.

    export/chip_quantize folds the weight scale INTO the firing threshold and resets
    parameter_scale to 1.0, because a spiking chip carries no scale register (nevresim has no
    parameter_scale at all -- verified). A value-domain target is the mirror image: no firing
    threshold, but a real quantization scale. So availability is a property of the DEPLOYMENT,
    and each quantity carries its own granularity rather than one being hard-coded.
    """

    def test_a_spiking_target_carries_a_threshold_and_no_scale(self):
        p = default_residency_policy(value_domain=False)
        assert p["threshold"] is Granularity.PER_CORE
        assert p["parameter_scale"] is Granularity.ABSENT

    def test_a_value_domain_target_is_the_mirror_image(self):
        p = default_residency_policy(value_domain=True)
        assert p["threshold"] is Granularity.ABSENT
        assert p["parameter_scale"] is Granularity.PER_CORE

    def test_only_per_core_quantities_constrain_residency(self):
        p = default_residency_policy(value_domain=True)
        names = constrained_names(p)
        assert "parameter_scale" in names
        assert "threshold" not in names, "an absent quantity cannot constrain packing"

    def test_an_absent_quantity_never_separates_two_cores(self):
        """A value-domain core has no threshold, so differing ones must not split the mapping."""
        p = default_residency_policy(value_domain=True)
        a, b = _Soft(threshold=1.0), _Soft(threshold=0.5)
        assert residency_key(a, constrained=constrained_names(p)) == residency_key(
            b, constrained=constrained_names(p)
        )

    def test_a_target_declares_per_neuron_hardware_by_name(self):
        p = resolve_residency_policy(
            {"core_value_granularity": {"threshold": "per_neuron"}}, value_domain=False
        )
        assert p["threshold"] is Granularity.PER_NEURON
        assert "threshold" not in constrained_names(p)

    def test_a_target_may_declare_both_present(self):
        """Nothing forces the encodings to be exclusive; a target carrying both says so."""
        p = resolve_residency_policy(
            {"core_value_granularity": {"parameter_scale": "per_core"}}, value_domain=False
        )
        assert p["threshold"] is Granularity.PER_CORE
        assert p["parameter_scale"] is Granularity.PER_CORE

    def test_an_unknown_quantity_is_refused(self):
        with pytest.raises(ValueError, match="not a core-level quantity"):
            resolve_residency_policy({"core_value_granularity": {"nonsense": "per_core"}})


class TestTheUngroupedFallbackHasOneDefinition:
    """Four copies of `-(id+1)` made "unknown means share with nothing" unchangeable in one place."""

    def test_all_call_sites_agree(self):
        from mimarsinan.mapping.packing.canonical import _read_residency_class
        from mimarsinan.mapping.platform.core_residency import (
            provenance_group_id,
            ungrouped_fallback_id,
        )

        class _Ungrouped:
            id = 7
            residency_class_id = None
            perceptron_index = None

        expected = ungrouped_fallback_id(7)
        assert _read_residency_class(_Ungrouped()) == expected
        assert provenance_group_id(None, fallback=ungrouped_fallback_id(7)) == expected

    def test_it_never_collides_with_a_real_group(self):
        from mimarsinan.mapping.platform.core_residency import ungrouped_fallback_id

        assert all(ungrouped_fallback_id(i) < 0 for i in range(64))
        assert len({ungrouped_fallback_id(i) for i in range(64)}) == 64


class TestTheGranularityDeclarationIsRegistered:
    """It enters through the config registry, not by editing DEFAULT_PLATFORM_CONSTRAINTS."""

    def test_the_key_is_a_registered_config_key(self):
        from mimarsinan.config_schema.defaults import CONFIG_KEYS_SET
        from mimarsinan.mapping.platform.core_residency import RESIDENCY_KEY

        assert RESIDENCY_KEY in CONFIG_KEYS_SET

    def test_it_round_trips_through_resolved_platform_constraints(self):
        from mimarsinan.mapping.platform.core_residency import (
            Granularity,
            RESIDENCY_KEY,
            resolve_residency_policy,
        )
        from mimarsinan.pipelining.core.platform_constraints_resolver import (
            build_platform_constraints_resolved,
        )

        resolved = build_platform_constraints_resolved(
            {RESIDENCY_KEY: {"threshold": "per_neuron"}}
        )
        policy = resolve_residency_policy(resolved, value_domain=False)
        assert policy["threshold"] is Granularity.PER_NEURON

    def test_an_undeclared_target_gets_the_domain_default(self):
        from mimarsinan.mapping.platform.core_residency import (
            Granularity,
            resolve_residency_policy,
        )
        from mimarsinan.pipelining.core.platform_constraints_resolver import (
            build_platform_constraints_resolved,
        )

        resolved = build_platform_constraints_resolved({})
        assert resolve_residency_policy(resolved, value_domain=True)["threshold"] is (
            Granularity.ABSENT
        )


class TestTheConceptHasOneName:
    """`threshold_group_id` named one of five values and, on a value-domain target, the one the
    grouping EXCLUDES. A half-true name is trusted where a wrong one is questioned."""

    def test_no_threshold_group_identifier_survives(self):
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parents[3] / "src" / "mimarsinan"
        offenders = []
        for path in root.rglob("*.py"):
            for n, line in enumerate(path.read_text().splitlines(), 1):
                if re.search(r"threshold.group|Threshold group", line):
                    if "Activation" in line or "activation-quantization" in line:
                        continue        # target_tq: activation-QUANTIZATION thresholds
                    offenders.append(f"{path.name}:{n}")
        assert not offenders, offenders

    def test_genuine_thresholds_were_not_renamed(self):
        """The rename targeted the compound token only; a firing threshold is a real concept,
        and so is the activation-quantization threshold group behind `target_tq`."""
        import dataclasses

        from mimarsinan.mapping.ir.types import NeuralCore

        fields = {f.name for f in dataclasses.fields(NeuralCore)}
        assert "threshold" in fields
        assert "residency" not in " ".join(fields)
