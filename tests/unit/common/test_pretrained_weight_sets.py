"""The pretrained-weight-set registration contract (round-7).

A model builder does not have *a* pretrained source; it registers a SET of
weight sets — different tasks, datasets, input geometries, class counts and
applicable model configs. The framework never enumerates workloads: it only
reads the records a builder declared, injected into the config as
``pretrained_weight_sets``.

Everything below is a PURE function of that injected config. ``None`` means
"no builder was consulted in this view" (a raw document) and is never confused
with ``[]`` ("the builder was consulted and registers nothing").
"""

import pytest

from mimarsinan.common.pretrained import (
    PretrainedWeightSet,
    applicable_weight_sets,
    legal_preload_values,
    legal_weight_set_ids,
    preload_regime_error,
    preload_unavailable_reason,
    registered_weight_sets,
    select_weight_set,
    selected_source,
    weight_set_mismatch,
)
from mimarsinan.common.workload_profile import ModelWorkloadProfile


def _set(**kwargs) -> PretrainedWeightSet:
    base = dict(
        id="imagenet1k_v1",
        label="ImageNet-1K (V1)",
        task="image classification",
        dataset="ImageNet-1K",
        input_shape=(3, 224, 224),
        num_classes=1000,
        source="torchvision",
    )
    base.update(kwargs)
    return PretrainedWeightSet(**base)  # type: ignore[arg-type]


def _cfg(sets, **extra):
    cfg = {"pretrained_weight_sets": [s.as_dict() for s in sets]}
    cfg.update(extra)
    return cfg


class TestPretrainedWeightSetRecord:
    def test_declares_the_minimum_contract(self):
        ws = _set()
        assert (ws.id, ws.task, ws.dataset, ws.num_classes) == (
            "imagenet1k_v1", "image classification", "ImageNet-1K", 1000
        )
        assert ws.input_shape == (3, 224, 224)
        assert ws.source == "torchvision"

    def test_as_dict_is_json_safe_and_round_trips_the_facts(self):
        import json

        ws = _set(expected_accuracy=0.7613, license="BSD-3-Clause",
                  num_parameters=25557032, recipe="https://example/recipe",
                  preprocessing={"resize_to": 224}, adapts_input_shape=True,
                  adapts_num_classes=True, model_config_requires={"a": 1})
        record = ws.as_dict()
        assert json.loads(json.dumps(record)) == record
        assert record["id"] == "imagenet1k_v1"
        assert record["input_shape"] == [3, 224, 224]
        assert record["expected_accuracy"] == pytest.approx(0.7613)
        assert record["adapts_input_shape"] is True
        assert record["model_config_requires"] == {"a": 1}

    def test_every_declared_fact_survives_as_dict(self):
        """The panel reveals EVERY registered detail — none may be dropped."""
        from dataclasses import fields

        record = _set().as_dict()
        assert set(record) == {f.name for f in fields(PretrainedWeightSet)}

    @pytest.mark.parametrize("kwargs", [
        {"id": ""}, {"source": ""}, {"num_classes": 0}, {"input_shape": ()},
    ])
    def test_malformed_registration_fails_loud(self, kwargs):
        with pytest.raises(ValueError):
            _set(**kwargs)


class TestRegisteredWeightSets:
    def test_absent_key_means_no_builder_consulted(self):
        assert registered_weight_sets({}) is None
        assert legal_weight_set_ids({}) is None
        assert legal_preload_values({}) is None

    def test_empty_list_means_the_builder_registers_nothing(self):
        cfg = _cfg([])
        assert registered_weight_sets(cfg) == ()
        assert legal_weight_set_ids(cfg) == ()
        assert legal_preload_values(cfg) == (False,)

    def test_registered_sets_are_legal_by_id(self):
        cfg = _cfg([_set(), _set(id="imagenet1k_v2", label="V2")])
        assert legal_weight_set_ids(cfg) == ("imagenet1k_v1", "imagenet1k_v2")
        assert legal_preload_values(cfg) == (False, True)


class TestApplicability:
    def test_model_config_requirement_filters_the_set(self):
        gelu_only = _set(id="gelu", model_config_requires={"base_activation": "GELU"})
        cfg = _cfg([gelu_only], model_config={"base_activation": "ReLU"})
        assert "base_activation" in (weight_set_mismatch(gelu_only.as_dict(), cfg) or "")
        assert applicable_weight_sets(cfg) == ()
        assert legal_preload_values(cfg) == (False,)

        ok = _cfg([gelu_only], model_config={"base_activation": "GELU"})
        assert weight_set_mismatch(gelu_only.as_dict(), ok) is None
        assert legal_weight_set_ids(ok) == ("gelu",)

    def test_geometry_mismatch_filters_a_set_that_cannot_adapt(self):
        strict = _set(id="strict", adapts_input_shape=False, adapts_num_classes=True)
        cfg = _cfg([strict], input_shape=(1, 28, 28), num_classes=10)
        reason = weight_set_mismatch(strict.as_dict(), cfg)
        assert reason is not None and "(3, 224, 224)" in reason and "(1, 28, 28)" in reason
        assert legal_weight_set_ids(cfg) == ()

    def test_class_count_mismatch_filters_a_set_that_cannot_adapt(self):
        strict = _set(id="strict", adapts_input_shape=True, adapts_num_classes=False)
        cfg = _cfg([strict], input_shape=(1, 28, 28), num_classes=10)
        reason = weight_set_mismatch(strict.as_dict(), cfg)
        assert reason is not None and "1000 classes" in reason and "10" in reason
        assert legal_weight_set_ids(cfg) == ()

    def test_a_set_that_adapts_both_survives_a_contradicting_provider(self):
        """The torchvision builders project conv weights, interpolate embeddings
        and rebuild the head — a geometry difference is an ADAPTATION, not an
        incompatibility. Filtering on equality would delete the tier-1/2 corpus."""
        adaptive = _set(adapts_input_shape=True, adapts_num_classes=True)
        cfg = _cfg([adaptive], input_shape=(3, 32, 32), num_classes=10)
        assert weight_set_mismatch(adaptive.as_dict(), cfg) is None
        assert legal_weight_set_ids(cfg) == ("imagenet1k_v1",)

    def test_geometry_is_not_judged_when_the_provider_facts_are_absent(self):
        """A raw draft carries no runtime provider facts; the set stays legal and
        the RUN-time merged config is the authority."""
        strict = _set(id="strict")
        assert legal_weight_set_ids(_cfg([strict])) == ("strict",)


class TestSelection:
    def test_single_registered_set_is_the_selection(self):
        cfg = _cfg([_set()], preload_weights=True)
        assert select_weight_set(cfg)["id"] == "imagenet1k_v1"
        assert selected_source(cfg) == "torchvision"

    def test_first_registered_applicable_set_is_the_builder_default(self):
        cfg = _cfg([_set(), _set(id="v2", source="https://x/y.pt")], preload_weights=True)
        assert select_weight_set(cfg)["id"] == "imagenet1k_v1"

    def test_explicit_id_wins(self):
        cfg = _cfg([_set(), _set(id="v2", source="https://x/y.pt")],
                   preload_weights=True, pretrained_weight_set="v2")
        assert select_weight_set(cfg)["id"] == "v2"
        assert selected_source(cfg) == "https://x/y.pt"

    def test_explicit_inapplicable_id_selects_nothing(self):
        gelu = _set(id="gelu", model_config_requires={"base_activation": "GELU"})
        cfg = _cfg([gelu], preload_weights=True, pretrained_weight_set="gelu",
                   model_config={"base_activation": "ReLU"})
        assert select_weight_set(cfg) is None

    def test_no_selection_without_the_regime(self):
        assert select_weight_set(_cfg([_set()])) is None
        assert selected_source(_cfg([_set()])) is None

    def test_a_stray_set_id_alone_never_preloads(self):
        """The regime is the ``preload_weights`` FLAG; a set id is only the choice
        within it. A false flag beside a loading pipeline would be a lie."""
        cfg = _cfg([_set()], pretrained_weight_set="imagenet1k_v1")
        assert select_weight_set(cfg) is None
        assert selected_source(cfg) is None

    def test_selected_source_is_the_registered_source_not_an_override(self):
        """The explicit-``weight_source`` override is applied by the resolver, not
        here — this returns the REGISTERED set's source only."""
        cfg = _cfg([_set()], preload_weights=True, weight_source="/ckpt/best.pt")
        assert selected_source(cfg) == "torchvision"


class TestPreloadLegality:
    def test_explicit_source_locks_the_regime_on(self):
        """A document pinning a checkpoint declares the regime (the resolver flags
        it via ``source_declared``); the flag cannot honestly read false while the
        pipeline preloads."""
        assert legal_preload_values(_cfg([]), source_declared=True) == (True,)
        assert legal_preload_values({}, source_declared=True) == (True,)

    def test_no_registration_admits_only_off(self):
        assert legal_preload_values(_cfg([])) == (False,)

    def test_registration_admits_both(self):
        assert legal_preload_values(_cfg([_set()])) == (False, True)

    def test_unknown_registration_is_not_judged(self):
        assert legal_preload_values({"model_type": "whatever"}) is None


class TestUnavailabilityReason:
    def test_none_registered(self):
        reason = preload_unavailable_reason(_cfg([], model_type="lenet5"))
        assert reason is not None and "lenet5" in reason and "no pretrained" in reason

    def test_registered_but_none_applicable_names_every_mismatch(self):
        gelu = _set(id="gelu", model_config_requires={"base_activation": "GELU"})
        cfg = _cfg([gelu], model_type="torch_vit", model_config={"base_activation": "ReLU"})
        reason = preload_unavailable_reason(cfg)
        assert reason is not None
        assert "gelu" in reason and "base_activation" in reason

    def test_applicable_set_has_no_reason(self):
        assert preload_unavailable_reason(_cfg([_set()])) is None

    def test_unknown_registration_has_no_reason(self):
        assert preload_unavailable_reason({}) is None


class TestRegimeError:
    def test_message_names_only_reachable_remedies(self):
        """Round-5 removed the weight_source FIELD; the message must never send a
        configurator user to a control that does not exist."""
        message = str(preload_regime_error(_cfg([], model_type="lenet5", preload_weights=True)))
        assert "lenet5 registers no pretrained weights" in message
        assert "Turn the preload regime off" in message
        assert "config data" in message
        assert "Declare weight_source explicitly" not in message

    def test_declared_set_error_names_the_mismatch(self):
        gelu = _set(id="gelu", model_config_requires={"base_activation": "GELU"})
        cfg = _cfg([gelu], preload_weights=True, pretrained_weight_set="gelu",
                   model_config={"base_activation": "ReLU"})
        message = str(preload_regime_error(cfg))
        assert "pretrained_weight_set='gelu' cannot be used here" in message
        assert "base_activation" in message

    def test_unknown_set_id_error_lists_what_is_registered(self):
        cfg = _cfg([_set()], preload_weights=True, pretrained_weight_set="nope")
        message = str(preload_regime_error(cfg))
        assert "nope" in message and "imagenet1k_v1" in message


class TestProfileInjection:
    def test_profile_injects_its_sets_as_json_records(self):
        profile = ModelWorkloadProfile(pretrained_weight_sets=(_set(),))
        updates = profile.config_updates()
        assert updates["pretrained_weight_sets"] == [_set().as_dict()]

    def test_a_builder_that_registers_nothing_injects_an_empty_list(self):
        """`[]` is the CLAIM 'this builder registers no pretrained weights';
        it is what disables the regime, and it must not be confused with absence."""
        assert ModelWorkloadProfile().config_updates()["pretrained_weight_sets"] == []

    def test_scalar_pretrained_weight_source_is_retired(self):
        from dataclasses import fields

        names = {f.name for f in fields(ModelWorkloadProfile)}
        assert "pretrained_weight_source" not in names
        assert "pretrained_weight_sets" in names
