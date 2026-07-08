"""Normalization presets are provider-registered dataset facts, not framework constants."""

import pytest

import mimarsinan.data_handling.data_providers  # noqa: F401 — registers providers + presets
from mimarsinan.data_handling.data_providers.ecg_data_provider import (
    ECG_DataProvider,
)
from mimarsinan.data_handling.preprocessing import (
    NORMALIZATION_PRESETS,
    register_normalization_preset,
    resolve_preprocessing,
)


class TestProviderRegisteredPresets:
    def test_shipped_providers_register_their_canonical_normalizations(self):
        assert set(NORMALIZATION_PRESETS) >= {"imagenet", "cifar", "cifar10", "cifar100"}

    def test_registered_values_are_the_canonical_dataset_facts(self):
        mean, std = NORMALIZATION_PRESETS["cifar10"]
        assert mean == (0.4914, 0.4822, 0.4465)
        assert std == (0.2470, 0.2435, 0.2616)
        assert NORMALIZATION_PRESETS["cifar"] == NORMALIZATION_PRESETS["cifar10"]
        assert NORMALIZATION_PRESETS["imagenet"][0] == (0.485, 0.456, 0.406)

    def test_presets_resolve_through_preprocessing(self):
        spec = resolve_preprocessing({"normalize": "imagenet"})
        assert spec is not None
        assert spec.mean == (0.485, 0.456, 0.406)

    def test_unknown_preset_fails_loud_and_names_the_registry(self):
        with pytest.raises(ValueError, match="provider-registered"):
            resolve_preprocessing({"normalize": "no_such_dataset"})

    def test_registration_is_idempotent_and_alias_aware(self):
        register_normalization_preset("_t", (0.5,), (0.5,), aliases=("_t_alias",))
        register_normalization_preset("_t", (0.5,), (0.5,), aliases=("_t_alias",))
        assert NORMALIZATION_PRESETS["_t"] == NORMALIZATION_PRESETS["_t_alias"]
        NORMALIZATION_PRESETS.pop("_t")
        NORMALIZATION_PRESETS.pop("_t_alias")


class TestEcgBatchOverrideContract:
    def _provider(self, batch_size=None):
        provider = object.__new__(ECG_DataProvider)
        provider._batch_size_override = batch_size
        return provider

    def test_override_wins_over_the_full_test_batch(self, monkeypatch):
        provider = self._provider(batch_size=64)
        monkeypatch.setattr(
            ECG_DataProvider, "get_test_set_size", lambda self: 5000
        )
        assert provider.get_test_batch_size() == 64

    def test_default_stays_the_full_test_set_batch(self, monkeypatch):
        provider = self._provider()
        monkeypatch.setattr(
            ECG_DataProvider, "get_test_set_size", lambda self: 5000
        )
        assert provider.get_test_batch_size() == 5000
