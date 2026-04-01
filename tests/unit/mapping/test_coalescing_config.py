"""Tests for canonical coalescing flag resolution and normalization."""

import pytest

from mimarsinan.mapping.coalescing import (
    CANONICAL_KEY,
    CoalescingConfigError,
    coalescing_config_errors,
    normalize_coalescing_config,
    resolve_allow_coalescing,
)


class TestResolveAllowCoalescing:
    def test_empty_defaults_false(self):
        assert resolve_allow_coalescing({}, default=False) is False
        assert resolve_allow_coalescing({}, default=True) is True

    def test_canonical_true_false(self):
        assert resolve_allow_coalescing({CANONICAL_KEY: True}) is True
        assert resolve_allow_coalescing({CANONICAL_KEY: False}) is False

    def test_legacy_key_raises(self):
        with pytest.raises(CoalescingConfigError):
            resolve_allow_coalescing({"allow_core_coalescing": True})


class TestCoalescingConfigErrors:
    def test_no_legacy_empty(self):
        assert coalescing_config_errors({}) == []

    def test_legacy_reported(self):
        errs = coalescing_config_errors({"allow_core_coalescing": True})
        assert len(errs) == 1
        assert CANONICAL_KEY in errs[0]


class TestNormalizeCoalescingConfig:
    def test_sets_default_false(self):
        d: dict = {"cores": []}
        assert normalize_coalescing_config(d) is False
        assert d[CANONICAL_KEY] is False

    def test_canonical_preserved(self):
        d = {CANONICAL_KEY: True, "cores": []}
        assert normalize_coalescing_config(d) is True
        assert d[CANONICAL_KEY] is True

    def test_legacy_raises(self):
        with pytest.raises(CoalescingConfigError):
            normalize_coalescing_config({"allow_core_coalescing": True, "cores": []})
