"""core_semantics SSOT: the chip-domain axis, queried by intent."""

import pytest

from mimarsinan.chip_simulation.core_semantics import (
    CORE_SEMANTICS_MVM,
    CORE_SEMANTICS_SPIKING,
    CORE_SEMANTICS_VALUES,
    INERT_SPIKING_MODE,
    is_mvm_core_semantics,
    require_known_core_semantics,
    resolve_core_semantics,
)


class TestTaxonomy:
    def test_values(self):
        assert CORE_SEMANTICS_VALUES == frozenset({"spiking", "mvm"})
        assert CORE_SEMANTICS_SPIKING == "spiking"
        assert CORE_SEMANTICS_MVM == "mvm"

    def test_require_known_normalizes_absent_to_spiking(self):
        assert require_known_core_semantics(None) == "spiking"
        assert require_known_core_semantics("") == "spiking"
        assert require_known_core_semantics("mvm") == "mvm"

    def test_require_known_rejects_unknown(self):
        with pytest.raises(ValueError, match="core_semantics"):
            require_known_core_semantics("analog")

    def test_inert_spiking_mode_is_not_a_spiking_mode(self):
        from mimarsinan.chip_simulation.spiking_semantics import (
            ALL_SPIKING_MODES,
            is_cycle_based,
            is_lif,
            requires_ttfs_firing,
        )

        assert INERT_SPIKING_MODE not in ALL_SPIKING_MODES
        # The sentinel must never normalize into a live spiking family.
        assert not is_lif(INERT_SPIKING_MODE)
        assert not is_cycle_based(INERT_SPIKING_MODE)
        assert not requires_ttfs_firing(INERT_SPIKING_MODE)


class TestResolve:
    def test_default_is_spiking(self):
        assert resolve_core_semantics({}) == "spiking"

    def test_mvm_resolves(self):
        assert resolve_core_semantics({"core_semantics": "mvm"}) == "mvm"
        assert is_mvm_core_semantics(resolve_core_semantics({"core_semantics": "mvm"}))

    def test_unknown_fails_loud(self):
        with pytest.raises(ValueError):
            resolve_core_semantics({"core_semantics": "crossbar"})

    def test_is_mvm_on_values(self):
        assert is_mvm_core_semantics("mvm") is True
        assert is_mvm_core_semantics("spiking") is False
        assert is_mvm_core_semantics(None) is False
