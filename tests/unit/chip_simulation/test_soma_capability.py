"""Point-keyed backend capability: every backend refuses the new soma law BY NAME.

Nothing executes ``per_event`` firing or a saturating membrane yet, so at P1
EVERY backend — hcm and nevresim included — must refuse the point rather than
run a different physics and report it as the deployed number. The default point
must answer byte-identically to the mode-keyed matrix that preceded the axes.
"""

import pytest

from mimarsinan.chip_simulation.backend import BACKEND_REGISTRY
from mimarsinan.chip_simulation.firing_strategy import (
    FiringMode,
    FiringStrategy,
)
from mimarsinan.chip_simulation.soma_capability import (
    BackendSomaLawError,
    require_soma_law_supported,
    supports_soma_law,
)
from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW, SomaLaw
from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode
from mimarsinan.chip_simulation.spiking_semantics import (
    ALL_SPIKING_MODES,
    backend_capabilities,
    supports_spiking_mode,
)

_BACKENDS = ("hcm", "nevresim", "unified", "hybrid", "sanafe", "lava", "loihi",
             "training")

_PER_EVENT = SomaLaw.resolve({
    "spiking_family": "lif", "spiking_variant": "streamed",
    "firing_mode": "Novena", "firing_granularity": "per_event",
    "membrane_bits": 8,
})
_SATURATING = SomaLaw.resolve({
    "spiking_family": "lif", "spiking_variant": "streamed", "membrane_bits": 8,
})


class TestTheCapabilityMatrixDeclaresTheNewAxes:
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_no_backend_declares_per_event_or_saturating_at_p1(self, backend):
        caps = backend_capabilities(backend)
        assert caps.per_event_firing is False, backend
        assert caps.saturating_membrane is False, backend

    def test_the_existing_eight_positional_entries_stay_valid(self):
        for backend in _BACKENDS:
            caps = backend_capabilities(backend)
            assert isinstance(caps.lif, bool)


class TestRefusalByName:
    @pytest.mark.parametrize("backend", _BACKENDS)
    @pytest.mark.parametrize("law,axis", [
        (_PER_EVENT, "per_event"), (_SATURATING, "saturating_unsigned"),
    ])
    def test_every_backend_refuses_the_point_by_name(self, backend, law, axis):
        assert supports_soma_law(backend, law) is False
        with pytest.raises(BackendSomaLawError) as exc:
            require_soma_law_supported(law, backend=backend, context="ctx")
        message = str(exc.value)
        assert backend in message and axis in message
        assert "ctx" in message

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_the_default_point_is_never_refused(self, backend):
        assert supports_soma_law(backend, DEFAULT_SOMA_LAW) is True
        require_soma_law_supported(DEFAULT_SOMA_LAW, backend=backend, context="ctx")

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_an_absent_law_is_never_refused(self, backend):
        """The legacy mode-string surface carries no point and keeps its answer."""
        assert supports_soma_law(backend, None) is True
        require_soma_law_supported(None, backend=backend, context="ctx")


class TestThePolicyChainIsPointAware:
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_the_policy_refuses_the_point(self, backend):
        policy = policy_for_spiking_mode("lif", soma_law=_PER_EVENT)
        assert policy.supports_backend(backend) is False
        with pytest.raises(ValueError, match=backend):
            policy.require_backend_supported(backend=backend, context="ctx")

    @pytest.mark.parametrize("backend", _BACKENDS)
    @pytest.mark.parametrize("mode", sorted(ALL_SPIKING_MODES))
    def test_the_default_point_answers_exactly_as_the_mode_matrix(self, backend, mode):
        assert (
            policy_for_spiking_mode(mode, soma_law=DEFAULT_SOMA_LAW).supports_backend(
                backend)
            is supports_spiking_mode(backend, mode)
        )
        assert (
            policy_for_spiking_mode(mode).supports_backend(backend)
            is supports_spiking_mode(backend, mode)
        )

    def test_valid_backends_drops_every_backend_under_the_point(self):
        policy = policy_for_spiking_mode("lif", soma_law=_PER_EVENT)
        assert policy.valid_backends(_BACKENDS) == ()

    def test_the_policy_carries_the_law_it_was_given(self):
        assert policy_for_spiking_mode("lif", soma_law=_PER_EVENT).soma_law is (
            _PER_EVENT
        )
        assert policy_for_spiking_mode("lif").soma_law is None


class TestTheRawModeStringSurfaceIsUnchanged:
    @pytest.mark.parametrize("backend", _BACKENDS)
    @pytest.mark.parametrize("mode", sorted(ALL_SPIKING_MODES))
    def test_supports_spiking_mode_keeps_the_mode_keyed_answer(self, backend, mode):
        caps = backend_capabilities(backend)
        expected = {
            "lif": caps.lif, "ttfs": caps.ttfs,
            "ttfs_quantized": caps.ttfs_quantized,
            "ttfs_cycle_based": caps.ttfs_cycle_based,
        }[mode]
        assert supports_spiking_mode(backend, mode) is expected

    def test_backend_supports_accepts_a_raw_mode_string_unchanged(self):
        assert BACKEND_REGISTRY.get("loihi").supports("lif") is True
        assert BACKEND_REGISTRY.get("loihi").supports("ttfs") is False


class TestTheSecondFiringTableDelegates:
    """``FiringStrategy.capabilities`` folds into the ONE table; an unknown
    backend must NOT silently pass (the permissive default is gone)."""

    @pytest.mark.parametrize("backend", _BACKENDS)
    @pytest.mark.parametrize("mode", [FiringMode.DEFAULT, FiringMode.NOVENA])
    def test_known_backends_keep_their_answer(self, backend, mode):
        strategy = FiringStrategy(mode=mode, thresholding_mode="<=")
        caps = strategy.capabilities(backend)
        assert caps.supports_default is True
        assert caps.supports_novena is True
        strategy.require_backend(backend)

    def test_the_ttfs_column_reads_the_one_table(self):
        strategy = FiringStrategy(mode=FiringMode.DEFAULT, thresholding_mode="<=")
        assert strategy.capabilities("sanafe").supports_ttfs is True
        assert strategy.capabilities("loihi").supports_ttfs is False

    @pytest.mark.parametrize("mode", [FiringMode.DEFAULT, FiringMode.NOVENA])
    def test_an_undeclared_backend_refuses_instead_of_passing(self, mode):
        strategy = FiringStrategy(mode=mode, thresholding_mode="<=")
        caps = strategy.capabilities("odin")
        assert not (caps.supports_default or caps.supports_novena or caps.supports_ttfs)
        with pytest.raises(ValueError, match="odin"):
            strategy.require_backend("odin")
