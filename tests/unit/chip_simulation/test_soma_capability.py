"""Point-keyed backend capability: a backend answers on the POINT, by name.

At P1 nothing executed the new axes and every backend refused. P2 opens the
TORCH executors (hcm / hybrid / unified) — and only those: the legal set grows
exactly where an executor landed, so a backend without one still refuses
rather than run a different physics and report it as the deployed number. The
default point must answer byte-identically to the mode-keyed matrix that
preceded the axes.
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
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

# [ODIN P2] the torch executors implement the fold; everything else waits for
# its own phase (nevresim P3, exporter/RTL P4-P7) or refuses permanently.
_EXECUTING_BACKENDS = ("hcm", "unified", "hybrid")
_REFUSING_BACKENDS = ("nevresim", "sanafe", "lava", "loihi", "training")
_BACKENDS = _EXECUTING_BACKENDS + _REFUSING_BACKENDS
_REGISTERED_BACKENDS = tuple(
    backend.name for backend in BACKEND_REGISTRY.simulation_backends()
)

_PER_EVENT = SomaLaw.resolve({
    "spiking_family": "lif", "spiking_variant": "streamed",
    "firing_mode": "Novena", "firing_granularity": "per_event",
    "membrane_bits": 8,
})
_SATURATING = SomaLaw.resolve({
    "spiking_family": "lif", "spiking_variant": "streamed", "membrane_bits": 8,
})


class TestTheCapabilityMatrixDeclaresTheNewAxes:
    @pytest.mark.parametrize("backend", _REFUSING_BACKENDS)
    def test_a_backend_without_an_executor_declares_neither_axis(self, backend):
        caps = backend_capabilities(backend)
        assert caps.per_event_firing is False, backend
        assert caps.saturating_membrane is False, backend

    @pytest.mark.parametrize("backend", _EXECUTING_BACKENDS)
    def test_the_torch_executors_declare_both_axes(self, backend):
        caps = backend_capabilities(backend)
        assert caps.per_event_firing is True, backend
        assert caps.saturating_membrane is True, backend

    def test_the_existing_eight_positional_entries_stay_valid(self):
        for backend in _BACKENDS:
            caps = backend_capabilities(backend)
            assert isinstance(caps.lif, bool)


class TestRefusalByName:
    @pytest.mark.parametrize("backend", _REFUSING_BACKENDS)
    @pytest.mark.parametrize("law,axis", [
        (_PER_EVENT, "per_event"), (_SATURATING, "saturating_unsigned"),
    ])
    def test_every_executorless_backend_refuses_the_point_by_name(
        self, backend, law, axis
    ):
        assert supports_soma_law(backend, law) is False
        with pytest.raises(BackendSomaLawError) as exc:
            require_soma_law_supported(law, backend=backend, context="ctx")
        message = str(exc.value)
        assert backend in message and axis in message
        assert "ctx" in message

    @pytest.mark.parametrize("backend", _EXECUTING_BACKENDS)
    @pytest.mark.parametrize("law", [_PER_EVENT, _SATURATING])
    def test_the_torch_executors_admit_the_point(self, backend, law):
        assert supports_soma_law(backend, law) is True
        require_soma_law_supported(law, backend=backend, context="ctx")

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
    @pytest.mark.parametrize("backend", _REFUSING_BACKENDS)
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

    def test_valid_backends_keeps_exactly_the_backends_with_an_executor(self):
        policy = policy_for_spiking_mode("lif", soma_law=_PER_EVENT)
        assert policy.valid_backends(_BACKENDS) == _EXECUTING_BACKENDS

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


class TestTheStepLevelGuardNamesTheCauseOfTheRefusal:
    """§7 row 4 at the STEP-LEVEL guards: a refusal names the axis that refused.

    A backend may register its own message for the LEGACY mode refusal; using
    that message for a soma-point refusal would report a factually false cause
    (loihi does implement ``spiking_mode='lif'`` — what it cannot run is the
    declared point), which is the failure class these axes exist to prevent.
    """

    _POINTS = {
        "per_event": ({"firing_granularity": "per_event"}, "firing_granularity"),
        "saturating": ({"membrane_bits": 8}, "membrane_arithmetic"),
    }

    def _point_plan(self, backend, point):
        declaration, _axis = self._POINTS[point]
        return DeploymentPlan.resolve({
            "configuration_mode": "user", "model_type": "mlp_mixer",
            "spiking_family": "lif", "spiking_variant": "streamed",
            **{f"enable_{name}_simulation": (name == backend)
               for name in _REGISTERED_BACKENDS},
            **declaration,
        })

    @pytest.mark.parametrize("point", sorted(_POINTS))
    @pytest.mark.parametrize("backend", _REGISTERED_BACKENDS)
    def test_require_supported_refuses_the_point_by_axis(self, backend, point):
        plan = self._point_plan(backend, point)
        with pytest.raises(BackendSomaLawError) as exc:
            BACKEND_REGISTRY.get(backend).require_supported(plan, context="ctx")
        message = str(exc.value)
        assert self._POINTS[point][1] in message
        assert backend in message and "ctx" in message

    @pytest.mark.parametrize("point", sorted(_POINTS))
    @pytest.mark.parametrize("backend", _REGISTERED_BACKENDS)
    def test_selected_step_specs_refuses_the_enabled_backend_by_axis(
        self, backend, point
    ):
        plan = self._point_plan(backend, point)
        with pytest.raises(BackendSomaLawError) as exc:
            BACKEND_REGISTRY.selected_step_specs(plan)
        message = str(exc.value)
        assert self._POINTS[point][1] in message
        assert backend in message
        assert "only implements LIF dynamics" not in message


class TestTheLegacyModeRefusalKeepsItsExactMessage:
    """The registered per-backend message survives verbatim for its ACTUAL
    cause — an unsupported spiking mode at the default soma point."""

    def _mode_plan(self, mode):
        return DeploymentPlan.resolve({
            "configuration_mode": "user", "model_type": "mlp_mixer",
            "spiking_mode": mode, "enable_loihi_simulation": True,
        })

    @pytest.mark.parametrize("mode", ["ttfs", "ttfs_quantized", "ttfs_cycle_based"])
    def test_loihi_pins_its_historical_text_and_untyped_class(self, mode):
        with pytest.raises(ValueError) as exc:
            BACKEND_REGISTRY.get("loihi").require_supported(
                self._mode_plan(mode), context="Loihi Simulation"
            )
        assert type(exc.value) is ValueError
        assert str(exc.value) == (
            f"enable_loihi_simulation is not supported for spiking_mode={mode!r}; "
            "Loihi/Lava only implements LIF dynamics."
        )

    def test_loihi_admits_lif_at_the_default_point(self):
        BACKEND_REGISTRY.get("loihi").require_supported(
            self._mode_plan("lif"), context="Loihi Simulation"
        )
