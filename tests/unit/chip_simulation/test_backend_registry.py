"""Backend ABC + capability-validated registry (Vector V3).

These tests pin the V3 dispatch/precedence rules now made explicit:

- ``Backend.supports`` / ``require_supported`` consult the ``_BACKEND_CAPS``
  capability matrix (matrix-driven, not flag-driven).
- ``BackendRegistry.selected_step_specs`` validates every *enabled* backend
  UP-FRONT (at assembly) and returns its step specs in registry order, applying
  the per-backend extra gates (nevresim's synchronized-skip).
- An enabled backend×unsupported-mode raises an actionable error at assembly.
"""

import pytest

from mimarsinan.chip_simulation.backend import (
    BACKEND_REGISTRY,
    Backend,
    BackendRegistry,
    SimulationBackend,
)
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.pipeline_steps import (
    LoihiSimulationStep,
    SanafeSimulationStep,
    SimulationStep,
)


def _plan(**cfg) -> DeploymentPlan:
    base = {
        "configuration_mode": "user",
        "spiking_mode": "lif",
        "model_type": "mlp_mixer",
    }
    base.update(cfg)
    return DeploymentPlan.resolve(base)


# ── ABC contract ─────────────────────────────────────────────────────────────

class TestBackendSupports:
    def test_supports_reads_capability_matrix(self):
        nevresim = BACKEND_REGISTRY.get("nevresim")
        loihi = BACKEND_REGISTRY.get("loihi")
        sanafe = BACKEND_REGISTRY.get("sanafe")
        # nevresim/sanafe support every mode; loihi is LIF-only.
        for mode in ("lif", "ttfs", "ttfs_quantized", "ttfs_cycle_based"):
            assert nevresim.supports(_plan(spiking_mode=mode)), mode
            assert sanafe.supports(_plan(spiking_mode=mode)), mode
        assert loihi.supports(_plan(spiking_mode="lif"))
        for mode in ("ttfs", "ttfs_quantized", "ttfs_cycle_based"):
            assert not loihi.supports(_plan(spiking_mode=mode)), mode

    def test_supports_accepts_raw_mode_string(self):
        assert BACKEND_REGISTRY.get("loihi").supports("lif")
        assert not BACKEND_REGISTRY.get("loihi").supports("ttfs")

    def test_require_supported_passes_for_supported(self):
        BACKEND_REGISTRY.get("loihi").require_supported(
            _plan(spiking_mode="lif"), context="ctx"
        )

    def test_require_supported_raises_for_unsupported(self):
        with pytest.raises(ValueError, match="enable_loihi_simulation"):
            BACKEND_REGISTRY.get("loihi").require_supported(
                _plan(spiking_mode="ttfs"), context="ctx"
            )

    def test_sanafe_unsupported_uses_generic_matrix_error(self):
        # A backend without a custom error message falls back to the matrix error.
        custom = SimulationBackend(
            "loihi",
            step_name="X",
            step_class=SimulationStep,
            enabled_for=lambda p: True,
        )
        with pytest.raises(ValueError, match="does not support"):
            custom.require_supported(_plan(spiking_mode="ttfs"), context="ctx")


# ── registry selection / precedence ──────────────────────────────────────────

class TestRegistrySelection:
    def test_default_selects_nevresim_only(self):
        specs = BACKEND_REGISTRY.selected_step_specs(_plan())
        assert specs == [("Simulation", SimulationStep)]

    def test_nevresim_disabled_omits_step(self):
        specs = BACKEND_REGISTRY.selected_step_specs(
            _plan(enable_nevresim_simulation=False)
        )
        assert specs == []

    def test_all_enabled_registry_order(self):
        specs = BACKEND_REGISTRY.selected_step_specs(_plan(
            enable_loihi_simulation=True,
            enable_sanafe_simulation=True,
        ))
        assert specs == [
            ("Simulation", SimulationStep),
            ("Loihi Simulation", LoihiSimulationStep),
            ("SANA-FE Simulation", SanafeSimulationStep),
        ]

    def test_nevresim_synchronized_skip(self):
        # nevresim has no synchronized-window backend yet → skipped only there.
        specs = BACKEND_REGISTRY.selected_step_specs(_plan(
            spiking_mode="ttfs_cycle_based",
            ttfs_cycle_schedule="synchronized",
            enable_sanafe_simulation=True,
        ))
        assert ("Simulation", SimulationStep) not in specs
        assert ("SANA-FE Simulation", SanafeSimulationStep) in specs

    def test_nevresim_cascaded_kept(self):
        specs = BACKEND_REGISTRY.selected_step_specs(_plan(
            spiking_mode="ttfs_cycle_based",
            ttfs_cycle_schedule="cascaded",
        ))
        assert ("Simulation", SimulationStep) in specs

    def test_loihi_ttfs_raises_at_selection(self):
        with pytest.raises(ValueError, match="enable_loihi_simulation"):
            BACKEND_REGISTRY.selected_step_specs(_plan(
                spiking_mode="ttfs_quantized",
                enable_loihi_simulation=True,
            ))

    def test_disabled_unsupported_backend_does_not_raise(self):
        # loihi disabled + ttfs is fine — only enabled backends are validated.
        specs = BACKEND_REGISTRY.selected_step_specs(_plan(
            spiking_mode="ttfs_quantized",
            enable_loihi_simulation=False,
        ))
        assert ("Loihi Simulation", LoihiSimulationStep) not in specs


# ── registry data structure ──────────────────────────────────────────────────

class TestRegistryStructure:
    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="unknown backend"):
            BACKEND_REGISTRY.get("does_not_exist")

    def test_contains(self):
        assert "nevresim" in BACKEND_REGISTRY
        assert "sanafe" in BACKEND_REGISTRY
        assert "loihi" in BACKEND_REGISTRY
        assert "bogus" not in BACKEND_REGISTRY

    def test_simulation_backends_are_backends(self):
        for b in BACKEND_REGISTRY.simulation_backends():
            assert isinstance(b, Backend)
            assert isinstance(b, SimulationBackend)

    def test_abc_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            Backend()  # type: ignore[abstract]

    def test_empty_registry(self):
        reg = BackendRegistry([])
        assert reg.all() == ()
        assert reg.selected_step_specs(_plan()) == []


# ── byte-identity lock vs. the pre-V3 inline append block ─────────────────────

def _legacy_backend_specs(plan: DeploymentPlan) -> list[tuple[str, type]]:
    """Verbatim pre-V3 backend selection block (the strangler-fig baseline)."""
    specs: list[tuple[str, type]] = []
    if plan.enable_nevresim_simulation and not plan.is_synchronized_ttfs:
        specs.append(("Simulation", SimulationStep))
    if plan.enable_loihi_simulation:
        specs.append(("Loihi Simulation", LoihiSimulationStep))
    if plan.enable_sanafe_simulation:
        specs.append(("SANA-FE Simulation", SanafeSimulationStep))
    if plan.enable_loihi_simulation and plan.requires_ttfs_firing:
        raise ValueError(
            "enable_loihi_simulation is not supported for "
            f"spiking_mode={plan.spiking_mode!r}; "
            "Loihi/Lava only implements LIF dynamics."
        )
    return specs


class TestByteIdentityVsLegacy:
    """Registry output == verbatim pre-V3 block across the backend cross-product."""

    @pytest.mark.parametrize("spiking,schedule", [
        ("lif", None),
        ("ttfs", None),
        ("ttfs_quantized", None),
        ("ttfs_cycle_based", "cascaded"),
        ("ttfs_cycle_based", "synchronized"),
    ])
    @pytest.mark.parametrize("nevresim", [True, False])
    @pytest.mark.parametrize("loihi", [True, False])
    @pytest.mark.parametrize("sanafe", [True, False])
    def test_matches_legacy(self, spiking, schedule, nevresim, loihi, sanafe):
        plan = _plan(
            spiking_mode=spiking,
            ttfs_cycle_schedule=schedule,
            enable_nevresim_simulation=nevresim,
            enable_loihi_simulation=loihi,
            enable_sanafe_simulation=sanafe,
        )
        legacy_error = None
        legacy_specs = None
        try:
            legacy_specs = _legacy_backend_specs(plan)
        except ValueError as exc:
            legacy_error = str(exc)

        new_error = None
        new_specs = None
        try:
            new_specs = BACKEND_REGISTRY.selected_step_specs(plan)
        except ValueError as exc:
            new_error = str(exc)

        if legacy_error is not None:
            assert new_error == legacy_error
        else:
            assert new_error is None
            assert new_specs == legacy_specs
