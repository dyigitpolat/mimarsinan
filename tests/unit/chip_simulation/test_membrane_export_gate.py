"""[C2/WS-V] Deployed membrane-readout honesty gate over backend capability.

The membrane-augmented logits decode (``Q_T = theta*c_T + m_T``) is a
legitimate DEPLOYED read only when every chip backend enabled for the run can
export end-of-window membranes: nevresim via its ``NEVRESIM_EXPORT_MEMBRANE``
read port, SANA-FE via the plugin soma ``get_potential``; Lava/Loihi reads
spike parity only. The gate must fail toward counts.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mimarsinan.chip_simulation.backend import BACKEND_REGISTRY
from mimarsinan.chip_simulation.membrane_export import (
    backend_exports_final_membrane,
    deployed_membrane_readout_enabled,
    enabled_backend_names,
    half_step_charge_from_config,
)


def _plan(*, nevresim=True, loihi=False, sanafe=False, spiking_mode="lif"):
    return SimpleNamespace(
        enable_nevresim_simulation=nevresim,
        enable_loihi_simulation=loihi,
        enable_sanafe_simulation=sanafe,
        spiking_mode=spiking_mode,
    )


ARMED = {"lif_membrane_readout": True}


class TestBackendCapabilityDeclarations:
    """Capability is declared on the backend registry (the backend SSOT)."""

    def test_nevresim_exports_final_membrane(self):
        assert backend_exports_final_membrane("nevresim") is True
        assert BACKEND_REGISTRY.get("nevresim").exports_final_membrane is True

    def test_sanafe_exports_final_membrane(self):
        assert backend_exports_final_membrane("sanafe") is True

    def test_loihi_cannot_export_final_membrane(self):
        assert backend_exports_final_membrane("loihi") is False

    def test_unknown_backend_raises(self):
        with pytest.raises(KeyError):
            backend_exports_final_membrane("tpu")


class TestEnabledBackendNames:
    def test_default_run_enables_nevresim_only(self):
        assert enabled_backend_names(_plan()) == ("nevresim",)

    def test_all_enabled(self):
        assert enabled_backend_names(_plan(loihi=True, sanafe=True)) == (
            "nevresim", "loihi", "sanafe",
        )

    def test_none_enabled(self):
        assert enabled_backend_names(_plan(nevresim=False)) == ()


class TestDeployedMembraneReadoutGate:
    def test_armed_lif_nevresim_only_is_enabled(self):
        assert deployed_membrane_readout_enabled(ARMED, _plan()) is True

    def test_armed_with_sanafe_stays_enabled(self):
        assert deployed_membrane_readout_enabled(ARMED, _plan(sanafe=True)) is True

    def test_loihi_enabled_keeps_the_decode(self):
        """Loihi is a parity-currency backend: it compares per-neuron COUNTS
        (untouched by the logits decode) and never produces a deployed
        accuracy read — its membrane-incapability cannot make the accuracy
        decode dishonest. The gate quantifies over accuracy-producing
        backends only."""
        assert deployed_membrane_readout_enabled(ARMED, _plan(loihi=True)) is True

    def test_incapable_accuracy_backend_fails_toward_counts(self):
        """A hypothetical accuracy-producing backend without membrane export
        MUST veto the decode — honesty over accuracy."""
        from mimarsinan.chip_simulation import membrane_export

        class _Fake:
            name = "fake"
            exports_final_membrane = False
            decodes_accuracy = True

        assert membrane_export._backend_vetoes_membrane_decode(_Fake()) is True

    def test_unarmed_knob_is_counts(self):
        assert deployed_membrane_readout_enabled({}, _plan()) is False
        assert deployed_membrane_readout_enabled(
            {"lif_membrane_readout": False}, _plan(),
        ) is False

    def test_non_lif_mode_is_counts(self):
        for mode in ("ttfs", "ttfs_quantized", "ttfs_cycle_based"):
            assert deployed_membrane_readout_enabled(
                ARMED, _plan(spiking_mode=mode),
            ) is False

    def test_no_enabled_backends_keeps_decode(self):
        """A torch-only run has no incapable backend; the decode stays
        chip-realizable through the in-repo nevresim read port."""
        assert deployed_membrane_readout_enabled(
            ARMED, _plan(nevresim=False),
        ) is True


class TestHalfStepCharge:
    """The [C1] half-step wire-bias fold contributes exactly theta/2 to the
    terminal charge; the membrane decode removes that charge iff it was baked."""

    def test_armed_half_step_is_half(self):
        assert half_step_charge_from_config({"lif_half_step_bias": True}) == 0.5

    def test_unarmed_half_step_is_zero(self):
        assert half_step_charge_from_config({}) == 0.0
        assert half_step_charge_from_config({"lif_half_step_bias": False}) == 0.0
