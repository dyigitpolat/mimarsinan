"""R6 / Frontier E5 — deployment-faithfulness as standing infrastructure.

Locks the three E5 guards so the deployed-forward number stays the only number
of record and the four "the torch metric lied" failure modes fail LOUD:

(a) the torch<->sim parity gate is STANDING (default-on), not opt-in;
(b) every external-dependency integration boundary declares a guard, and the
    audit checklist fails loud if one does not / if a guard is missing;
(c) drift detection — a silent SANA-FE upgrade (or a one-sided pin bump) and a
    metric-protocol rewire fail loud.
"""

import pytest
import torch

from conftest import MockPipeline

from mimarsinan.chip_simulation.deployment_faithfulness import (
    DEPLOYED_METRIC_PROTOCOL,
    DEPLOYMENT_FAITHFULNESS_GATES,
    EXTERNAL_DEPENDENCY_BOUNDARIES,
    GUARD_KINDS,
    assert_sanafe_pin_consistent,
    boundary_for,
    bootstrap_pinned_sanafe_version,
    sanafe_supported_versions,
    standing_gates,
)


# --------------------------------------------------------------------------- #
# (a) Standing gates: default-on, not opt-in.
# --------------------------------------------------------------------------- #

class TestStandingGatesRegistry:
    def test_both_faithfulness_gates_are_standing(self):
        names = {g.name for g in standing_gates()}
        assert "torch_vs_deployed_sim_parity" in names
        assert "nf_scm_per_neuron_parity" in names

    def test_every_declared_gate_names_a_config_flag(self):
        for gate in DEPLOYMENT_FAITHFULNESS_GATES:
            assert gate.config_flag, gate.name


class TestTorchSimParityIsStanding:
    """E5(a): the torch<->deployed-sim parity check runs on a deployment run
    WITHOUT the config opting in — it is standing, not opt-in. We drive the real
    SoftCoreMappingStep gate method with a config that does not mention the flag
    and assert the executor build is reached (the gate did not early-return)."""

    class _StubTrainer:
        def __init__(self, batch):
            self._batch = batch

        def iter_validation_batches(self, n):
            yield self._batch, None

    def _make_step(self, spiking_mode):
        from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import (
            SoftCoreMappingStep,
        )

        p = MockPipeline()
        p.config["spiking_mode"] = spiking_mode
        p.config["firing_mode"] = "TTFS"
        p.config["spike_generation_mode"] = "TTFS"
        p.config["thresholding_mode"] = "<="
        p.config["simulation_steps"] = 4
        p.config["ttfs_cycle_schedule"] = "synchronized"
        step = SoftCoreMappingStep(p)
        step.trainer = self._StubTrainer(torch.rand(8, 8, dtype=torch.float64))
        return step

    def test_runs_by_default_without_opting_in(self, monkeypatch):
        # The gate re-imports its builders from simulation_factory each call, so
        # patch there (and let nf_scm_parity_enabled run for real: synchronized
        # ttfs_cycle is gated ON).
        from mimarsinan.pipelining.core import simulation_factory as sim_factory
        import mimarsinan.pipelining.core.nf_scm_parity as nf_scm_parity

        built = []
        monkeypatch.setattr(
            sim_factory, "build_spiking_hybrid_flow",
            lambda *a, **k: built.append(1) or object(),
        )
        monkeypatch.setattr(
            sim_factory, "build_identity_mapping_for_pipeline",
            lambda *a, **k: object(),
        )
        monkeypatch.setattr(
            nf_scm_parity, "assert_torch_vs_deployed_sim_parity_or_raise",
            lambda *a, **k: 1.0,
        )

        step = self._make_step("ttfs_cycle_based")
        assert "scm_torch_sim_parity_check" not in step.pipeline.config
        step._run_torch_sim_parity_check(model=object(), ir_graph=object())
        assert built == [1], "torch<->sim parity gate must be standing (default-on)"

    def test_can_be_explicitly_disabled(self, monkeypatch):
        from mimarsinan.pipelining.core import simulation_factory as sim_factory

        built = []
        monkeypatch.setattr(
            sim_factory, "build_spiking_hybrid_flow",
            lambda *a, **k: built.append(1) or object(),
        )
        step = self._make_step("ttfs_cycle_based")
        step.pipeline.config["scm_torch_sim_parity_check"] = False
        step._run_torch_sim_parity_check(model=object(), ir_graph=object())
        assert built == [], "explicit opt-out must skip the gate"


# --------------------------------------------------------------------------- #
# (b) External-dependency boundary audit checklist.
# --------------------------------------------------------------------------- #

class TestExternalDependencyBoundaryAudit:
    """E5(b): every declared boundary must carry at least one guard of a known
    kind, and its verify() (if any) must confirm the guard is actually present.
    A boundary added without a guard fails this loud."""

    def test_every_boundary_declares_a_known_guard(self):
        assert EXTERNAL_DEPENDENCY_BOUNDARIES, "the audit registry must be non-empty"
        for b in EXTERNAL_DEPENDENCY_BOUNDARIES:
            assert b.guards, (
                f"boundary {b.package!r} has NO guard — an unguarded external-dep "
                f"import can break a deployment number silently (the SANA-FE 2.2.x "
                f"SIGFPE lesson). Declare a guard or add one."
            )
            for kind in b.guards:
                assert kind in GUARD_KINDS, (
                    f"boundary {b.package!r} declares unknown guard kind {kind!r}"
                )

    def test_every_boundary_rationale_is_present(self):
        for b in EXTERNAL_DEPENDENCY_BOUNDARIES:
            assert b.rationale.strip(), b.package

    def test_boundary_verifiers_confirm_the_guard(self):
        for b in EXTERNAL_DEPENDENCY_BOUNDARIES:
            if b.verify is not None:
                b.verify()  # must not raise — the guard is present + live

    def test_sanafe_boundary_is_version_pinned_and_capability_gated(self):
        b = boundary_for("sanafe")
        assert b is not None
        assert "version_pin" in b.guards
        assert "capability_gate" in b.guards
        assert b.verify is not None

    def test_lava_boundary_is_capability_gated(self):
        b = boundary_for("lava")
        assert b is not None
        assert "capability_gate" in b.guards


# --------------------------------------------------------------------------- #
# (c) Drift detection.
# --------------------------------------------------------------------------- #

class TestSanafePinDriftDetection:
    """E5(c): the code guard's supported-version pin and the bootstrap script's
    `pip install sanafe==X` literal must agree. A one-sided bump (or a silent
    `pip install sanafe` upgrade past the pin) fails loud here in CI instead of
    SIGFPE-ing at deploy time."""

    def test_bootstrap_pin_matches_code_guard(self):
        version = assert_sanafe_pin_consistent()
        assert version in sanafe_supported_versions()

    def test_bootstrap_script_declares_a_pin(self):
        assert bootstrap_pinned_sanafe_version() is not None, (
            "bootstrap_sanafe.sh must pin sanafe==<version>"
        )

    def test_drift_fails_loud_on_mismatch(self, tmp_path):
        script = tmp_path / "bootstrap_sanafe.sh"
        script.write_text('pip install "sanafe==99.0.0"\n', encoding="utf-8")
        with pytest.raises(AssertionError, match="drift"):
            assert_sanafe_pin_consistent(str(script))

    def test_drift_fails_loud_when_pin_absent(self, tmp_path):
        script = tmp_path / "bootstrap_sanafe.sh"
        script.write_text("pip install sanafe  # unpinned!\n", encoding="utf-8")
        with pytest.raises(AssertionError, match="no .*pin"):
            assert_sanafe_pin_consistent(str(script))


class TestMetricProtocolDriftLock:
    """E5(c): the deployed metric protocol — which entry points constitute the
    deployed-forward number of record — is pinned. A silent rename/rewire that
    would change what "the deployed number" means fails this lock loud."""

    def test_protocol_entry_points_exist(self):
        from mimarsinan.pipelining.core import simulation_factory
        from mimarsinan.pipelining.core import nf_scm_parity
        from mimarsinan.pipelining.pipeline_steps.mapping import soft_core_mapping_step

        assert hasattr(simulation_factory, DEPLOYED_METRIC_PROTOCOL["metric_entrypoint"])
        assert hasattr(
            simulation_factory, DEPLOYED_METRIC_PROTOCOL["deployed_executor_builder"]
        )
        assert hasattr(nf_scm_parity, DEPLOYED_METRIC_PROTOCOL["parity_gate"])
        assert hasattr(
            soft_core_mapping_step, DEPLOYED_METRIC_PROTOCOL["metric_step"]
        )

    def test_metric_step_runs_metric_after_parity_gates(self):
        """The deployment step must call both parity gates BEFORE producing the
        metric (the gate cannot run after the number is already trusted)."""
        import inspect
        from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import (
            SoftCoreMappingStep,
        )

        src = inspect.getsource(SoftCoreMappingStep.process)
        gate_pos = src.index("_run_torch_sim_parity_check")
        nf_gate_pos = src.index("_run_nf_scm_parity_gate")
        metric_pos = src.index("run_scm_identity_metric")
        assert nf_gate_pos < metric_pos
        assert gate_pos < metric_pos
