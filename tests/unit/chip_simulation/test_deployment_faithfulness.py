"""R6 / Frontier E5 — deployment-faithfulness as standing infrastructure.

Locks the three E5 guards so the deployed-forward number stays the only number
of record and the four "the torch metric lied" failure modes fail LOUD:

(a) the torch<->sim parity gate is STANDING (default-on), not opt-in;
(b) every external-dependency integration boundary declares a guard, and the
    audit checklist fails loud if one does not / if a guard is missing;
(c) drift detection — a silent SANA-FE upgrade (or a one-sided pin bump) and a
    metric-protocol rewire fail loud.
"""

from types import SimpleNamespace

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
    manifest_pinned_sanafe_version,
    sanafe_supported_versions,
    standing_gates,
)


# --------------------------------------------------------------------------- #
# (a) Standing gates: default-on, not opt-in.
# --------------------------------------------------------------------------- #

class TestStandingGatesRegistry:
    def test_both_faithfulness_rows_are_standing(self):
        names = {g.name for g in standing_gates()}
        assert "readout_decision_drift" in names
        assert "nf_scm_per_neuron_parity" in names

    def test_only_the_per_neuron_row_can_fail_a_run(self):
        """The drift row is a standing REPORT: it runs on every deployment run
        and can never be the reason one dies (t0_55 died on it at 0.7969 while
        the per-neuron gate was green at atol=0)."""
        fatal = {g.name for g in DEPLOYMENT_FAITHFULNESS_GATES if g.fatal}
        assert fatal == {"nf_scm_per_neuron_parity", "readout_decision_drift"}

    def test_every_declared_gate_names_a_config_flag(self):
        for gate in DEPLOYMENT_FAITHFULNESS_GATES:
            assert gate.config_flag, gate.name


class TestReadoutDriftReportIsStanding:
    """E5(a): the readout-decision-drift report runs on a deployment run WITHOUT
    the config opting in — it is standing, not opt-in. We drive the real
    SoftCoreMappingStep method with a config that does not mention the flag and
    assert the executor build is reached (it did not early-return)."""

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
        # Patch the builders where the step looks them up (its own module
        # globals); nf_scm_parity_enabled runs for real: synchronized
        # ttfs_cycle is gated ON.
        import mimarsinan.pipelining.core.nf_scm_parity as nf_scm_parity
        import mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step as scm_mod

        built = []
        monkeypatch.setattr(
            scm_mod, "build_spiking_hybrid_flow",
            lambda *a, **k: built.append(1) or object(),
        )
        monkeypatch.setattr(
            scm_mod, "build_identity_mapping_for_pipeline",
            lambda *a, **k: object(),
        )
        monkeypatch.setattr(
            nf_scm_parity, "measure_readout_decision_drift",
            lambda *a, **k: 1.0,
        )

        step = self._make_step("ttfs_cycle_based")
        assert "scm_torch_sim_parity_check" not in step.pipeline.config
        model = SimpleNamespace(get_perceptrons=lambda: [])
        step._run_readout_decision_drift_diagnostic(model=model, ir_graph=object())
        assert built == [1], "the readout-drift report must be standing (default-on)"

    def test_sole_guard_below_floor_fails_the_step(self, monkeypatch):
        """Where no exactness gate arms, the floor is the hop's only fatal
        guard (A1): agreement below it must raise, naming the floor."""
        import mimarsinan.pipelining.core.nf_scm_parity as nf_scm_parity
        import mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step as scm_mod

        monkeypatch.setattr(
            scm_mod, "build_spiking_hybrid_flow", lambda *a, **k: object())
        monkeypatch.setattr(
            scm_mod, "build_identity_mapping_for_pipeline",
            lambda *a, **k: object())
        monkeypatch.setattr(
            nf_scm_parity, "measure_readout_decision_drift",
            lambda *a, **k: 0.5)

        step = self._make_step("ttfs_cycle_based")
        model = SimpleNamespace(get_perceptrons=lambda: [])
        with pytest.raises(RuntimeError, match="only fatal guard"):
            step._run_readout_decision_drift_diagnostic(
                model=model, ir_graph=object())

    def test_sole_guard_above_floor_passes(self, monkeypatch):
        import mimarsinan.pipelining.core.nf_scm_parity as nf_scm_parity
        import mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step as scm_mod

        monkeypatch.setattr(
            scm_mod, "build_spiking_hybrid_flow", lambda *a, **k: object())
        monkeypatch.setattr(
            scm_mod, "build_identity_mapping_for_pipeline",
            lambda *a, **k: object())
        monkeypatch.setattr(
            nf_scm_parity, "measure_readout_decision_drift",
            lambda *a, **k: 1.0)

        step = self._make_step("ttfs_cycle_based")
        model = SimpleNamespace(get_perceptrons=lambda: [])
        step._run_readout_decision_drift_diagnostic(
            model=model, ir_graph=object())

    def test_can_be_explicitly_disabled(self, monkeypatch):
        import mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step as scm_mod

        built = []
        monkeypatch.setattr(
            scm_mod, "build_spiking_hybrid_flow",
            lambda *a, **k: built.append(1) or object(),
        )
        step = self._make_step("ttfs_cycle_based")
        step.pipeline.config["scm_torch_sim_parity_check"] = False
        step._run_readout_decision_drift_diagnostic(model=object(), ir_graph=object())
        assert built == [], "explicit opt-out must skip the report"


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
    """E5(c): the code guard's supported-version pin and the version the install
    manifest declares (`pyproject.toml`'s `sanafe` extra) must agree. A one-sided
    bump (or an unpinned `sanafe` requirement that floats past the guard) fails
    loud here in CI instead of SIGFPE-ing at deploy time."""

    def test_manifest_pin_matches_code_guard(self):
        version = assert_sanafe_pin_consistent()
        assert version in sanafe_supported_versions()

    def test_manifest_declares_a_pin(self):
        assert manifest_pinned_sanafe_version() is not None, (
            "pyproject.toml's `sanafe` extra must pin sanafe==<version>"
        )

    def test_drift_fails_loud_on_mismatch(self, tmp_path):
        manifest = tmp_path / "pyproject.toml"
        manifest.write_text(
            '[project.optional-dependencies]\nsanafe = ["sanafe==99.0.0"]\n',
            encoding="utf-8",
        )
        with pytest.raises(AssertionError, match="drift"):
            assert_sanafe_pin_consistent(str(manifest))

    def test_drift_fails_loud_when_pin_absent(self, tmp_path):
        manifest = tmp_path / "pyproject.toml"
        manifest.write_text(
            '[project.optional-dependencies]\nsanafe = ["sanafe"]  # unpinned!\n',
            encoding="utf-8",
        )
        with pytest.raises(AssertionError, match="no .*pin"):
            assert_sanafe_pin_consistent(str(manifest))


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
            nf_scm_parity, DEPLOYED_METRIC_PROTOCOL["readout_drift_report"]
        )
        assert hasattr(
            soft_core_mapping_step, DEPLOYED_METRIC_PROTOCOL["metric_step"]
        )

    def test_metric_step_runs_metric_after_the_gate_and_the_report(self):
        """The deployment step must run the gate and the report BEFORE producing
        the metric (neither can run after the number is already trusted)."""
        import inspect
        from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import (
            SoftCoreMappingStep,
        )

        src = inspect.getsource(SoftCoreMappingStep.process)
        gate_pos = src.index("_run_readout_decision_drift_diagnostic")
        nf_gate_pos = src.index("_run_nf_scm_parity_gate")
        metric_pos = src.index("run_scm_identity_metric")
        assert nf_gate_pos < metric_pos
        assert gate_pos < metric_pos
