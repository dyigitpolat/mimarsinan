"""The D-hat-gated fast ladder — the DEFAULT ladder driver (MBH X3, from X2/T3).

Every fast-ladder rung is a D-hat trust-region attempt: snapshot -> train ->
measure the deployed full-transform accuracy (fp32, clone-based) -> ACCEPT iff
D-hat >= best - 0.01, else restore and retry the midpoint rate (max 3
refinements), then CONSTRUCTIVE STALL: stop consuming rungs and restore the
best-D-hat snapshot. There is no ungated fast path anymore: the recipe IS the
gate. The pipeline's force-to-1.0 finalize contract stays intact; the
``_continue_to_full_rate`` micro-ramp is always skipped on the fixed ladder (no
training on destructive intermediate rates). ``MIMARSINAN_MBH_LEDGER`` is the
verbose-diagnostics flag only: gate probes run regardless; per-rung [MBH] lines
print only under the flag.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from conftest import (
    MockPipeline,
    default_config,
    make_activation_scale_stats,
    make_tiny_supermodel,
)
from mimarsinan.common import env
from mimarsinan.tuning.orchestration import mbh_ledger
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
    create_adaptation_manager_for_model,
)


def _gate_lines(text):
    return [line for line in text.splitlines() if line.startswith("[MBH-GATE] ")]


def _ledger_lines(text):
    return [line for line in text.splitlines() if line.startswith("[MBH] ")]


# -- fixtures (mirror test_mbh_ledger) -------------------------------------------

def _clamp_tuner(tmp_path, *, steps_per_rate=2, rates=(0.5, 1.0), target_metric=0.5):
    from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner

    cfg = default_config()
    cfg["spiking_mode"] = "ttfs_quantized"
    cfg["activation_quantization"] = True
    cfg["optimization_driver"] = "fast"
    cfg["clamp_fast_rates"] = list(rates)
    cfg["clamp_fast_steps_per_rate"] = steps_per_rate
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = target_metric
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    scales = [1.0 for _ in model.get_perceptrons()]
    stats = make_activation_scale_stats(model, scales)
    return ClampTuner(pipeline, model, 0.5, cfg["lr"], manager, scales, stats)


def _lif_tuner(tmp_path, *, steps_per_rate=2, rates=(0.5, 1.0), target_metric=0.0):
    from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

    cfg = default_config()
    cfg["spiking_mode"] = "lif"
    cfg["firing_mode"] = "Default"
    cfg["thresholding_mode"] = "<"
    cfg["simulation_steps"] = 4
    cfg["lif_blend_fast"] = True
    cfg["lif_blend_fast_steps_per_rate"] = steps_per_rate
    cfg["lif_blend_fast_rates"] = list(rates)
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = target_metric
    model = make_tiny_supermodel()
    return LIFAdaptationTuner(
        pipeline, model=model, target_accuracy=0.5, lr=cfg["lr"],
        adaptation_manager=AdaptationManager(),
    )


def _inject_measurements(monkeypatch, *, entry, full_accs):
    """Deterministic gate inputs: patch the mbh_ledger measurement functions the
    gate consults (it calls them through the module namespace by design)."""
    seq = list(full_accs)
    calls = {"i": 0}

    def fake_rung_measurements(tuner):
        i = min(calls["i"], len(seq) - 1)
        calls["i"] += 1
        full = float(seq[i])
        return {
            "blended_fp32": full, "full_acc": full,
            "rho": 1.0, "grad_norm_t": 0.0,
        }

    monkeypatch.setattr(mbh_ledger, "rung_measurements", fake_rung_measurements)
    monkeypatch.setattr(
        mbh_ledger, "full_transform_measurement", lambda tuner: float(entry)
    )
    return calls


def _prepare_direct_attempts(tuner):
    """Run-scope scratch normally set by ``run()`` for driving attempts directly."""
    tuner._phase_seconds = {}
    tuner._mbh_rung_index = -1
    tuner._mbh_gate_state = None


def _state_dict_clone(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def _assert_state_equal(sd_a, sd_b):
    assert sd_a.keys() == sd_b.keys()
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), key


def _ungated_ladder_replay(tuner):
    """The pre-X3 (flag-off) fast ladder, replayed through the same seam verbs:
    train every rung, commit unconditionally. The equivalence reference for the
    all-accepts trajectory."""
    _prepare_direct_attempts(tuner)
    for target in tuner._fixed_ladder_rates:
        tuner._ensure_fast_optimizer()
        tuner._fast_ramp(float(target))
        tuner._committed_rate = float(target)
        post_acc = tuner.probe()
        tuner._last_post_acc = post_acc
        tuner._fast_probe(float(target))
    return tuner


# -- the two-sided rung trust region (WS-A A1) --------------------------------------

class TestRetentionGate:
    """A rung must retain the BLENDED metric (post_acc >= previous committed
    post_acc - tolerance) in addition to the D-hat bound. Measured motivation:
    a ViT rung at lr 2.89e-3 destroyed the blend 0.82->0.31 while D-hat
    improved 0.02->0.22 and the one-sided gate ACCEPTED the wreck."""

    def _probe_seq(self, tuner, values):
        seq = list(values)
        tuner.probe = lambda: float(seq.pop(0)) if seq else float(values[-1])

    @staticmethod
    def _arm(tuner):
        # 100 classes -> chance floor 5/100 = 0.05: the 0.8x anchors arm.
        tuner.pipeline.config["num_classes"] = 100

    def test_destructive_rung_is_rejected_restored_and_lr_backed_off(
        self, tmp_path, monkeypatch, capsys,
    ):
        # D-hat improves but the blend craters: reject, restore, halve the LR,
        # retry the midpoint; the retry (post-backoff) retains -> accepted.
        _inject_measurements(monkeypatch, entry=0.02, full_accs=[0.22, 0.25])
        tuner = _clamp_tuner(tmp_path)
        try:
            _prepare_direct_attempts(tuner)
            self._arm(tuner)
            # probe reads: entry anchor 0.85, wrecked rung 0.30, healthy retry 0.84
            self._probe_seq(tuner, [0.85, 0.30, 0.84])
            tuner._ensure_fast_optimizer()
            base_before = [
                list(s.base_lrs) for s in tuner._fast_lr_schedule._schedulers
            ]
            committed = tuner._driver_attempt(0.5)
            out = capsys.readouterr().out
            assert "reject" in out and "reason=retention" in out
            assert committed == pytest.approx(0.25)  # midpoint of (0.0, 0.5)
            base_after = [
                list(s.base_lrs) for s in tuner._fast_lr_schedule._schedulers
            ]
            for before, after in zip(base_before, base_after):
                for b, a in zip(before, after):
                    assert a == pytest.approx(0.5 * b)
            # LR x steps preserved: the accepted retry ran a DOUBLED budget
            # and the scale reset on accept.
            assert tuner._fast_retry_step_scale == 1
            assert tuner._fast_optimizer_steps == 2 * tuner._fast_steps_per_rate
        finally:
            tuner.close()

    def test_dhat_reject_does_not_back_off_the_lr(
        self, tmp_path, monkeypatch, capsys,
    ):
        # The one-sided D-hat reject is a RATE problem (midpoint retry), not an
        # LR problem: the backoff must not fire.
        _inject_measurements(monkeypatch, entry=0.50, full_accs=[0.20, 0.55])
        tuner = _clamp_tuner(tmp_path)
        try:
            _prepare_direct_attempts(tuner)
            self._arm(tuner)
            self._probe_seq(tuner, [0.85, 0.84, 0.84])
            tuner._ensure_fast_optimizer()
            base_before = [
                list(s.base_lrs) for s in tuner._fast_lr_schedule._schedulers
            ]
            tuner._driver_attempt(0.5)
            out = capsys.readouterr().out
            assert "reason=dhat" in out
            base_after = [
                list(s.base_lrs) for s in tuner._fast_lr_schedule._schedulers
            ]
            assert base_after == base_before
        finally:
            tuner.close()

    def test_anchor_drifts_to_the_accepted_post_acc(self, tmp_path, monkeypatch):
        _inject_measurements(monkeypatch, entry=0.4, full_accs=[0.5, 0.6])
        tuner = _clamp_tuner(tmp_path)
        try:
            _prepare_direct_attempts(tuner)
            self._arm(tuner)
            self._probe_seq(tuner, [0.85, 0.845, 0.84])
            for target in tuner._fixed_ladder_rates:
                tuner._driver_attempt(target)
            assert tuner._mbh_gate_state.prev_post_acc == pytest.approx(0.84)
        finally:
            tuner.close()

    def test_entry_anchor_read_is_rng_isolated(self, tmp_path, monkeypatch):
        from mimarsinan.tuning.orchestration.mbh_gate import _ensure_gate_state

        _inject_measurements(monkeypatch, entry=0.4, full_accs=[0.5])
        tuner = _clamp_tuner(tmp_path)
        try:
            _prepare_direct_attempts(tuner)
            self._arm(tuner)
            rng_before = torch.random.get_rng_state()
            state = _ensure_gate_state(tuner)
            assert state.prev_post_acc is not None
            assert torch.equal(rng_before, torch.random.get_rng_state())
        finally:
            tuner.close()

    def test_chance_level_anchor_disarms_retention(self, tmp_path, monkeypatch):
        # A relative retention bound on a chance-level backbone gates noise:
        # entry below 2/num_classes leaves the gate one-sided (D-hat only).
        _inject_measurements(monkeypatch, entry=0.02, full_accs=[0.22])
        tuner = _clamp_tuner(tmp_path)
        try:
            _prepare_direct_attempts(tuner)
            # entry anchor 0.10 < 2/4 classes: disarmed; wrecked rung accepted
            # on the D-hat criterion alone (the historical one-sided gate).
            self._probe_seq(tuner, [0.10, 0.05])
            committed = tuner._driver_attempt(0.5)
            assert committed == pytest.approx(0.5)
            assert tuner._mbh_gate_state.retention_armed is False
        finally:
            tuner.close()

    def test_retention_exhaustion_stalls_on_the_best_state(
        self, tmp_path, monkeypatch, capsys,
    ):
        # Every attempt wrecks the blend: refinements exhaust into the
        # constructive stall exactly like the D-hat path.
        _inject_measurements(
            monkeypatch, entry=0.02, full_accs=[0.3, 0.3, 0.3, 0.3],
        )
        tuner = _clamp_tuner(tmp_path)
        try:
            _prepare_direct_attempts(tuner)
            self._arm(tuner)
            self._probe_seq(tuner, [0.85] + [0.30] * 8)
            committed = tuner._driver_attempt(0.5)
            out = capsys.readouterr().out
            assert "constructive_stall" in out
            assert committed == pytest.approx(0.0)
            assert tuner._mbh_gate_state.stalled is True
        finally:
            tuner.close()


class TestBestDeployedFinalize:
    """[WS-A A1] finalize arbitration: when the ladder ends below its target
    rate, the hard finalize can wreck the model while a better DEPLOYED
    candidate was observed mid-ladder (measured: a rate-1.0 attempt read
    0.7515 full-ReLU and was retention-rejected; the sub-1.0 commit's hard
    swap read 0.06). The gate tracks the best deployed state across ALL
    attempts (accepted or rejected) and the finalize seam restores it when the
    final state's deployed read falls short by more than the tolerance."""

    def _probe_seq(self, tuner, values):
        seq = list(values)
        tuner.probe = lambda: float(seq.pop(0)) if seq else float(values[-1])

    @staticmethod
    def _arm(tuner):
        tuner.pipeline.config["num_classes"] = 100

    def test_rejected_attempt_with_best_deployed_read_is_tracked(
        self, tmp_path, monkeypatch,
    ):
        # attempt full_accs: rung0 accepted 0.30; rung1 attempt REJECTED by
        # retention with deployed 0.75 (the best), retry accepted at 0.28.
        _inject_measurements(
            monkeypatch, entry=0.02, full_accs=[0.30, 0.75, 0.28],
        )
        tuner = _clamp_tuner(tmp_path)
        try:
            _prepare_direct_attempts(tuner)
            self._arm(tuner)
            self._probe_seq(tuner, [0.87, 0.86, 0.10, 0.855])
            tuner._driver_attempt(0.5)
            tuner._driver_attempt(1.0)
            state = tuner._mbh_gate_state
            assert state.best_deployed_acc == pytest.approx(0.75)
            assert state.best_deployed_state is not None
        finally:
            tuner.close()

    def test_finalize_restores_the_best_deployed_state_when_final_falls_short(
        self, tmp_path, monkeypatch,
    ):
        from mimarsinan.tuning.orchestration.mbh_gate import (
            finalize_on_best_deployed,
        )

        _inject_measurements(
            monkeypatch, entry=0.02, full_accs=[0.30, 0.75, 0.28, 0.06],
        )
        tuner = _clamp_tuner(tmp_path)
        try:
            _prepare_direct_attempts(tuner)
            self._arm(tuner)
            self._probe_seq(tuner, [0.87, 0.86, 0.10, 0.855])
            tuner._driver_attempt(0.5)
            tuner._driver_attempt(1.0)
            # final read (the injected entry, 0.02) < best deployed (0.75) - tol:
            # the restore fires and reports the restored deployed read.
            restored = finalize_on_best_deployed(tuner)
            assert restored == pytest.approx(0.75)
        finally:
            tuner.close()

    def test_finalize_is_inert_when_the_final_state_is_the_best(
        self, tmp_path, monkeypatch,
    ):
        from mimarsinan.tuning.orchestration.mbh_gate import (
            finalize_on_best_deployed,
        )

        # Healthy ladder: deployed climbs monotonically; final read matches.
        _inject_measurements(
            monkeypatch, entry=0.02, full_accs=[0.30, 0.60, 0.60],
        )
        tuner = _clamp_tuner(tmp_path)
        try:
            _prepare_direct_attempts(tuner)
            self._arm(tuner)
            self._probe_seq(tuner, [0.87, 0.86, 0.855])
            tuner._driver_attempt(0.5)
            tuner._driver_attempt(1.0)
            # the final state reads at the deployed best: arbitration is inert.
            monkeypatch.setattr(
                mbh_ledger, "full_transform_measurement", lambda t: 0.60,
            )
            assert finalize_on_best_deployed(tuner) is None
        finally:
            tuner.close()

    def test_finalize_without_gate_state_is_inert(self, tmp_path):
        from mimarsinan.tuning.orchestration.mbh_gate import (
            finalize_on_best_deployed,
        )

        tuner = _clamp_tuner(tmp_path)
        try:
            tuner._mbh_gate_state = None
            assert finalize_on_best_deployed(tuner) is None
        finally:
            tuner.close()


# -- default equivalence: all-accepts == the historical ungated ladder --------------

class TestDefaultEquivalence:
    def test_all_accepts_match_the_ungated_ladder_bitwise(
        self, tmp_path, monkeypatch, capsys,
    ):
        # With a never-regressing D-hat the gated trajectory is bit-identical to
        # the historical ungated ladder: measurements are isolated, snapshots are
        # read-only. This is the recipe-default form of the old flag-off==flag-on
        # equivalence proof.
        torch.manual_seed(0)
        t_ref = _clamp_tuner(tmp_path / "ref")
        try:
            _ungated_ladder_replay(t_ref)
        finally:
            t_ref.close()
        capsys.readouterr()

        _inject_measurements(monkeypatch, entry=0.4, full_accs=[0.5, 0.6])
        torch.manual_seed(0)
        t_gated = _clamp_tuner(tmp_path / "gated")
        try:
            _prepare_direct_attempts(t_gated)
            for target in t_gated._fixed_ladder_rates:
                t_gated._driver_attempt(target)
        finally:
            t_gated.close()
        out = capsys.readouterr().out

        _assert_state_equal(t_ref.model.state_dict(), t_gated.model.state_dict())
        assert t_gated._fast_optimizer_steps == t_ref._fast_optimizer_steps
        assert t_gated._committed_rate == pytest.approx(1.0)
        gate = _gate_lines(out)
        assert gate[0].startswith("[MBH-GATE] tuner=ClampTuner entry best_full_acc=")
        assert all(" accept " in line for line in gate[1:])


# -- accept path -------------------------------------------------------------------

class TestAcceptPath:
    def test_full_run_commits_every_rung(self, tmp_path, monkeypatch, capsys):
        _inject_measurements(monkeypatch, entry=0.4, full_accs=[0.5, 0.6])
        torch.manual_seed(0)
        tuner = _clamp_tuner(tmp_path)
        try:
            tuner.run()
        finally:
            tuner.close()
        out = capsys.readouterr().out

        assert tuner._committed_rate == pytest.approx(1.0)
        assert [e["outcome"] for e in tuner._cycle_log] == ["commit", "commit"]
        gate = _gate_lines(out)
        assert len(gate) == 3
        assert gate[0].startswith("[MBH-GATE] tuner=ClampTuner entry best_full_acc=")
        assert all(" accept " in line for line in gate[1:])

    def test_within_tolerance_dip_is_accepted(self, tmp_path, monkeypatch, capsys):
        _inject_measurements(monkeypatch, entry=0.5, full_accs=[0.495, 0.505])
        torch.manual_seed(0)
        tuner = _clamp_tuner(tmp_path)
        try:
            tuner.run()
        finally:
            tuner.close()
        out = capsys.readouterr().out
        assert not any(" reject " in line for line in _gate_lines(out))
        assert tuner._committed_rate == pytest.approx(1.0)
        assert tuner._mbh_gate_state.best_full_acc == pytest.approx(0.505)

    def test_real_measurements_smoke(self, tmp_path, monkeypatch, capsys):
        # No injection: the actual clone-based D-hat plumbing drives the gate.
        torch.manual_seed(0)
        tuner = _clamp_tuner(tmp_path)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tuner.run()
            assert tuner._committed_rate == pytest.approx(1.0)
        finally:
            tuner.close()
        out = capsys.readouterr().out
        assert _gate_lines(out)[0].startswith(
            "[MBH-GATE] tuner=ClampTuner entry best_full_acc="
        )


# -- the ledger flag is verbose diagnostics only ------------------------------------

class TestLedgerFlagIsVerboseOnly:
    def test_default_emits_gate_lines_but_no_ledger_lines(
        self, tmp_path, monkeypatch, capsys,
    ):
        monkeypatch.delenv(env.MBH_LEDGER_VAR, raising=False)
        _inject_measurements(monkeypatch, entry=0.4, full_accs=[0.5, 0.6])
        torch.manual_seed(0)
        tuner = _clamp_tuner(tmp_path)
        try:
            tuner.run()
        finally:
            tuner.close()
        out = capsys.readouterr().out
        assert _gate_lines(out), "gate decisions always print"
        assert not _ledger_lines(out), "[MBH] rung lines are ledger-flag-only"

    def test_ledger_flag_adds_rung_lines(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv(env.MBH_LEDGER_VAR, "1")
        _inject_measurements(monkeypatch, entry=0.4, full_accs=[0.5, 0.6])
        torch.manual_seed(0)
        tuner = _clamp_tuner(tmp_path)
        try:
            tuner.run()
        finally:
            tuner.close()
        out = capsys.readouterr().out
        assert len(_ledger_lines(out)) == 2, "one [MBH] line per attempted rung"


# -- the fixed ladder always skips the forced-jump micro-ramp -----------------------

class TestContinueToFullRate:
    def test_fixed_ladder_skips_the_micro_ramp(self, tmp_path, monkeypatch):
        from mimarsinan.tuning.orchestration.smooth_adaptation_run import (
            SmoothAdaptationRunMixin,
        )

        calls = []
        monkeypatch.setattr(
            SmoothAdaptationRunMixin, "_continue_to_full_rate",
            lambda self: calls.append(True),
        )
        tuner = _clamp_tuner(tmp_path)
        try:
            tuner._committed_rate = 0.5
            tuner._continue_to_full_rate()
            assert calls == []
        finally:
            tuner.close()

    def test_controller_path_delegates_to_run_mixin(self, tmp_path, monkeypatch):
        from mimarsinan.tuning.orchestration.smooth_adaptation_run import (
            SmoothAdaptationRunMixin,
        )

        calls = []
        monkeypatch.setattr(
            SmoothAdaptationRunMixin, "_continue_to_full_rate",
            lambda self: calls.append(True),
        )
        tuner = _clamp_tuner(tmp_path)
        try:
            tuner._fixed_ladder_policy = False
            tuner._continue_to_full_rate()
            assert calls == [True]
        finally:
            tuner.close()


# -- reject-restore path -----------------------------------------------------------

class TestRejectRestore:
    def test_reject_restores_and_bisects_then_stalls(
        self, tmp_path, monkeypatch, capsys,
    ):
        # rung 0 (rate 0.5) improves past the entry; rung 1 regresses on every
        # attempt: 1.0 -> midpoints 0.75, 0.625, 0.5625 -> constructive stall.
        _inject_measurements(
            monkeypatch, entry=0.9,
            full_accs=[0.95, 0.5, 0.5, 0.5, 0.5],
        )
        torch.manual_seed(0)
        tuner = _clamp_tuner(tmp_path)
        try:
            _prepare_direct_attempts(tuner)
            rates_seen = []
            orig_ramp = tuner._fast_ramp
            monkeypatch.setattr(
                tuner, "_fast_ramp",
                lambda r: (rates_seen.append(float(r)), orig_ramp(r))[1],
            )

            committed = tuner._driver_attempt(0.5)
            assert committed == pytest.approx(0.5)
            best_sd = _state_dict_clone(tuner.model)
            steps_after_rung0 = tuner._fast_optimizer_steps

            committed = tuner._driver_attempt(1.0)
            assert committed == pytest.approx(0.5)
            assert rates_seen == [0.5, 1.0, 0.75, 0.625, 0.5625]
            # the stall restored the best-D-hat snapshot (post-rung-0 state)
            _assert_state_equal(tuner.model.state_dict(), best_sd)
            assert tuner._fast_optimizer_steps == steps_after_rung0
            assert tuner._committed_rate == pytest.approx(0.5)
            assert [e["outcome"] for e in tuner._cycle_log] == \
                ["commit"] + ["rollback"] * 4

            out = capsys.readouterr().out
            stalls = [l for l in _gate_lines(out) if "constructive_stall" in l]
            assert stalls == [
                "[MBH-GATE] constructive_stall committed=0.500000 "
                "best_full_acc=0.950000"
            ]

            # stalled: further rungs are not consumed (no training, no attempts)
            committed = tuner._driver_attempt(1.0)
            assert committed == pytest.approx(0.5)
            assert rates_seen == [0.5, 1.0, 0.75, 0.625, 0.5625]
        finally:
            tuner.close()

    def test_reject_restores_optimizer_state(self, tmp_path, monkeypatch):
        _inject_measurements(monkeypatch, entry=0.9, full_accs=[0.1])
        torch.manual_seed(0)
        tuner = _clamp_tuner(tmp_path)
        try:
            _prepare_direct_attempts(tuner)
            tuner._ensure_fast_optimizer()
            pre_lr = float(tuner._fast_optimizer.param_groups[0]["lr"])
            pre_sd = _state_dict_clone(tuner.model)

            tuner._driver_attempt(0.5)

            # everything rolled back to the best (== entry) snapshot
            _assert_state_equal(tuner.model.state_dict(), pre_sd)
            assert tuner._fast_optimizer_steps == 0
            assert float(tuner._fast_optimizer.param_groups[0]["lr"]) == \
                pytest.approx(pre_lr)
        finally:
            tuner.close()


# -- stall path through the full run (finalize contract intact) --------------------

class TestStallRun:
    def test_stall_run_completes_and_forces_full_rate(
        self, tmp_path, monkeypatch, capsys,
    ):
        _inject_measurements(monkeypatch, entry=0.9, full_accs=[0.1])
        torch.manual_seed(0)
        tuner = _clamp_tuner(tmp_path, target_metric=0.0)
        try:
            adaptation_calls = []
            monkeypatch.setattr(
                tuner, "_adaptation",
                lambda rate: adaptation_calls.append(rate),
            )
            with pytest.warns(UserWarning, match="natural adaptation reached only"):
                tuner.run()
            # forced to 1.0 by _after_run; the finalize assertion held
            assert tuner._committed_rate == pytest.approx(1.0)
            assert tuner._natural_rate == pytest.approx(0.0)
            # no _continue_to_full_rate micro-ramp cycles on the fixed ladder
            assert adaptation_calls == []
            # every rejected attempt was rolled back (optimizer steps restored)
            assert tuner._fast_optimizer_steps == 0
        finally:
            tuner.close()
        out = capsys.readouterr().out
        stalls = [l for l in _gate_lines(out) if "constructive_stall" in l]
        assert len(stalls) == 1

    def test_kd_blend_stall_finalizes_on_best_state(
        self, tmp_path, monkeypatch, capsys,
    ):
        _inject_measurements(monkeypatch, entry=0.9, full_accs=[0.1])
        torch.manual_seed(0)
        tuner = _lif_tuner(tmp_path)
        try:
            with pytest.warns(UserWarning, match="natural adaptation reached only"):
                tuner.run()
            assert tuner._committed_rate == pytest.approx(1.0)
            assert tuner._mbh_gate_state.stalled is True
        finally:
            tuner.close()
        out = capsys.readouterr().out
        assert any("constructive_stall" in l for l in _gate_lines(out))
