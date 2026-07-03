"""[MBH-GATE] probe-gated fast ladder (MBH theory, experiment X2/E1, prediction T3).

Gated by ``MIMARSINAN_MBH_GATE`` through the env SSOT, default OFF (flag off ==
today's no-reject fast ladder, bit-identical). When ON, every fast-ladder rung
becomes a D-hat trust-region attempt: snapshot -> train -> measure the deployed
full-transform accuracy (fp32, clone-based) -> ACCEPT iff D-hat >= best - 0.01,
else restore and retry the midpoint rate (max 3 refinements), then CONSTRUCTIVE
STALL: stop consuming rungs and restore the best-D-hat snapshot. The pipeline's
force-to-1.0 finalize contract stays intact; the ``_continue_to_full_rate``
micro-ramp is skipped under the flag (no training on destructive intermediate
rates); gate ON implies the [MBH] ledger measurements (both line families emit).
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


def _set_gate(monkeypatch, enabled, *, ledger=False):
    if enabled:
        monkeypatch.setenv(env.MBH_GATE_VAR, "1")
    else:
        monkeypatch.delenv(env.MBH_GATE_VAR, raising=False)
    if ledger:
        monkeypatch.setenv(env.MBH_LEDGER_VAR, "1")
    else:
        monkeypatch.delenv(env.MBH_LEDGER_VAR, raising=False)


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


# -- the env SSOT accessor --------------------------------------------------------

class TestEnvFlag:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv(env.MBH_GATE_VAR, raising=False)
        assert env.mbh_gate_enabled() is False

    def test_exactly_one_enables(self, monkeypatch):
        monkeypatch.setenv(env.MBH_GATE_VAR, "1")
        assert env.mbh_gate_enabled() is True

    def test_other_values_stay_off(self, monkeypatch):
        for value in ("0", "true", "yes", ""):
            monkeypatch.setenv(env.MBH_GATE_VAR, value)
            assert env.mbh_gate_enabled() is False


# -- flag OFF == today ------------------------------------------------------------

class TestGateOffBitIdentity:
    def test_off_matches_unset_and_emits_nothing(self, tmp_path, monkeypatch, capsys):
        _set_gate(monkeypatch, False)
        torch.manual_seed(0)
        t_unset = _clamp_tuner(tmp_path / "unset")
        try:
            t_unset.run()
        finally:
            t_unset.close()
        out_unset = capsys.readouterr().out

        monkeypatch.setenv(env.MBH_GATE_VAR, "0")
        torch.manual_seed(0)
        t_zero = _clamp_tuner(tmp_path / "zero")
        try:
            t_zero.run()
        finally:
            t_zero.close()
        out_zero = capsys.readouterr().out

        assert not _gate_lines(out_unset) and not _gate_lines(out_zero)
        assert not _ledger_lines(out_unset) and not _ledger_lines(out_zero)
        _assert_state_equal(
            t_unset.model.state_dict(), t_zero.model.state_dict()
        )

    def test_off_continue_to_full_rate_delegates_to_run_mixin(
        self, tmp_path, monkeypatch,
    ):
        from mimarsinan.tuning.orchestration.smooth_adaptation_run import (
            SmoothAdaptationRunMixin,
        )

        _set_gate(monkeypatch, False)
        calls = []
        monkeypatch.setattr(
            SmoothAdaptationRunMixin, "_continue_to_full_rate",
            lambda self: calls.append(True),
        )
        tuner = _clamp_tuner(tmp_path)
        try:
            tuner._continue_to_full_rate()
            assert calls == [True]
        finally:
            tuner.close()

    def test_on_skips_continue_to_full_rate(self, tmp_path, monkeypatch):
        from mimarsinan.tuning.orchestration.smooth_adaptation_run import (
            SmoothAdaptationRunMixin,
        )

        _set_gate(monkeypatch, True)
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


# -- accept path -------------------------------------------------------------------

class TestAcceptPath:
    def test_all_accepts_match_flag_off_bitwise(self, tmp_path, monkeypatch, capsys):
        # With a never-regressing D-hat, the gated trajectory is bit-identical to
        # the ungated one: measurements are isolated, snapshots are read-only.
        _set_gate(monkeypatch, False)
        torch.manual_seed(0)
        t_off = _clamp_tuner(tmp_path / "off")
        try:
            t_off.run()
        finally:
            t_off.close()
        capsys.readouterr()

        _set_gate(monkeypatch, True)
        _inject_measurements(monkeypatch, entry=0.4, full_accs=[0.5, 0.6])
        torch.manual_seed(0)
        t_on = _clamp_tuner(tmp_path / "on")
        try:
            t_on.run()
        finally:
            t_on.close()
        out_on = capsys.readouterr().out

        _assert_state_equal(t_off.model.state_dict(), t_on.model.state_dict())
        assert [e["post_acc"] for e in t_off._cycle_log] == \
            [e["post_acc"] for e in t_on._cycle_log]
        assert t_on._committed_rate == pytest.approx(1.0)
        assert [e["outcome"] for e in t_on._cycle_log] == ["commit", "commit"]

        gate = _gate_lines(out_on)
        assert len(gate) == 3
        assert gate[0].startswith("[MBH-GATE] tuner=ClampTuner entry best_full_acc=")
        assert all(" accept " in line for line in gate[1:])

    def test_within_tolerance_dip_is_accepted(self, tmp_path, monkeypatch, capsys):
        _set_gate(monkeypatch, True)
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

    def test_gate_implies_ledger_lines(self, tmp_path, monkeypatch, capsys):
        _set_gate(monkeypatch, True, ledger=False)
        _inject_measurements(monkeypatch, entry=0.4, full_accs=[0.5, 0.6])
        torch.manual_seed(0)
        tuner = _clamp_tuner(tmp_path)
        try:
            tuner.run()
        finally:
            tuner.close()
        out = capsys.readouterr().out
        assert len(_ledger_lines(out)) == 2, "gate ON must emit [MBH] lines too"

    def test_real_measurements_smoke(self, tmp_path, monkeypatch, capsys):
        # No injection: the actual clone-based D-hat plumbing drives the gate.
        _set_gate(monkeypatch, True)
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


# -- reject-restore path -----------------------------------------------------------

class TestRejectRestore:
    def test_reject_restores_and_bisects_then_stalls(
        self, tmp_path, monkeypatch, capsys,
    ):
        _set_gate(monkeypatch, True)
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

            # stalled: further rungs are not consumed (no training, no lines)
            committed = tuner._driver_attempt(1.0)
            assert committed == pytest.approx(0.5)
            assert rates_seen == [0.5, 1.0, 0.75, 0.625, 0.5625]
            assert not _ledger_lines(capsys.readouterr().out)
        finally:
            tuner.close()

    def test_reject_restores_optimizer_state(self, tmp_path, monkeypatch):
        _set_gate(monkeypatch, True)
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
        _set_gate(monkeypatch, True)
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
            # no _continue_to_full_rate micro-ramp cycles under the gate
            assert adaptation_calls == []
            # every rejected attempt was rolled back (optimizer steps restored)
            assert tuner._fast_optimizer_steps == 0
        finally:
            tuner.close()
        out = capsys.readouterr().out
        stalls = [l for l in _gate_lines(out) if "constructive_stall" in l]
        assert len(stalls) == 1
        # only rung 0's 4 attempts were measured; the second rung was not consumed
        assert len(_ledger_lines(out)) == 4

    def test_kd_blend_stall_finalizes_on_best_state(
        self, tmp_path, monkeypatch, capsys,
    ):
        _set_gate(monkeypatch, True)
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
