"""Measurement-only [MBH] fast-ladder rung ledger (MBH theory, experiments X1/X2).

Gated by ``MIMARSINAN_MBH_LEDGER`` through the env SSOT. Per fast-ladder rung the
shared seam emits exactly one stdout line::

    [MBH] tuner=<cls> rung=<i> rate=<r> blended_acc=<a> blended_fp32=<b> full_acc=<d> rho=<rho> grad_norm_t=<g>

where ``blended_acc`` reuses the rung's existing probe, ``blended_fp32`` re-reads the
blended model gate-grade (fp32, autocast-disabled — X2/E0), ``full_acc`` (D-hat) is
the full-transformation (rate 1.0) accuracy on an isolated deepcopy, and ``rho`` is
the transfer alignment <g1, gt>/||gt||^2 from plain-CE gradients on one fixed
validation batch (computed only under the ledger flag — the default D-hat gate
does not consume it). HARD INVARIANT: ledger ON == OFF bit-identical training
trajectory; all measurement runs on deepcopies inside ``fork_rng`` with live
model/optimizer/RNG untouched. One [MBH] line prints per fast-ladder ATTEMPT
(the default gate's rejects included), flag-gated.
"""

from __future__ import annotations

import math
import re

import pytest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from conftest import (
    MockPipeline,
    default_config,
    make_activation_scale_stats,
    make_tiny_supermodel,
)
from mimarsinan.common import env
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
    create_adaptation_manager_for_model,
)

_LEDGER_RE = re.compile(
    r"^\[MBH\] tuner=(?P<tuner>\S+) rung=(?P<rung>\d+) rate=(?P<rate>[0-9.]+) "
    r"blended_acc=(?P<blended>[0-9.]+) blended_fp32=(?P<blended_fp32>[0-9.]+) "
    r"full_acc=(?P<full>[0-9.]+) rho=(?P<rho>\S+) grad_norm_t=(?P<grad>\S+) "
    r"nonzero_grad_frac=(?P<nonzero_grad_frac>\S+)$"
)


def _ledger_lines(text):
    return [line for line in text.splitlines() if line.startswith("[MBH] ")]


def _set_flag(monkeypatch, enabled):
    if enabled:
        monkeypatch.setenv(env.MBH_LEDGER_VAR, "1")
    else:
        monkeypatch.delenv(env.MBH_LEDGER_VAR, raising=False)


# -- per-family fixtures (mirror test_lif_blend_fast / test_fast_ladder_base_lift /
#    test_genuine_blend_fast) ---------------------------------------------------

def _lif_tuner(tmp_path, *, fast=True, steps_per_rate=2, rates=(0.5, 1.0)):
    from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

    cfg = default_config()
    cfg["spiking_mode"] = "lif"
    cfg["firing_mode"] = "Default"
    cfg["thresholding_mode"] = "<"
    cfg["simulation_steps"] = 4
    cfg["lif_blend_fast"] = fast
    cfg["lif_blend_fast_steps_per_rate"] = steps_per_rate
    cfg["lif_blend_fast_rates"] = list(rates)
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    model = make_tiny_supermodel()
    return LIFAdaptationTuner(
        pipeline, model=model, target_accuracy=0.5, lr=cfg["lr"],
        adaptation_manager=AdaptationManager(),
    )


def _aq_tuner(tmp_path, *, steps_per_rate=2, rates=(0.5, 1.0)):
    from mimarsinan.tuning.tuners.activation_quantization_tuner import (
        ActivationQuantizationTuner,
    )

    cfg = default_config()
    cfg["spiking_mode"] = "ttfs_quantized"
    cfg["activation_quantization"] = True
    cfg["optimization_driver"] = "fast"
    cfg["manager_rate_fast_rates"] = list(rates)
    cfg["manager_rate_fast_steps_per_rate"] = steps_per_rate
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    return ActivationQuantizationTuner(pipeline, model, 4, 0.5, cfg["lr"], manager)


def _clamp_tuner(tmp_path, *, steps_per_rate=2, rates=(0.5, 1.0)):
    from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner

    cfg = default_config()
    cfg["spiking_mode"] = "ttfs_quantized"
    cfg["activation_quantization"] = True
    cfg["optimization_driver"] = "fast"
    cfg["clamp_fast_rates"] = list(rates)
    cfg["clamp_fast_steps_per_rate"] = steps_per_rate
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    scales = [1.0 for _ in model.get_perceptrons()]
    stats = make_activation_scale_stats(model, scales)
    return ClampTuner(pipeline, model, 0.5, cfg["lr"], manager, scales, stats)


def _ttfs_tuner(tmp_path, *, steps_per_rate=2, rates=(0.5, 1.0)):
    from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import (
        TTFSCycleAdaptationTuner,
    )

    cfg = default_config()
    cfg["spiking_mode"] = "ttfs_cycle_based"
    cfg["ttfs_cycle_schedule"] = "cascaded"
    cfg["activation_quantization"] = True
    cfg["simulation_steps"] = 16
    cfg["ttfs_genuine_blend_ramp"] = True
    cfg["ttfs_genuine_blend_fast"] = True
    cfg["ttfs_blend_fast_steps_per_rate"] = steps_per_rate
    cfg["ttfs_blend_fast_rates"] = list(rates)
    cfg["ttfs_distmatch_bias_iters"] = 3
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    model = make_tiny_supermodel()
    return TTFSCycleAdaptationTuner(
        pipeline, model=model, target_accuracy=0.5, lr=cfg["lr"],
        adaptation_manager=AdaptationManager(),
    )


def _run_seeded(builder, tmp_path, monkeypatch, enabled, **kw):
    _set_flag(monkeypatch, enabled)
    torch.manual_seed(0)
    tuner = builder(tmp_path, **kw)
    try:
        tuner.run()
    finally:
        tuner.close()
    return tuner


# -- the env SSOT accessor -----------------------------------------------------

class TestEnvFlag:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv(env.MBH_LEDGER_VAR, raising=False)
        assert env.mbh_ledger_enabled() is False

    def test_exactly_one_enables(self, monkeypatch):
        monkeypatch.setenv(env.MBH_LEDGER_VAR, "1")
        assert env.mbh_ledger_enabled() is True

    def test_other_values_stay_off(self, monkeypatch):
        for value in ("0", "true", "yes", ""):
            monkeypatch.setenv(env.MBH_LEDGER_VAR, value)
            assert env.mbh_ledger_enabled() is False


# -- (a) trajectory invariance: ON == OFF, bit-identical -------------------------

class TestTrajectoryInvariance:
    def _assert_invariant(self, builder, tmp_path, monkeypatch, capsys, **kw):
        t_off = _run_seeded(builder, tmp_path / "off", monkeypatch, False, **kw)
        out_off = capsys.readouterr().out
        t_on = _run_seeded(builder, tmp_path / "on", monkeypatch, True, **kw)
        out_on = capsys.readouterr().out

        assert not _ledger_lines(out_off), "flag OFF must emit no [MBH] lines"
        lines = _ledger_lines(out_on)
        assert len(lines) == len(t_on._cycle_log), (
            "flag ON must emit exactly one [MBH] line per fast-ladder attempt"
        )

        sd_off, sd_on = t_off.model.state_dict(), t_on.model.state_dict()
        assert sd_off.keys() == sd_on.keys()
        for key in sd_off:
            assert torch.equal(sd_off[key], sd_on[key]), (
                f"state_dict[{key}] diverged: the ledger perturbed the trajectory"
            )
        assert [entry["post_acc"] for entry in t_off._cycle_log] == \
            [entry["post_acc"] for entry in t_on._cycle_log]

    def test_lif_blend_family(self, tmp_path, monkeypatch, capsys):
        self._assert_invariant(_lif_tuner, tmp_path, monkeypatch, capsys)

    def test_manager_rate_family_stochastic_masks(self, tmp_path, monkeypatch, capsys):
        self._assert_invariant(_aq_tuner, tmp_path, monkeypatch, capsys)

    def test_ttfs_genuine_blend_family(self, tmp_path, monkeypatch, capsys):
        self._assert_invariant(_ttfs_tuner, tmp_path, monkeypatch, capsys)


# -- (b) ledger line format ------------------------------------------------------

class TestLedgerFormat:
    def test_lines_parse_and_rho_finite_on_smooth_ramp(
        self, tmp_path, monkeypatch, capsys,
    ):
        _set_flag(monkeypatch, True)
        torch.manual_seed(0)
        tuner = _clamp_tuner(tmp_path, rates=(0.25, 0.5, 1.0))
        try:
            tuner.run()
        finally:
            tuner.close()
        lines = _ledger_lines(capsys.readouterr().out)
        assert len(lines) == len(tuner._cycle_log)
        for rung, line in enumerate(lines):
            match = _LEDGER_RE.match(line)
            assert match is not None, f"unparseable ledger line: {line!r}"
            assert match["tuner"] == "ClampTuner"
            assert int(match["rung"]) == rung
            assert 0.0 <= float(match["blended"]) <= 1.0
            assert 0.0 <= float(match["full"]) <= 1.0
            # the clamp Mix blend is smooth everywhere: rho must be well-defined
            assert math.isfinite(float(match["rho"])), line
            assert math.isfinite(float(match["grad"])), line
        rates = [float(_LEDGER_RE.match(line)["rate"]) for line in lines]
        assert rates == [e["rate"] for e in tuner._cycle_log]

    def test_lif_line_names_the_tuner_class(self, tmp_path, monkeypatch, capsys):
        _set_flag(monkeypatch, True)
        torch.manual_seed(0)
        tuner = _lif_tuner(tmp_path, rates=(0.5, 1.0))
        try:
            tuner.run()
        finally:
            tuner.close()
        lines = _ledger_lines(capsys.readouterr().out)
        assert lines
        for line in lines:
            match = _LEDGER_RE.match(line)
            assert match is not None, line
            assert match["tuner"] == "LIFAdaptationTuner"

    def test_controller_path_emits_nothing(self, tmp_path, monkeypatch, capsys):
        # X1 instruments the fast ladder only; the controller path is untouched.
        _run_seeded(_lif_tuner, tmp_path, monkeypatch, True, fast=False)
        assert not _ledger_lines(capsys.readouterr().out)


class TestNonzeroGradFraction:
    """A5 (theory §4): rho is subspace-local — the ledger must also report the
    fraction of trainable parameter elements with nonzero gradient under the
    RAMP forward (captured at each rung's first backward), the one number that
    exposes severed-boundary gradient support (14/16 frozen layers, §5p)."""

    def test_trained_rungs_report_a_fraction_in_unit_interval(
        self, tmp_path, monkeypatch, capsys,
    ):
        _set_flag(monkeypatch, True)
        torch.manual_seed(0)
        tuner = _lif_tuner(tmp_path, rates=(0.5, 1.0))
        try:
            tuner.run()
        finally:
            tuner.close()
        lines = _ledger_lines(capsys.readouterr().out)
        assert lines
        for line in lines:
            match = _LEDGER_RE.match(line)
            assert match is not None, line
            fraction = float(match["nonzero_grad_frac"])
            assert 0.0 < fraction <= 1.0, line

    def test_projection_only_rungs_report_nan(self, tmp_path, monkeypatch, capsys):
        # steps_per_rate=0: no backward happens, the field must read nan.
        _set_flag(monkeypatch, True)
        torch.manual_seed(0)
        tuner = _lif_tuner(tmp_path, steps_per_rate=0, rates=(0.5, 1.0))
        try:
            tuner.run()
        finally:
            tuner.close()
        lines = _ledger_lines(capsys.readouterr().out)
        assert lines
        for line in lines:
            match = _LEDGER_RE.match(line)
            assert match is not None, line
            assert math.isnan(float(match["nonzero_grad_frac"])), line

    def test_helper_counts_missing_grads_as_zero(self):
        import torch.nn as nn

        from mimarsinan.tuning.orchestration.mbh_ledger import nonzero_grad_fraction

        model = nn.Sequential(nn.Linear(4, 3, bias=False), nn.Linear(3, 2, bias=False))
        x = torch.randn(2, 4)
        # Sever the second layer: loss depends only on the first layer's output.
        model[0](x).sum().backward()
        fraction = nonzero_grad_fraction(model)
        expected = 12 / (12 + 6)
        assert fraction == pytest.approx(expected)

    def test_helper_nan_when_no_trainable_params(self):
        import torch.nn as nn

        from mimarsinan.tuning.orchestration.mbh_ledger import nonzero_grad_fraction

        model = nn.Linear(2, 2)
        for p in model.parameters():
            p.requires_grad_(False)
        assert math.isnan(nonzero_grad_fraction(model))


# -- (c) the D-hat probe is non-destructive ---------------------------------------

class TestProbeNonDestructive:
    def test_manager_rate_probe_leaves_rate_state_and_rng_untouched(
        self, tmp_path, monkeypatch,
    ):
        from mimarsinan.tuning.orchestration.mbh_ledger import emit_fast_rung_ledger

        _set_flag(monkeypatch, True)
        torch.manual_seed(0)
        tuner = _aq_tuner(tmp_path)
        try:
            tuner._fast_set_rate(0.5)
            blended = tuner.probe()
            manager = tuner.adaptation_manager
            buffer = manager._rate_buffer("quantization_rate")
            assert buffer is not None
            pre_alpha = float(buffer.alpha)
            pre_field = float(manager.quantization_rate)
            pre_sd = {k: v.clone() for k, v in tuner.model.state_dict().items()}
            pre_cursor = tuner.trainer._gpu_val_cursor
            pre_rng = torch.get_rng_state()
            pre_generators = {
                key: gen.get_state().clone()
                for key, gen in tuner._axis._decision_generators.items()
            }

            line = emit_fast_rung_ledger(tuner, rate=0.5, blended_acc=blended)

            assert line is not None and _LEDGER_RE.match(line), line
            assert float(buffer.alpha) == pre_alpha == 0.5
            assert float(manager.quantization_rate) == pre_field
            post_sd = tuner.model.state_dict()
            for key in pre_sd:
                assert torch.equal(pre_sd[key], post_sd[key]), key
            assert tuner.trainer._gpu_val_cursor == pre_cursor
            assert torch.equal(torch.get_rng_state(), pre_rng)
            for key, state in pre_generators.items():
                assert torch.equal(
                    tuner._axis._decision_generators[key].get_state(), state
                ), f"live decision generator {key} advanced"
        finally:
            tuner.close()

    def test_kd_blend_probe_leaves_blend_rates_and_flags_untouched(
        self, tmp_path, monkeypatch,
    ):
        from mimarsinan.tuning.orchestration.mbh_ledger import emit_fast_rung_ledger

        _set_flag(monkeypatch, True)
        torch.manual_seed(0)
        tuner = _lif_tuner(tmp_path)
        try:
            tuner._fast_set_rate(0.5)
            blended = tuner.probe()
            pre_rates = [
                float(p.base_activation.rate) for p in tuner.model.get_perceptrons()
            ]
            assert tuner.adaptation_manager.lif_active is False
            pre_sd = {k: v.clone() for k, v in tuner.model.state_dict().items()}
            pre_rng = torch.get_rng_state()

            line = emit_fast_rung_ledger(tuner, rate=0.5, blended_acc=blended)

            assert line is not None and _LEDGER_RE.match(line), line
            post_rates = [
                float(p.base_activation.rate) for p in tuner.model.get_perceptrons()
            ]
            assert post_rates == pre_rates
            assert tuner.adaptation_manager.lif_active is False
            assert "forward" not in tuner.model.__dict__, (
                "the D-hat probe must never patch the live model's forward"
            )
            post_sd = tuner.model.state_dict()
            for key in pre_sd:
                assert torch.equal(pre_sd[key], post_sd[key]), key
            assert torch.equal(torch.get_rng_state(), pre_rng)
        finally:
            tuner.close()

    def test_disabled_emit_is_a_no_op(self, tmp_path, monkeypatch, capsys):
        from mimarsinan.tuning.orchestration.mbh_ledger import emit_fast_rung_ledger

        _set_flag(monkeypatch, False)
        torch.manual_seed(0)
        tuner = _lif_tuner(tmp_path)
        try:
            assert emit_fast_rung_ledger(tuner, rate=0.5, blended_acc=0.5) is None
            assert not _ledger_lines(capsys.readouterr().out)
        finally:
            tuner.close()


# -- (d) X2/E0: gate-grade probe precision — every MBH read is fp32 ---------------

class _BinEdgeModel(nn.Module):
    """Staircase whose bin flips under reduced-precision autocast.

    fp32: F.linear lands at 1 - 2**-10 (just below the floor edge at 1.0);
    bf16 rounds the weight up to 1.0, crossing the edge and flipping the argmax.
    """

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.full((1, 1), 1.0 - 2**-10))

    def forward(self, x):
        h = F.linear(x, self.w)
        stair = torch.floor(h)
        return torch.cat([torch.full_like(stair, 0.5), stair], dim=1)


class _OneBatchTrainer:
    """Minimal trainer surface for the MBH eval helpers: one cached val batch."""

    def __init__(self, x, y):
        self._gpu_val_cache = [(x, y)]
        self._gpu_val_cursor = 0

    def iter_validation_batches(self, n):
        for _ in range(int(n)):
            yield self._gpu_val_cache[self._gpu_val_cursor % len(self._gpu_val_cache)]
            self._gpu_val_cursor += 1


class _StubTuner:
    """Just enough tuner surface for ``rung_measurements``; blend == deploy."""

    def __init__(self, model, trainer):
        self.model = model
        self.trainer = trainer
        self.pipeline = SimpleNamespace(config={"device": "cpu"})
        self._budget = SimpleNamespace(eval_n_batches=1)
        self._mbh_rung_index = -1

    def _mbh_full_transform_forward(self, clone):
        return clone


def _bin_edge_fixture():
    x = torch.ones(4, 1)
    y = torch.zeros(4, dtype=torch.long)
    model = _BinEdgeModel()
    return model, _OneBatchTrainer(x, y)


class TestGateGradeFp32Precision:
    def test_ambient_autocast_flips_the_staircase_bin(self):
        # Control: the X1 skew mechanism — the same eval under autocast flips a bin.
        from mimarsinan.tuning.orchestration.genuine_probe import eval_forward_over_val

        model, trainer = _bin_edge_fixture()
        assert eval_forward_over_val(trainer, model, model, 1, "cpu") == 1.0
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assert eval_forward_over_val(trainer, model, model, 1, "cpu") == 0.0

    def test_fp32_eval_is_ambient_autocast_proof(self):
        from mimarsinan.tuning.orchestration.mbh_ledger import (
            fp32_eval_forward_over_val,
        )

        model, trainer = _bin_edge_fixture()
        assert fp32_eval_forward_over_val(trainer, model, model, 1, "cpu") == 1.0
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assert fp32_eval_forward_over_val(trainer, model, model, 1, "cpu") == 1.0

    def test_mbh_rung_reads_agree_fp32_vs_fp32_under_ambient_autocast(self, monkeypatch):
        # Blended and D-hat must be like-for-like: both fp32, even when the
        # surrounding code runs an autocast region (ledger on: blended measured).
        from mimarsinan.tuning.orchestration.mbh_ledger import rung_measurements

        _set_flag(monkeypatch, True)
        model, trainer = _bin_edge_fixture()
        tuner = _StubTuner(model, trainer)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            m = rung_measurements(tuner)
        assert m["blended_fp32"] == 1.0
        assert m["full_acc"] == 1.0
        assert m["blended_fp32"] == m["full_acc"]

    def test_full_transform_measurement_is_fp32(self):
        from mimarsinan.tuning.orchestration.mbh_ledger import (
            full_transform_measurement,
        )

        model, trainer = _bin_edge_fixture()
        tuner = _StubTuner(model, trainer)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assert full_transform_measurement(tuner) == 1.0
        assert trainer._gpu_val_cursor == 0, "measurement must restore the cursor"


# -- (e) the alignment probe is verbose diagnostics only ---------------------------

class TestAlignmentIsLedgerOnly:
    def test_rho_and_blended_skipped_without_the_flag(self, monkeypatch):
        # The default gate consumes full_acc only; rho costs an extra clone and
        # two backward passes, and blended_fp32 a full extra probe eval, so both
        # run only under the ledger flag (A4 eval consolidation).
        from mimarsinan.tuning.orchestration.mbh_ledger import rung_measurements

        _set_flag(monkeypatch, False)
        model, trainer = _bin_edge_fixture()
        tuner = _StubTuner(model, trainer)
        m = rung_measurements(tuner)
        assert math.isnan(m["rho"]) and math.isnan(m["grad_norm_t"])
        assert math.isnan(m["blended_fp32"])
        assert m["full_acc"] == 1.0

    def test_rho_and_blended_computed_with_the_flag(self, monkeypatch):
        from mimarsinan.tuning.orchestration.mbh_ledger import rung_measurements

        _set_flag(monkeypatch, True)
        model, trainer = _bin_edge_fixture()
        tuner = _StubTuner(model, trainer)
        m = rung_measurements(tuner)
        assert not math.isnan(m["grad_norm_t"])
        assert m["blended_fp32"] == 1.0
