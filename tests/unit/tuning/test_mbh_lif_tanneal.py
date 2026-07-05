"""LIF T-annealing realizable family — the default LIF recipe (MBH X3, from X2b).

Driven by the ConversionPolicy recipe knob ``lif_tanneal`` (folded into the
config for lif mode; knob off keeps the value-blend ramp bit-identical). When on
and the mode is lif (spiking_semantics predicate), the LIF adaptation tuner
replaces the value-blend ladder with a T-annealing ladder: the LIF spiking
activation is installed FULLY from the first rung (blend rate pinned at 1.0 —
no old/target mixture) and the ladder rate instead anneals the node
simulation-step count T (and the encoding ``ChipInputQuantizer`` grid) down a
pow2-snapped geometric schedule that terminates EXACTLY at ``simulation_steps``
(P1' endpoint exactness). Every rung is a genuine deployable LIF network at the
rung's T. Budget, KD loss, optimizer and rung count are the value-blend
recipe's (equal-budget T1 comparison); D-hat stays the FULL target behavior at
target_T; the [MBH] ledger and the default D-hat gate compose unchanged.
"""

from __future__ import annotations

import copy
import re

import pytest
import torch

from conftest import MockPipeline, default_config, make_tiny_supermodel
from mimarsinan.common import env
from mimarsinan.models.nn.activations import ChipInputQuantizer, LIFActivation
from mimarsinan.tuning.orchestration import mbh_ledger
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager

_LEDGER_RE = re.compile(r"^\[MBH\] tuner=\S+ rung=\d+ rate=[0-9.]+ ")


def _lif_tuner(tmp_path, *, tanneal, steps_per_rate=2, rates=(0.25, 0.5, 0.75, 1.0),
               simulation_steps=4, target_metric=0.0):
    from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

    cfg = default_config()
    cfg["spiking_mode"] = "lif"
    cfg["firing_mode"] = "Default"
    cfg["thresholding_mode"] = "<"
    cfg["simulation_steps"] = simulation_steps
    cfg["lif_blend_fast"] = True
    cfg["lif_blend_fast_steps_per_rate"] = steps_per_rate
    cfg["lif_blend_fast_rates"] = list(rates)
    cfg["lif_tanneal"] = bool(tanneal)
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = target_metric
    model = make_tiny_supermodel()
    return LIFAdaptationTuner(
        pipeline, model=model, target_accuracy=0.5, lr=cfg["lr"],
        adaptation_manager=AdaptationManager(),
    )


def _lif_targets(model):
    """The genuine LIFActivation nodes under the installed blends."""
    return [
        p.base_activation.target_activation
        for p in model.get_perceptrons()
    ]


def _input_quantizers(model):
    return [m for m in model.modules() if isinstance(m, ChipInputQuantizer)]


def _inject_accepting_gate(monkeypatch):
    """Never-regressing D-hat measurements: the default gate rides the accept path."""
    monkeypatch.setattr(
        mbh_ledger, "rung_measurements",
        lambda tuner: {
            "blended_fp32": 0.5, "full_acc": 0.5,
            "rho": 1.0, "grad_norm_t": 0.0,
        },
    )
    monkeypatch.setattr(
        mbh_ledger, "full_transform_measurement", lambda tuner: 0.5,
    )


def _run_seeded(tmp_path, monkeypatch, tanneal, **kw):
    _inject_accepting_gate(monkeypatch)
    torch.manual_seed(0)
    tuner = _lif_tuner(tmp_path, tanneal=tanneal, **kw)
    try:
        tuner.run()
    finally:
        tuner.close()
    return tuner


# -- the recipe carries the knob ---------------------------------------------------

class TestRecipeKnob:
    def test_lif_recipe_turns_tanneal_on(self):
        from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy

        assert ConversionPolicy.derive("lif").knobs["lif_tanneal"] is True

    def test_non_lif_recipes_carry_no_tanneal_knob(self):
        from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy

        for mode, schedule in (
            ("ttfs", None),
            ("ttfs_quantized", None),
            ("ttfs_cycle_based", "cascaded"),
            ("ttfs_cycle_based", "synchronized"),
        ):
            assert "lif_tanneal" not in ConversionPolicy.derive(mode, schedule).knobs


# -- schedule math (the rate -> T SSOT) ------------------------------------------

class TestStartRule:
    def test_small_targets_start_at_32(self):
        from mimarsinan.tuning.orchestration.mbh_tanneal import tanneal_start_T

        for target in (1, 4, 8, 16):
            assert tanneal_start_T(target) == 32

    def test_high_targets_start_at_4x(self):
        from mimarsinan.tuning.orchestration.mbh_tanneal import tanneal_start_T

        assert tanneal_start_T(32) == 128
        assert tanneal_start_T(64) == 256

    def test_start_is_never_below_target(self):
        from mimarsinan.tuning.orchestration.mbh_tanneal import tanneal_start_T

        for target in range(1, 130):
            assert tanneal_start_T(target) >= target


class TestSchedule:
    def _schedule(self, target_T, rates=(0.25, 0.5, 0.75, 1.0)):
        from mimarsinan.tuning.orchestration.mbh_tanneal import TAnnealSchedule

        return TAnnealSchedule(target_T=target_T, ladder_rates=tuple(rates))

    def test_t0_01_shape_is_the_halving_ladder(self):
        # target 4 (< 32): start 32, four rungs -> the canonical 32/16/8/4.
        assert self._schedule(4).rung_Ts == (32, 16, 8, 4)

    def test_s32_vehicle_starts_at_4x_target(self):
        # target 32: start 128; pow2 snapping of the geometric interior.
        assert self._schedule(32).rung_Ts == (128, 64, 64, 32)

    @pytest.mark.parametrize("target", [1, 4, 8, 12, 16, 32, 48])
    @pytest.mark.parametrize(
        "rates", [(0.25, 0.5, 0.75, 1.0), (0.5, 1.0), (1.0,)],
    )
    def test_schedule_properties(self, target, rates):
        from mimarsinan.tuning.orchestration.mbh_tanneal import tanneal_start_T

        schedule = self._schedule(target, rates)
        rungs = schedule.rung_Ts
        assert len(rungs) == len(rates)
        assert rungs[-1] == target, "the last rung IS the target behavior"
        assert all(isinstance(T, int) for T in rungs)
        assert all(a >= b for a, b in zip(rungs, rungs[1:])), "monotone decreasing"
        assert all(target <= T <= tanneal_start_T(target) for T in rungs)
        # For pow2 targets (every tier-0 LIF vehicle) the whole ladder is powers
        # of two: bit-exact IFNode<->staircase ties at every rung. Non-pow2
        # targets keep an integer geometric descent (their own tie grid drifts
        # regardless — the target behavior fixes it).
        if target & (target - 1) == 0:
            for T in rungs:
                assert T & (T - 1) == 0, f"rung T={T} is not a power of two"

    def test_endpoint_is_exact_even_for_non_pow2_target(self):
        assert self._schedule(12).rung_Ts[-1] == 12
        assert self._schedule(12).T_for_rate(1.0) == 12

    def test_midpoint_rates_are_monotone_integers_in_range(self):
        schedule = self._schedule(4)
        rs = [0.05 + 0.95 * i / 39 for i in range(40)]
        ts = [schedule.T_for_rate(r) for r in rs]
        assert all(isinstance(t, int) for t in ts)
        assert all(4 <= t <= 32 for t in ts)
        assert all(a >= b for a, b in zip(ts, ts[1:]))

    def test_rates_below_the_first_rung_clamp_to_start(self):
        assert self._schedule(4).T_for_rate(0.0) == 32
        assert self._schedule(4).T_for_rate(0.1) == 32

    def test_gate_midpoint_between_rungs_lands_between_rung_Ts(self):
        schedule = self._schedule(4)
        # the gate's first retry after rejecting 0.5 from committed 0.25
        t = schedule.T_for_rate((0.25 + 0.5) / 2.0)
        assert 16 <= t <= 32

    def test_non_increasing_ladder_rates_fail_loud(self):
        with pytest.raises(ValueError):
            self._schedule(4, rates=(0.5, 0.5, 1.0))

    def test_ladder_must_end_at_full_rate(self):
        with pytest.raises(ValueError):
            self._schedule(4, rates=(0.25, 0.5))

    def test_invalid_target_fails_loud(self):
        with pytest.raises(ValueError):
            self._schedule(0)


class TestDerive:
    def test_knob_off_is_none(self):
        from mimarsinan.tuning.tuners.lif_adaptation_tuner import (
            derive_lif_tanneal_schedule,
        )

        cfg = {"spiking_mode": "lif", "simulation_steps": 4}
        assert derive_lif_tanneal_schedule(cfg, ladder_rates=[0.5, 1.0]) is None
        cfg["lif_tanneal"] = False
        assert derive_lif_tanneal_schedule(cfg, ladder_rates=[0.5, 1.0]) is None

    def test_knob_on_lif_derives_from_simulation_steps(self):
        from mimarsinan.tuning.tuners.lif_adaptation_tuner import (
            derive_lif_tanneal_schedule,
        )

        cfg = {"spiking_mode": "lif", "simulation_steps": 4, "lif_tanneal": True}
        schedule = derive_lif_tanneal_schedule(cfg, ladder_rates=[0.5, 1.0])
        assert schedule is not None
        assert schedule.target_T == 4
        assert schedule.ladder_rates == (0.5, 1.0)

    def test_knob_on_non_lif_modes_are_none(self):
        from mimarsinan.tuning.tuners.lif_adaptation_tuner import (
            derive_lif_tanneal_schedule,
        )

        for mode in ("ttfs", "ttfs_quantized", "ttfs_cycle_based"):
            cfg = {"spiking_mode": mode, "simulation_steps": 4, "lif_tanneal": True}
            assert derive_lif_tanneal_schedule(cfg, ladder_rates=[0.5, 1.0]) is None


# -- every rung is a genuine deployable LIF network -------------------------------

class TestEveryRungRealizable:
    def test_each_ladder_rate_installs_full_lif_at_the_rung_T(self, tmp_path):
        torch.manual_seed(0)
        tuner = _lif_tuner(tmp_path, tanneal=True)
        try:
            expected = dict(zip((0.25, 0.5, 0.75, 1.0), (32, 16, 8, 4)))
            for rate, rung_T in expected.items():
                tuner._set_rate(rate)
                for p in tuner.model.get_perceptrons():
                    assert float(p.base_activation.rate) == 1.0, (
                        "T-anneal rungs must never mix old and target activations"
                    )
                    assert p.base_activation.activation_type == "LIF"
                for lif in _lif_targets(tuner.model):
                    assert isinstance(lif, LIFActivation)
                    assert lif.T == rung_T
                quantizers = _input_quantizers(tuner.model)
                assert quantizers, "the encoding layer must carry a ChipInputQuantizer"
                for q in quantizers:
                    assert q.T == rung_T
        finally:
            tuner.close()

    def test_rung_forward_is_bitwise_the_bare_lif_target(self, tmp_path):
        torch.manual_seed(0)
        tuner = _lif_tuner(tmp_path, tanneal=True)
        try:
            tuner._set_rate(0.5)
            x = torch.linspace(-1.0, 2.0, 23).unsqueeze(0)
            for p in tuner.model.get_perceptrons():
                blend = p.base_activation
                assert torch.equal(blend(x), blend.target_activation(x)), (
                    "the rung member must BE the LIF node, not a mixture"
                )
        finally:
            tuner.close()

    def test_full_rate_lands_exactly_on_target_T(self, tmp_path):
        torch.manual_seed(0)
        tuner = _lif_tuner(tmp_path, tanneal=True)
        try:
            tuner._set_rate(1.0)
            for lif in _lif_targets(tuner.model):
                assert lif.T == 4
            for q in _input_quantizers(tuner.model):
                assert q.T == 4
        finally:
            tuner.close()

    def test_extra_state_roundtrips_blend_rates_and_T(self, tmp_path):
        torch.manual_seed(0)
        tuner = _lif_tuner(tmp_path, tanneal=True)
        try:
            tuner._set_rate(0.5)
            snapshot = tuner._get_extra_state()
            tuner._set_rate(1.0)
            assert _lif_targets(tuner.model)[0].T == 4
            tuner._set_extra_state(snapshot)
            for lif in _lif_targets(tuner.model):
                assert lif.T == 16
            for q in _input_quantizers(tuner.model):
                assert q.T == 16
            for p in tuner.model.get_perceptrons():
                assert float(p.base_activation.rate) == 1.0
        finally:
            tuner.close()


# -- D-hat measures the FULL target behavior at target_T --------------------------

class TestDhatUsesTargetT:
    def test_full_transform_clone_is_at_target_T_live_stays_at_rung_T(
        self, tmp_path,
    ):
        torch.manual_seed(0)
        tuner = _lif_tuner(tmp_path, tanneal=True)
        try:
            tuner._set_rate(0.25)
            clone = copy.deepcopy(tuner.model)
            tuner._mbh_full_transform_forward(clone)
            for lif in _lif_targets(clone):
                assert lif.T == 4, "D-hat must run the target behavior at target_T"
            for q in _input_quantizers(clone):
                assert q.T == 4
            # the live model is untouched: still the rung's family member
            for lif in _lif_targets(tuner.model):
                assert lif.T == 32
            for q in _input_quantizers(tuner.model):
                assert q.T == 32
            assert tuner.adaptation_manager.lif_active is False
        finally:
            tuner.close()


# -- tuner wiring and knob-off bit-identity ---------------------------------------

class TestTunerWiring:
    def test_knob_on_swaps_axis_and_ramp_strategy(self, tmp_path):
        from mimarsinan.tuning.orchestration.mbh_tanneal import (
            LIFTAnnealAxis,
            TAnnealRealizableRamp,
        )

        tuner = _lif_tuner(tmp_path, tanneal=True)
        try:
            assert isinstance(tuner._ramp, TAnnealRealizableRamp)
            assert isinstance(tuner._axis, LIFTAnnealAxis)
        finally:
            tuner.close()

    def test_knob_off_keeps_the_value_blend_recipe(self, tmp_path):
        from mimarsinan.tuning.axes.blend_axis import BlendAxis
        from mimarsinan.tuning.orchestration.mbh_tanneal import LIFTAnnealAxis
        from mimarsinan.tuning.orchestration.ramp_strategy import ValueDomainProxyRamp

        tuner = _lif_tuner(tmp_path, tanneal=False)
        try:
            assert type(tuner._ramp) is ValueDomainProxyRamp
            assert isinstance(tuner._axis, BlendAxis)
            assert not isinstance(tuner._axis, LIFTAnnealAxis)
            tuner._set_rate(0.5)
            rates = {float(p.base_activation.rate) for p in tuner.model.get_perceptrons()}
            assert rates == {0.5}, "knob OFF must ramp the value blend as before"
        finally:
            tuner.close()

    def test_kd_loss_is_the_recipe_loss_in_both_modes(self, tmp_path):
        from mimarsinan.tuning.orchestration.blend_ramp import KDClassificationLoss

        for tanneal, sub in ((False, "off"), (True, "on")):
            tuner = _lif_tuner(tmp_path / sub, tanneal=tanneal)
            try:
                assert isinstance(tuner.trainer.loss_function, KDClassificationLoss)
            finally:
                tuner.close()

    def test_equal_budget_knob_on_vs_off(self, tmp_path, monkeypatch):
        t_off = _run_seeded(tmp_path / "off", monkeypatch, False)
        t_on = _run_seeded(tmp_path / "on", monkeypatch, True)
        assert t_on._fast_optimizer_steps == t_off._fast_optimizer_steps
        assert t_on._fast_optimizer_steps == \
            len(t_on._fixed_ladder_rates) * t_on._fast_steps_per_rate

    def test_knob_on_run_completes_at_target_T(self, tmp_path, monkeypatch, capsys):
        tuner = _run_seeded(tmp_path, monkeypatch, True)
        assert tuner._committed_rate == pytest.approx(1.0)
        for lif in _lif_targets(tuner.model):
            assert lif.T == 4
        for q in _input_quantizers(tuner.model):
            assert q.T == 4
        assert [e["outcome"] for e in tuner._cycle_log] == \
            ["commit"] * len(tuner._fixed_ladder_rates)
        out = capsys.readouterr().out
        assert not [l for l in out.splitlines() if l.startswith("[MBH] ")], (
            "no [MBH] ledger lines without the ledger flag"
        )


# -- composition with the [MBH] ledger and the default D-hat gate -------------------

class TestLedgerComposition:
    def test_one_ledger_line_per_rung(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv(env.MBH_LEDGER_VAR, "1")
        _inject_accepting_gate(monkeypatch)
        torch.manual_seed(0)
        tuner = _lif_tuner(tmp_path, tanneal=True, rates=(0.5, 1.0))
        try:
            tuner.run()
        finally:
            tuner.close()
        lines = [
            l for l in capsys.readouterr().out.splitlines()
            if l.startswith("[MBH] ")
        ]
        assert len(lines) == len(tuner._fixed_ladder_rates)
        for line in lines:
            assert _LEDGER_RE.match(line), line


class TestGateComposition:
    def _inject_measurements(self, monkeypatch, *, entry, full_accs):
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
            mbh_ledger, "full_transform_measurement", lambda tuner: float(entry),
        )

    def test_accepting_gate_walks_the_T_ladder(self, tmp_path, monkeypatch, capsys):
        self._inject_measurements(monkeypatch, entry=0.1, full_accs=[0.2, 0.3])
        torch.manual_seed(0)
        tuner = _lif_tuner(tmp_path, tanneal=True, rates=(0.5, 1.0))
        try:
            tuner.run()
            assert tuner._committed_rate == pytest.approx(1.0)
            for lif in _lif_targets(tuner.model):
                assert lif.T == 4
        finally:
            tuner.close()
        out = capsys.readouterr().out
        accepts = [l for l in out.splitlines() if "[MBH-GATE]" in l and " accept " in l]
        assert len(accepts) == 2

    def test_rejecting_gate_stalls_and_still_finalizes_at_target_T(
        self, tmp_path, monkeypatch, capsys,
    ):
        self._inject_measurements(monkeypatch, entry=0.9, full_accs=[0.1])
        torch.manual_seed(0)
        tuner = _lif_tuner(tmp_path, tanneal=True, rates=(0.5, 1.0))
        try:
            with pytest.warns(UserWarning, match="natural adaptation reached only"):
                tuner.run()
            assert tuner._committed_rate == pytest.approx(1.0)
            assert tuner._mbh_gate_state.stalled is True
            # rejected rungs rolled everything back; finalize still ships target_T
            assert tuner._fast_optimizer_steps == 0
            for lif in _lif_targets(tuner.model):
                assert lif.T == 4
            for q in _input_quantizers(tuner.model):
                assert q.T == 4
        finally:
            tuner.close()
        out = capsys.readouterr().out
        assert any("constructive_stall" in l for l in out.splitlines())
