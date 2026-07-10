"""[5v B1(iii)] hop-staged sync AQ install — the P4 frontier generalized below segments.

A monolithic coarse install on a deep unfragmented chain compounds per-hop
starvation (t0_21: 9 hops, prefix D-hat ~0.10 by block 2 while singles read
0.38-0.93); recovery decays with the same depth. When the A6 gauge FAILS at
the install grid AND the chain is deeper than the proven-recovery depth, the
AQ ladder walks the hop frontier instead: rung i installs the exact ceil
kernel on hops 0..k(i) at full rate (float beyond the frontier), with a
keep-best DFQ re-affine at each frontier step (T4 arm-B at hop granularity).
Everything else (knob off, shallow chains, passing gauges, other modes) stays
bit-identical to the monolithic install.
"""

from __future__ import annotations

import pytest
import torch

from conftest import MockPipeline, default_config, make_tiny_supermodel
from mimarsinan.models.nn.decorators.clamp_quantize import TTFSCeilStaircaseDecorator
from mimarsinan.tuning.orchestration.adaptation_manager import (
    HOP_DEPTH_ATTR,
    AdaptationManager,
    hop_frontier,
)
from mimarsinan.tuning.orchestration.frontier import hop_staging, reaffine
from mimarsinan.tuning.orchestration.frontier import frontier_ladder
from mimarsinan.tuning.orchestration.frontier.hop_staging import (
    HOP_STAGE_MIN_LEVELS,
    resolve_sync_hop_staging,
    stamp_hop_depths,
)
from mimarsinan.tuning.perceptron_rate import rebuild_activations


def _sync_cfg(steps=8, *, staged=True):
    cfg = default_config()
    cfg["spiking_mode"] = "ttfs_cycle_based"
    cfg["ttfs_cycle_schedule"] = "synchronized"
    cfg["activation_quantization"] = True
    cfg["sync_exact_qat"] = True
    cfg["optimization_driver"] = "fast"
    cfg["simulation_steps"] = steps
    cfg["target_tq"] = steps
    if staged:
        cfg["sync_hop_staged_install"] = True
    return cfg


def _deep_model():
    # p1 + 4 hidden + p2 = 6 perceptrons in one chain -> 6 hop levels.
    return make_tiny_supermodel(hidden_layers=5)


def _has_ceil_kernel(perceptron) -> bool:
    from mimarsinan.models.nn.decorators.adjustment import iter_activation_tree

    return any(
        isinstance(obj, TTFSCeilStaircaseDecorator)
        for obj in iter_activation_tree(perceptron.activation)
    )


class TestFrontierMath:
    def test_ceil_semantics_match_the_prefix_precedent(self):
        # ceil is load-bearing: gate midpoint retries between rungs must not
        # round the frontier DOWN to an already-accepted stage.
        assert hop_frontier(0.0, 9) == 0
        assert hop_frontier(1 / 9, 9) == 1
        assert hop_frontier(0.5, 9) == 5
        assert hop_frontier(0.51, 9) == 5
        assert hop_frontier(1.0, 9) == 9
        assert hop_frontier(1.2, 9) == 9

    def test_frontier_ladder_walks_every_hop_level(self):
        rates = frontier_ladder(6)
        assert rates == [i / 6 for i in range(1, 7)]


class TestStampHopDepths:
    def test_stamps_every_perceptron_and_returns_levels(self):
        model = _deep_model()
        n = stamp_hop_depths(model)
        depths = [getattr(p, HOP_DEPTH_ATTR) for p in model.get_perceptrons()]
        assert n == max(depths) + 1
        assert sorted(depths) == depths, "the tiny chain is depth-ordered"
        assert n == 6


class TestStagedDecoratorInstall:
    def _manager(self, model, cfg, *, levels, rate):
        manager = AdaptationManager()
        for p in model.get_perceptrons():
            p.base_activation = p.base_activation
        stamp_hop_depths(model)
        manager.quantization_hop_levels = levels
        manager.quantization_rate = rate
        rebuild_activations(model, manager, cfg)
        return manager

    def test_partial_rate_installs_only_the_frontier_prefix(self):
        model = _deep_model()
        cfg = _sync_cfg()
        self._manager(model, cfg, levels=6, rate=0.5)  # k = 3
        installed = [_has_ceil_kernel(p) for p in model.get_perceptrons()]
        depths = [getattr(p, HOP_DEPTH_ATTR) for p in model.get_perceptrons()]
        for has, depth in zip(installed, depths):
            assert has == (depth < 3), f"depth {depth}"

    def test_full_rate_installs_every_hop(self):
        model = _deep_model()
        cfg = _sync_cfg()
        self._manager(model, cfg, levels=6, rate=1.0)
        assert all(_has_ceil_kernel(p) for p in model.get_perceptrons())

    def test_unstaged_manager_is_bit_identical_monolithic(self):
        model = _deep_model()
        cfg = _sync_cfg(staged=False)
        manager = AdaptationManager()
        manager.quantization_rate = 0.5
        rebuild_activations(model, manager, cfg)
        assert all(_has_ceil_kernel(p) for p in model.get_perceptrons()), (
            "the monolithic install decorates every perceptron at the blend rate"
        )

    def test_staged_hops_are_marked_sync_exact_even_beyond_frontier(self):
        # The marker is a per-MODEL decision (the endpoint is exact for every
        # hop by rate 1.0); partial marking would trip the all-or-none assert.
        from mimarsinan.tuning.orchestration.adaptation_manager import (
            model_trained_sync_exact,
        )

        model = _deep_model()
        cfg = _sync_cfg()
        self._manager(model, cfg, levels=6, rate=0.5)
        assert model_trained_sync_exact(model) is True


class TestResolveArming:
    def _tuner_stub(self, tmp_path, cfg, model):
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        pipeline._target_metric = 0.0
        from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
            create_adaptation_manager_for_model,
        )
        from mimarsinan.tuning.tuners.activation_quantization_tuner import (
            ActivationQuantizationTuner,
        )

        manager = create_adaptation_manager_for_model(cfg, model)
        return ActivationQuantizationTuner(
            pipeline, model, cfg["target_tq"], 0.5, cfg["lr"], manager,
        )

    def test_arms_on_deep_chain_with_failing_gauge(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            hop_staging, "_install_gauge_fails", lambda tuner, levels: True,
        )
        tuner = self._tuner_stub(tmp_path, _sync_cfg(), _deep_model())
        try:
            assert tuner._hop_stage_levels == 6
            assert tuner.adaptation_manager.quantization_hop_levels == 6
            assert tuner._fixed_ladder_rates == pytest.approx(
                [i / 6 for i in range(1, 7)]
            )
        finally:
            tuner.close()

    def test_passing_gauge_stays_monolithic(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            hop_staging, "_install_gauge_fails", lambda tuner, levels: False,
        )
        tuner = self._tuner_stub(tmp_path, _sync_cfg(), _deep_model())
        try:
            assert tuner._hop_stage_levels is None
            assert tuner.adaptation_manager.quantization_hop_levels is None
            assert tuner._fixed_ladder_rates == [0.25, 0.5, 0.75, 1.0]
        finally:
            tuner.close()

    def test_shallow_chain_stays_monolithic(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            hop_staging, "_install_gauge_fails", lambda tuner, levels: True,
        )
        tuner = self._tuner_stub(tmp_path, _sync_cfg(), make_tiny_supermodel())
        try:
            assert tuner._hop_stage_levels is None
        finally:
            tuner.close()

    def test_knob_off_stays_monolithic(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            hop_staging, "_install_gauge_fails", lambda tuner, levels: True,
        )
        tuner = self._tuner_stub(tmp_path, _sync_cfg(staged=False), _deep_model())
        try:
            assert tuner._hop_stage_levels is None
        finally:
            tuner.close()

    def test_min_levels_constant_is_above_the_proven_recovery_depth(self):
        # Recovery from an equally-deep entry crater is MEASURED full at
        # L <= 4-5 (t0_22/t0_18/t0_03); staging must not touch those cells.
        assert HOP_STAGE_MIN_LEVELS == 6


class TestHopStageReaffine:
    def test_reaffine_runs_keepbest_dfq_and_reports(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            hop_staging, "_install_gauge_fails", lambda tuner, levels: True,
        )
        tuner = TestResolveArming()._tuner_stub(tmp_path, _sync_cfg(), _deep_model())
        try:
            probe_values = iter([0.5, 0.6, 0.55, 0.55, 0.55, 0.55, 0.55])
            monkeypatch.setattr(
                reaffine, "live_model_acc_fp32",
                lambda t: next(probe_values, 0.55),
            )
            stats = hop_staging.run_hop_stage_reaffine(tuner, 0.5)
            assert stats is not None
            assert stats["probe_best"] == pytest.approx(0.6)
            out = capsys.readouterr().out
            assert "[MBH-HOP]" in out
        finally:
            tuner.close()

    def test_fast_ramp_reaffines_before_training(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            hop_staging, "_install_gauge_fails", lambda tuner, levels: True,
        )
        tuner = TestResolveArming()._tuner_stub(tmp_path, _sync_cfg(), _deep_model())
        try:
            calls = []
            import mimarsinan.tuning.tuners.activation_quantization_tuner as aq_mod

            monkeypatch.setattr(
                aq_mod, "run_hop_stage_reaffine",
                lambda t, r: calls.append(("reaffine", r)),
            )
            monkeypatch.setattr(
                type(tuner).__mro__[1], "_fast_ramp",
                lambda self, rate: calls.append(("train", rate)),
            )
            tuner._fast_ramp(0.5)
            assert calls == [("reaffine", 0.5), ("train", 0.5)]
        finally:
            tuner.close()


class TestStagedHalfStepDeferral:
    """[fbb1 finding] the entry fold assumes the ceil kernel: applied at init it
    poisons the k-hybrid's FLOAT suffix (t0_21: live k=1 read 0.25, the rung's
    training dragged D-hat 0.68 -> 0.54, the gate refused every staged rung).
    Under hop staging the fold defers to the conversion endpoint, where the
    kernel is fully installed and the fold is literally the QAT's entry bias."""

    def _tuner(self, tmp_path, monkeypatch, *, gauge_fails):
        monkeypatch.setattr(
            hop_staging, "_install_gauge_fails",
            lambda tuner, levels: gauge_fails,
        )
        cfg = _sync_cfg()
        cfg["sync_entry_half_step"] = True
        from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
            create_adaptation_manager_for_model,
        )
        from mimarsinan.tuning.tuners.activation_quantization_tuner import (
            ActivationQuantizationTuner,
        )

        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        pipeline._target_metric = 0.0
        model = _deep_model()
        manager = create_adaptation_manager_for_model(cfg, model)
        return ActivationQuantizationTuner(
            pipeline, model, cfg["target_tq"], 0.5, cfg["lr"], manager,
        )

    def test_staged_run_defers_the_fold_past_init(self, tmp_path, monkeypatch):
        tuner = self._tuner(tmp_path, monkeypatch, gauge_fails=True)
        try:
            assert tuner._hop_stage_levels == 6
            assert not any(
                getattr(p, "_sync_entry_half_step_folded", False)
                for p in tuner.model.get_perceptrons()
            ), "the float suffix must never carry the kernel's half-step"
        finally:
            tuner.close()

    def test_deferred_fold_lands_at_the_conversion_endpoint(
        self, tmp_path, monkeypatch,
    ):
        import mimarsinan.tuning.tuners.activation_quantization_tuner as aq_mod

        tuner = self._tuner(tmp_path, monkeypatch, gauge_fails=True)
        try:
            monkeypatch.setattr(
                aq_mod, "run_endpoint_recovery",
                lambda t, *, base_steps: None,
            )
            tuner._post_stabilization_hook()
            perceptrons = list(tuner.model.get_perceptrons())
            # [E1] the subsumed encoder is a ceil-staircase hop: the deferred
            # fold covers it exactly like every on-chip core.
            assert any(getattr(p, "is_encoding_layer", False) for p in perceptrons)
            assert all(
                getattr(p, "_sync_entry_half_step_folded", False)
                for p in perceptrons
            )
        finally:
            tuner.close()

    def test_monolithic_run_still_folds_at_init(self, tmp_path, monkeypatch):
        tuner = self._tuner(tmp_path, monkeypatch, gauge_fails=False)
        try:
            assert tuner._hop_stage_levels is None
            perceptrons = list(tuner.model.get_perceptrons())
            assert any(getattr(p, "is_encoding_layer", False) for p in perceptrons)
            assert all(
                getattr(p, "_sync_entry_half_step_folded", False)
                for p in perceptrons
            )
        finally:
            tuner.close()
