"""[MBH-DRAWS] best-of-N conversion draws: independent RNG streams, D-hat-selected.

The mixer conversion quality is a high-variance distribution whose upper tail
crosses the acceptance bar; the harness samples N independent draws (torch RNG
streams seed+k — the whole search is deterministic given the config seed) and
keeps the artifact with the best full-transform fp32 D-hat. N=1 is bit-identical
to today's single run; each draw is independently keep-best/entry-floored inside
the tuner, so selection can only improve D-hat.
"""

from __future__ import annotations

import pytest
import torch

from mimarsinan.tuning.orchestration import conversion_draws as draws_mod
from mimarsinan.tuning.orchestration import dhat_highwater, endpoint_steps
from mimarsinan.tuning.orchestration.conversion_draws import (
    configured_draws,
    run_conversion_draws,
)


class _Reporter:
    def __init__(self):
        self.reports = []

    def report(self, name, value):
        self.reports.append((name, value))


class _Pipeline:
    def __init__(self, config=None):
        self.config = {"seed": 7}
        self.config.update(config or {})
        self.cache = {}
        self.reporter = _Reporter()


class _Model:
    def __init__(self):
        self.shared = []
        self.trained_by = None


class _Manager:
    def __init__(self, model):
        self.shared = model.shared  # shared reference: must survive the copy


class _FakeTuner:
    def __init__(self, model, manager, *, on_run=None):
        self.model = model
        self.manager = manager
        self.on_run = on_run
        self.ran = False
        self.closed = False

    def run(self):
        self.ran = True
        if self.on_run is not None:
            self.on_run(self)

    def close(self):
        self.closed = True


def _spy_seeds(monkeypatch):
    seeds = []
    monkeypatch.setattr(torch, "manual_seed", lambda s: seeds.append(int(s)))
    return seeds


def _script_reads(monkeypatch, values):
    reads = iter(values)
    calls = []

    def fake_read(tuner):
        calls.append(tuner)
        return next(reads)

    monkeypatch.setattr(draws_mod, "fp32_deployed_read", fake_read)
    return calls


class TestConfiguredDraws:
    def test_default_is_one(self):
        assert configured_draws(_Pipeline()) == 1

    def test_reads_the_config_knob(self):
        assert configured_draws(_Pipeline({"conversion_draws": 3})) == 3

    def test_floors_at_one(self):
        assert configured_draws(_Pipeline({"conversion_draws": 0})) == 1


class TestSingleDrawBitIdentity:
    def test_no_reseed_no_copy_no_eval(self, monkeypatch):
        pipeline = _Pipeline()
        seeds = _spy_seeds(monkeypatch)
        reads = _script_reads(monkeypatch, [])
        model, manager = _Model(), object()
        built = []

        def build(m, am):
            built.append((m, am))
            return _FakeTuner(m, am)

        tuner, out_model, out_manager = run_conversion_draws(
            pipeline, build, model, manager,
        )
        assert built == [(model, manager)], "the entry objects run in place"
        assert out_model is model and out_manager is manager
        assert tuner.ran is True and tuner.closed is False
        assert seeds == [], "N=1 must not touch the RNG stream"
        assert reads == [], "N=1 must not add eval reads"
        assert pipeline.cache == {}

    def test_single_draw_exceptions_propagate(self, monkeypatch):
        pipeline = _Pipeline()

        def build(m, am):
            return _FakeTuner(m, am, on_run=_raise)

        def _raise(tuner):
            raise RuntimeError("honest gate abort")

        with pytest.raises(RuntimeError, match="honest gate abort"):
            run_conversion_draws(pipeline, build, _Model(), object())


class TestBestOfNSelection:
    def test_selects_the_max_dhat_draw(self, monkeypatch):
        pipeline = _Pipeline({"conversion_draws": 3})
        _script_reads(monkeypatch, [0.5, 0.9, 0.7])
        model, manager = _Model(), object()
        tuners = []

        def build(m, am):
            tuner = _FakeTuner(m, am)
            tuners.append(tuner)
            return tuner

        tuner, out_model, out_manager = run_conversion_draws(
            pipeline, build, model, manager,
        )
        assert tuner is tuners[1]
        assert out_model is tuners[1].model
        assert out_model is not model, "the winner is its draw's own copy"
        assert tuners[0].closed and tuners[2].closed
        assert tuner.closed is False

    def test_tie_keeps_the_first_draw(self, monkeypatch):
        pipeline = _Pipeline({"conversion_draws": 2})
        _script_reads(monkeypatch, [0.9, 0.9])
        tuners = []

        def build(m, am):
            tuner = _FakeTuner(m, am)
            tuners.append(tuner)
            return tuner

        tuner, _, _ = run_conversion_draws(pipeline, build, _Model(), object())
        assert tuner is tuners[0], "deterministic tie-break: lowest k wins"

    def test_target_reached_skips_remaining_draws(self, monkeypatch, capsys):
        # Best-of-N is a fallback for the crater distribution, not a mandate:
        # a lossless first draw ends the search (healthy cells pay one draw).
        pipeline = _Pipeline({"conversion_draws": 3})
        reads = _script_reads(monkeypatch, [0.98])
        tuners = []

        def build(m, am):
            tuner = _FakeTuner(m, am)
            tuners.append(tuner)
            return tuner

        tuner, _, _ = run_conversion_draws(
            pipeline, build, _Model(), object(), target=0.97,
        )
        assert len(tuners) == 1
        assert tuner is tuners[0]
        assert len(reads) == 1
        assert "skipping remaining draws" in capsys.readouterr().out

    def test_below_target_draws_run_out_the_budget(self, monkeypatch):
        pipeline = _Pipeline({"conversion_draws": 3})
        _script_reads(monkeypatch, [0.5, 0.9, 0.7])
        tuners = []

        def build(m, am):
            tuner = _FakeTuner(m, am)
            tuners.append(tuner)
            return tuner

        tuner, _, _ = run_conversion_draws(
            pipeline, build, _Model(), object(), target=0.97,
        )
        assert len(tuners) == 3
        assert tuner is tuners[1]

    def test_mid_search_target_reach_keeps_the_reaching_draw(self, monkeypatch):
        pipeline = _Pipeline({"conversion_draws": 3})
        _script_reads(monkeypatch, [0.5, 0.98])
        tuners = []

        def build(m, am):
            tuner = _FakeTuner(m, am)
            tuners.append(tuner)
            return tuner

        tuner, _, _ = run_conversion_draws(
            pipeline, build, _Model(), object(), target=0.97,
        )
        assert len(tuners) == 2
        assert tuner is tuners[1]

    def test_draw_seeds_are_seed_plus_k(self, monkeypatch):
        pipeline = _Pipeline({"conversion_draws": 3})
        seeds = _spy_seeds(monkeypatch)
        _script_reads(monkeypatch, [0.1, 0.2, 0.3])
        run_conversion_draws(
            pipeline, lambda m, am: _FakeTuner(m, am), _Model(), object(),
        )
        assert seeds == [7, 8, 9]

    def test_model_manager_shared_references_survive_the_copy(self, monkeypatch):
        pipeline = _Pipeline({"conversion_draws": 2})
        _script_reads(monkeypatch, [0.2, 0.1])
        model = _Model()
        manager = _Manager(model)

        tuner, out_model, out_manager = run_conversion_draws(
            pipeline, lambda m, am: _FakeTuner(m, am), model, manager,
        )
        assert out_model is not model
        assert out_model.shared is out_manager.shared, (
            "the model<->manager object graph must be copied as ONE graph"
        )

    def test_ledger_lines_carry_dhat_and_kept(self, monkeypatch, capsys):
        pipeline = _Pipeline({"conversion_draws": 3})
        _script_reads(monkeypatch, [0.5, 0.9, 0.7])
        run_conversion_draws(
            pipeline, lambda m, am: _FakeTuner(m, am), _Model(), object(),
        )
        out = capsys.readouterr().out
        assert "[MBH-DRAWS] k=0 full_acc=0.500000 kept=True" in out
        assert "[MBH-DRAWS] k=1 full_acc=0.900000 kept=True" in out
        assert "[MBH-DRAWS] k=2 full_acc=0.700000 kept=False" in out
        assert "[MBH-DRAWS] selected k=1 full_acc=0.900000 n=3" in out
        assert (
            "conversion_draws_selected",
            {"k": 1, "full_acc": 0.9, "n": 3},
        ) in pipeline.reporter.reports


class TestRunLedgerScopeIsolation:
    def test_each_draw_starts_from_the_entry_scope_and_best_scope_persists(
        self, monkeypatch,
    ):
        pipeline = _Pipeline({"conversion_draws": 3})
        dhat_highwater.observe(pipeline, 0.5)
        endpoint_steps.consume(pipeline, 100)
        _script_reads(monkeypatch, [0.6, 0.9, 0.7])
        entries = []

        def on_run(tuner):
            entries.append((
                dhat_highwater.peek(pipeline), endpoint_steps.consumed(pipeline),
            ))
            k = len(entries) - 1
            dhat_highwater.observe(pipeline, 0.9 + 0.01 * k)
            endpoint_steps.consume(pipeline, 1000 * (k + 1))

        run_conversion_draws(
            pipeline, lambda m, am: _FakeTuner(m, am, on_run=on_run),
            _Model(), object(),
        )
        assert entries == [(0.5, 100)] * 3, "every draw enters the same scope"
        # The kept draw is k=1: its post-run scope is what persists.
        assert dhat_highwater.peek(pipeline) == pytest.approx(0.91)
        assert endpoint_steps.consumed(pipeline) == 100 + 2000


class TestDrawFragilityCoverage:
    def test_failed_draw_is_a_measured_outcome(self, monkeypatch, capsys):
        pipeline = _Pipeline({"conversion_draws": 3})
        _script_reads(monkeypatch, [0.4, 0.3])
        tuners = []

        def build(m, am):
            def on_run(tuner):
                if len(tuners) == 1:
                    raise AssertionError(
                        "step failed to retain performance within tolerable limits"
                    )

            tuner = _FakeTuner(m, am, on_run=on_run)
            tuners.append(tuner)
            return tuner

        tuner, _, _ = run_conversion_draws(pipeline, build, _Model(), object())
        assert tuner is tuners[1]
        assert tuners[0].closed is True, "a failed draw's workers are released"
        out = capsys.readouterr().out
        assert "[MBH-DRAWS] k=0 full_acc=nan kept=False error=AssertionError" in out

    def test_all_draws_failing_raises_the_last_error(self, monkeypatch):
        pipeline = _Pipeline({"conversion_draws": 2})
        _script_reads(monkeypatch, [])
        tuners = []

        def build(m, am):
            def on_run(tuner):
                raise RuntimeError(f"draw {len(tuners) - 1} failed")

            tuner = _FakeTuner(m, am, on_run=on_run)
            tuners.append(tuner)
            return tuner

        with pytest.raises(RuntimeError, match="draw 1 failed"):
            run_conversion_draws(pipeline, build, _Model(), object())
        assert all(t.closed for t in tuners)


class TestStepWiring:
    """TunerPipelineStep.run_tuner routes through the harness; only
    DRAW_SELECTED steps read the knob."""

    def _pipeline(self, config=None):
        class _StepPipeline(_Pipeline):
            def __init__(self, cfg):
                super().__init__(cfg)
                self.config.setdefault("lr", 0.001)
                self.committed = {}

            def get_target_metric(self):
                return 0.5

            def update_entry(self, step, key, value, strategy):
                self.committed[key] = value

        return _StepPipeline(config)

    def _step_cls(self, selected):
        from mimarsinan.pipelining.core.steps.tuner_pipeline_step import (
            TunerPipelineStep,
        )

        class _Step(TunerPipelineStep):
            DRAW_SELECTED = selected

        return _Step

    def _tuner_cls(self):
        class _Tuner(_FakeTuner):
            def __init__(self, pipeline, *, model, target_accuracy, lr,
                         adaptation_manager, **kwargs):
                super().__init__(model, adaptation_manager)
                self.kwargs = kwargs

        return _Tuner

    def test_selected_step_runs_the_configured_draws(self, monkeypatch):
        pipeline = self._pipeline({"conversion_draws": 2})
        _script_reads(monkeypatch, [0.4, 0.8])
        step = self._step_cls(True)([], [], ["model", "adaptation_manager"], [], pipeline)
        step._updated_entries = set()
        model, manager = _Model(), object()
        step.run_tuner(self._tuner_cls(), model, manager, extra_knob=5)
        assert step.tuner.kwargs == {"extra_knob": 5}
        assert pipeline.committed["model"] is step.tuner.model
        assert pipeline.committed["model"] is not model, (
            "the committed artifact is the best draw's copy"
        )

    def test_unselected_step_ignores_the_knob(self, monkeypatch):
        pipeline = self._pipeline({"conversion_draws": 3})
        reads = _script_reads(monkeypatch, [])
        step = self._step_cls(False)([], [], ["model", "adaptation_manager"], [], pipeline)
        step._updated_entries = set()
        model, manager = _Model(), object()
        step.run_tuner(self._tuner_cls(), model, manager)
        assert reads == []
        assert pipeline.committed["model"] is model
        assert step.tuner.model is model

    def test_conversion_stages_are_draw_selected(self):
        from mimarsinan.pipelining.pipeline_steps.adaptation.lif_adaptation_step import (
            LIFAdaptationStep,
        )
        from mimarsinan.pipelining.pipeline_steps.adaptation.ttfs_cycle_adaptation_step import (
            TTFSCycleAdaptationStep,
        )
        from mimarsinan.pipelining.pipeline_steps.quantization.activation_quantization_step import (
            ActivationQuantizationStep,
        )
        from mimarsinan.pipelining.pipeline_steps.quantization.weight_quantization_step import (
            WeightQuantizationStep,
        )

        assert LIFAdaptationStep.DRAW_SELECTED is True
        assert TTFSCycleAdaptationStep.DRAW_SELECTED is True
        assert ActivationQuantizationStep.DRAW_SELECTED is True
        # NAPQ's projection is deterministic and its 16k endpoint is the run's
        # dominant training cost: it stays single-draw by design.
        assert WeightQuantizationStep.DRAW_SELECTED is False
