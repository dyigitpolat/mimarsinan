"""A FRESH full pipeline run resets run-scoped MBH ledgers.

The endpoint step ledger and the D-hat high-water are RUN totals; a full
run() over a working directory whose cache carries a killed/previous
attempt's state must not inherit consumption (measured: a stale exhausted
ledger silently demoted a 16k-step floor to the 336-step patience geometry).
Explicit resume (run_from) keeps them — resumption continues the same run.
"""

from __future__ import annotations

from mimarsinan.pipelining.core.engine.pipeline import Pipeline
from mimarsinan.tuning.orchestration import dhat_highwater, endpoint_steps


class _Cache(dict):
    def add(self, key, value, strategy=None):
        self[key] = value

    def remove(self, key):
        self.pop(key, None)


def _bare_pipeline():
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.cache = _Cache()
    return pipeline


class TestRunScopedLedgerReset:
    def test_reset_clears_both_ledgers(self):
        pipeline = _bare_pipeline()
        endpoint_steps.consume(pipeline, 16000)
        dhat_highwater.observe(pipeline, 0.99)
        pipeline._reset_run_scoped_ledgers()
        assert endpoint_steps.consumed(pipeline) == 0
        assert dhat_highwater.peek(pipeline) is None

    def test_run_resets_before_the_first_step(self, monkeypatch):
        pipeline = _bare_pipeline()
        endpoint_steps.consume(pipeline, 16000)
        pipeline.steps = []
        monkeypatch.setattr(pipeline, "set_up_requirements", lambda: None)
        monkeypatch.setattr(pipeline, "verify", lambda: None)
        pipeline.run()
        assert endpoint_steps.consumed(pipeline) == 0

    def test_run_from_preserves_the_ledgers(self, monkeypatch):
        pipeline = _bare_pipeline()
        endpoint_steps.consume(pipeline, 1234)
        dhat_highwater.observe(pipeline, 0.9)
        pipeline.steps = [("only", object())]
        monkeypatch.setattr(pipeline, "set_up_requirements", lambda: None)
        monkeypatch.setattr(pipeline, "verify", lambda: None)
        import mimarsinan.pipelining.core.engine.pipeline as pipeline_mod

        monkeypatch.setattr(
            pipeline_mod.pipeline_resume, "find_starting_step_idx",
            lambda p, name: 1,
        )
        pipeline.run_from("only")
        assert endpoint_steps.consumed(pipeline) == 1234
        assert dhat_highwater.peek(pipeline) == 0.9
