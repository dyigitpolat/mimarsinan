"""Tests for the Pipeline engine: verification, key translation, step execution."""

import pytest
import os

from mimarsinan.pipelining.pipeline import Pipeline
from mimarsinan.pipelining.pipeline_step import PipelineStep


# ---------------------------------------------------------------------------
# Minimal concrete steps for testing the engine
# ---------------------------------------------------------------------------

class ProducerStep(PipelineStep):
    """Promises 'data' with a fixed value."""
    def __init__(self, pipeline, value=42):
        super().__init__(requires=[], promises=["data"], updates=[], clears=[], pipeline=pipeline)
        self._value = value

    def process(self):
        self.add_entry("data", self._value)

    def validate(self):
        return 1.0


class ConsumerStep(PipelineStep):
    """Requires 'data' and promises 'result'."""
    def __init__(self, pipeline):
        super().__init__(requires=["data"], promises=["result"], updates=[], clears=[], pipeline=pipeline)
        self.consumed = None

    def process(self):
        self.consumed = self.get_entry("data")
        self.add_entry("result", self.consumed * 2)

    def validate(self):
        return 1.0


class UpdaterStep(PipelineStep):
    """Reads and updates 'data'."""
    def __init__(self, pipeline):
        super().__init__(requires=["data"], promises=[], updates=["data"], clears=[], pipeline=pipeline)

    def process(self):
        old = self.get_entry("data")
        self.update_entry("data", old + 100)

    def validate(self):
        return 1.0


class ClearerStep(PipelineStep):
    """Requires and clears 'data'."""
    def __init__(self, pipeline):
        super().__init__(requires=["data"], promises=[], updates=[], clears=["data"], pipeline=pipeline)

    def process(self):
        _ = self.get_entry("data")

    def validate(self):
        return 1.0


class BrokenPromiseStep(PipelineStep):
    """Claims to promise 'x' but never adds it."""
    def __init__(self, pipeline):
        super().__init__(requires=[], promises=["x"], updates=[], clears=[], pipeline=pipeline)

    def process(self):
        pass

    def validate(self):
        return 1.0


class UnreadRequirementStep(PipelineStep):
    """Requires 'data' but never reads it."""
    def __init__(self, pipeline):
        super().__init__(requires=["data"], promises=[], updates=[], clears=[], pipeline=pipeline)

    def process(self):
        pass

    def validate(self):
        return 1.0


class PerformanceDropStep(PipelineStep):
    """Returns a low metric to trigger the tolerance check."""
    def __init__(self, pipeline, metric=0.01):
        super().__init__(requires=[], promises=[], updates=[], clears=[], pipeline=pipeline)
        self._metric = metric

    def process(self):
        pass

    def validate(self):
        return self._metric


class CleanupTrackingStep(PipelineStep):
    """Tracks that cleanup() was called; used to test pipeline cleanup contract."""
    def __init__(self, pipeline, *, promises_data=True, metric=1.0):
        promises = ["data"] if promises_data else []
        super().__init__(requires=[], promises=promises, updates=[], clears=[], pipeline=pipeline)
        self.cleanup_called = []
        self._metric = metric

    def process(self):
        if self.promises:
            self.add_entry("data", 1)

    def validate(self):
        return self._metric

    def cleanup(self):
        self.cleanup_called.append(True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineVerification:
    def test_valid_chain_passes(self, tmp_path):
        p = Pipeline(str(tmp_path / "cache"))
        p.add_pipeline_step("produce", ProducerStep(p))
        p.add_pipeline_step("consume", ConsumerStep(p))

    def test_missing_requirement_fails(self, tmp_path):
        p = Pipeline(str(tmp_path / "cache"))
        with pytest.raises(AssertionError, match="requires"):
            p.add_pipeline_step("consume", ConsumerStep(p))

    def test_duplicate_promise_accepted(self, tmp_path):
        """Two steps promising different keys should be fine."""
        p = Pipeline(str(tmp_path / "cache"))
        p.add_pipeline_step("p1", ProducerStep(p, value=1))
        p.add_pipeline_step("p2", ProducerStep(p, value=2))


class TestKeyTranslation:
    def test_consumer_reads_producer_key(self, tmp_path):
        p = Pipeline(str(tmp_path / "cache"))
        p.add_pipeline_step("produce", ProducerStep(p, value=7))
        c = ConsumerStep(p)
        p.add_pipeline_step("consume", c)
        p.set_up_requirements()

        assert p.key_translations["consume"]["data"] == "produce.data"

    def test_updater_reads_from_producer(self, tmp_path):
        p = Pipeline(str(tmp_path / "cache"))
        prod = ProducerStep(p, value=5)
        upd = UpdaterStep(p)
        p.add_pipeline_step("produce", prod)
        p.add_pipeline_step("update", upd)
        p.set_up_requirements()

        assert p.key_translations["update"]["data"] == "produce.data"


class TestPipelineRun:
    def test_simple_produce_consume(self, tmp_path):
        p = Pipeline(str(tmp_path / "cache"))
        prod = ProducerStep(p, value=10)
        cons = ConsumerStep(p)
        p.add_pipeline_step("produce", prod)
        p.add_pipeline_step("consume", cons)
        p.run()

        assert cons.consumed == 10
        assert p.cache["consume.result"] == 20

    def test_update_replaces_old_key(self, tmp_path):
        p = Pipeline(str(tmp_path / "cache"))
        prod = ProducerStep(p, value=5)
        upd = UpdaterStep(p)
        p.add_pipeline_step("produce", prod)
        p.add_pipeline_step("update", upd)
        p.run()

        assert "produce.data" not in p.cache
        assert p.cache["update.data"] == 105

    def test_broken_promise_raises(self, tmp_path):
        p = Pipeline(str(tmp_path / "cache"))
        p.add_pipeline_step("broken", BrokenPromiseStep(p))
        with pytest.raises(AssertionError, match="promised"):
            p.run()

    def test_unread_requirement_raises(self, tmp_path):
        p = Pipeline(str(tmp_path / "cache"))
        prod = ProducerStep(p, value=1)
        unread = UnreadRequirementStep(p)
        p.add_pipeline_step("produce", prod)
        p.add_pipeline_step("unread", unread)
        with pytest.raises(AssertionError, match="required entries"):
            p.run()

    def test_performance_tolerance_failure(self, tmp_path):
        p = Pipeline(str(tmp_path / "cache"))
        p.tolerance = 0.95
        good = ProducerStep(p, value=1)
        bad = PerformanceDropStep(p, metric=0.01)
        p.add_pipeline_step("good", good)
        p.add_pipeline_step("bad", bad)
        # With Phase A2 the assertion message distinguishes per-step vs.
        # cumulative-floor failure; both are legitimate tolerance violations.
        with pytest.raises(AssertionError, match="performance|floor|tolerance|limits"):
            p.run()

    def test_stop_step(self, tmp_path):
        p = Pipeline(str(tmp_path / "cache"))
        prod = ProducerStep(p, value=1)
        cons = ConsumerStep(p)
        p.add_pipeline_step("produce", prod)
        p.add_pipeline_step("consume", cons)
        p.run(stop_step="produce")

        assert "produce.data" in p.cache
        assert "consume.result" not in p.cache

    def test_hooks_called(self, tmp_path):
        p = Pipeline(str(tmp_path / "cache"))
        prod = ProducerStep(p, value=1)
        p.add_pipeline_step("produce", prod)

        pre_calls, post_calls = [], []
        p.register_pre_step_hook(lambda name, step: pre_calls.append(name))
        p.register_post_step_hook(lambda name, step: post_calls.append(name))
        p.run()

        assert pre_calls == ["produce"]
        assert post_calls == ["produce"]


class TestStepCleanup:
    """Tests that the pipeline calls step.cleanup() and that it runs in finally."""

    def test_cleanup_called_after_successful_step(self, tmp_path):
        p = Pipeline(str(tmp_path / "cache"))
        step = CleanupTrackingStep(p, promises_data=True)
        p.add_pipeline_step("tracked", step)
        p.run()

        assert step.cleanup_called == [True], "cleanup() should be called once after the step runs"

    def test_cleanup_called_for_each_step(self, tmp_path):
        p = Pipeline(str(tmp_path / "cache"))
        s1 = CleanupTrackingStep(p, promises_data=True)
        s2 = CleanupTrackingStep(p, promises_data=True)
        p.add_pipeline_step("first", s1)
        p.add_pipeline_step("second", s2)
        p.run()

        assert s1.cleanup_called == [True]
        assert s2.cleanup_called == [True]

    def test_cleanup_called_even_when_step_fails_after_validate(self, tmp_path):
        p = Pipeline(str(tmp_path / "cache"))
        p.tolerance = 0.95
        p.add_pipeline_step("good", ProducerStep(p, value=1))
        failing = CleanupTrackingStep(p, promises_data=False, metric=0.01)
        p.add_pipeline_step("bad", failing)

        with pytest.raises(AssertionError, match="performance|floor|tolerance|limits"):
            p.run()

        assert failing.cleanup_called == [True], (
            "cleanup() should be called in finally even when the step fails the tolerance check"
        )

    def test_cleanup_called_when_validate_raises(self, tmp_path):
        class ValidateRaisingStep(CleanupTrackingStep):
            def validate(self):
                raise RuntimeError("validate failed")

        p = Pipeline(str(tmp_path / "cache"))
        step = ValidateRaisingStep(p, promises_data=True)
        p.add_pipeline_step("raises", step)

        with pytest.raises(RuntimeError, match="validate failed"):
            p.run()

        assert step.cleanup_called == [True], "cleanup() should be called in finally when validate() raises"


class TestPipelineMetric:
    """Tests for pipeline_metric() auto-discovery and pipeline usage."""

    def test_pipeline_metric_fallback_to_validate(self, tmp_path):
        """Without a trainer/tuner, pipeline_metric() returns validate()."""
        p = Pipeline(str(tmp_path / "cache"))
        step = ProducerStep(p, value=1)
        step.name = "produce"
        assert step.pipeline_metric() == step.validate()

    def test_pipeline_metric_discovers_trainer_test(self, tmp_path):
        """When step.trainer has test(), pipeline_metric() calls it."""

        class TrainerStub:
            def test(self):
                return 0.99

        p = Pipeline(str(tmp_path / "cache"))
        step = ProducerStep(p, value=1)
        step.name = "produce"
        step.trainer = TrainerStub()
        assert step.pipeline_metric() == pytest.approx(0.99)

    def test_pipeline_metric_discovers_tuner_trainer(self, tmp_path):
        """When step.tuner.trainer has test(), pipeline_metric() calls it."""

        class TrainerStub:
            def test(self):
                return 0.88

        class TunerStub:
            def __init__(self):
                self.trainer = TrainerStub()

        p = Pipeline(str(tmp_path / "cache"))
        step = ProducerStep(p, value=1)
        step.name = "produce"
        step.tuner = TunerStub()
        assert step.pipeline_metric() == pytest.approx(0.88)

    def test_pipeline_uses_pipeline_metric_for_target(self, tmp_path):
        """Pipeline._run_step sets target metric from pipeline_metric(), not validate()."""

        class MetricStep(PipelineStep):
            def __init__(self, pipeline):
                super().__init__([], ["data"], [], [], pipeline)

            def process(self):
                self.add_entry("data", 1)

            def validate(self):
                return 0.50

            def pipeline_metric(self):
                return 0.95

        p = Pipeline(str(tmp_path / "cache"))
        p.add_pipeline_step("ms", MetricStep(p))
        p.run()
        assert p.get_target_metric() == pytest.approx(0.95)


class TestPipelineRunFrom:
    def test_run_from_requires_valid_step(self, tmp_path):
        p = Pipeline(str(tmp_path / "cache"))
        p.add_pipeline_step("produce", ProducerStep(p))
        with pytest.raises(AssertionError):
            p.run_from("nonexistent")

    def test_run_from_skips_earlier_steps(self, tmp_path):
        p = Pipeline(str(tmp_path / "cache"))
        prod = ProducerStep(p, value=3)
        cons = ConsumerStep(p)
        p.add_pipeline_step("produce", prod)
        p.add_pipeline_step("consume", cons)

        p.run(stop_step="produce")
        p.save_cache()

        p2 = Pipeline(str(tmp_path / "cache"))
        prod2 = ProducerStep(p2, value=99)
        cons2 = ConsumerStep(p2)
        p2.add_pipeline_step("produce", prod2)
        p2.add_pipeline_step("consume", cons2)
        p2.run_from("consume")

        assert cons2.consumed == 3
