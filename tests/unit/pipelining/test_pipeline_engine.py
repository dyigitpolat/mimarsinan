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
        with pytest.raises(AssertionError, match="performance"):
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
