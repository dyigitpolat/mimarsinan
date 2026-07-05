"""Pretrain-envelope gate: absolute floor on the FIRST seeded metric (W2 gate).

``AccuracyBudget`` seeds from the first nonzero metric and every downstream
check is relative — a chance-level pretrain used to zombie-pass the whole
pipeline. The gate raises ``PretrainEnvelopeError`` at the unseeded->seeded
transition of ``Pipeline._record_step_metric`` when the seed metric is below
``pretrain_floor_chance_multiple / num_classes``.
"""

import pytest

from mimarsinan.pipelining.core.accuracy_budget import PretrainEnvelopeError
from mimarsinan.pipelining.core.engine.pipeline import Pipeline
from mimarsinan.pipelining.core.steps.pipeline_step import PipelineStep


class _MetricStep(PipelineStep):
    def __init__(self, pipeline, metric, key):
        super().__init__(
            requires=[], promises=[key], updates=[], clears=[], pipeline=pipeline
        )
        self._metric = float(metric)
        self._key = key

    def process(self):
        self.add_entry(self._key, 1)

    def validate(self):
        return self._metric


_WD_COUNTER = iter(range(1000))


def _pipeline(tmp_path, metrics, *, config=..., tolerance=None):
    pipeline = Pipeline(str(tmp_path / f"wd{next(_WD_COUNTER)}"))
    if config is ...:
        config = {"num_classes": 10}
    if config is not None:
        pipeline.config = config
    if tolerance is not None:
        pipeline.tolerance = tolerance
    for i, metric in enumerate(metrics):
        pipeline.add_pipeline_step(f"step{i}", _MetricStep(pipeline, metric, f"k{i}"))
    return pipeline


class TestEnvelopeGate:
    def test_chance_level_seed_dies_loud(self, tmp_path):
        """The t0_19 zombie: 0.1135 on MNIST must die at seeding, not exit 0."""
        pipeline = _pipeline(tmp_path, [0.1135])
        with pytest.raises(PretrainEnvelopeError) as exc:
            pipeline.run()
        message = str(exc.value)
        assert "0.1135" in message
        assert "0.5" in message  # the 5x-chance MNIST floor
        assert "10" in message  # num_classes
        assert "pretrain_floor_chance_multiple" in message

    def test_current_matrix_seed_levels_pass(self, tmp_path):
        """Min observed tier-0 pretrain is 0.9495: >= 0.44 above the 0.5 floor."""
        assert 0.9495 - 5.0 / 10 >= 0.44
        pipeline = _pipeline(tmp_path, [0.9495, 0.95])
        pipeline.run()  # must not raise

    def test_zero_multiple_disables_the_gate(self, tmp_path):
        pipeline = _pipeline(
            tmp_path, [0.1135],
            config={"num_classes": 10, "pretrain_floor_chance_multiple": 0},
        )
        pipeline.run()

    def test_gate_fires_only_at_the_seeding_transition(self, tmp_path):
        """Later steps below the absolute floor are the retention gate's business."""
        pipeline = _pipeline(tmp_path, [0.9, 0.45], tolerance=0.1)
        pipeline.run()  # 0.45 < floor 0.5 but the budget is already seeded

    def test_zero_metric_does_not_seed_the_gate_fires_on_first_nonzero(self, tmp_path):
        """The gate rides the budget's seeding rule: zero metrics never seed."""
        pipeline = _pipeline(tmp_path, [0.0, 0.1135], tolerance=0.1)
        with pytest.raises(PretrainEnvelopeError) as exc:
            pipeline.run()
        assert "step1" in str(exc.value)

    def test_missing_num_classes_disables_the_gate(self, tmp_path):
        pipeline = _pipeline(tmp_path, [0.1135], config={})
        pipeline.run()

    def test_configless_engine_pipeline_is_unaffected(self, tmp_path):
        pipeline = _pipeline(tmp_path, [0.1135], config=None)
        pipeline.run()

    def test_custom_multiple_scales_the_floor(self, tmp_path):
        config = {"num_classes": 10, "pretrain_floor_chance_multiple": 2.0}
        with pytest.raises(PretrainEnvelopeError):
            _pipeline(tmp_path, [0.15], config=dict(config)).run()  # floor 0.2
        _pipeline(tmp_path, [0.25], config=dict(config)).run()  # above floor
