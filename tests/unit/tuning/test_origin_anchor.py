"""Origin-anchored conversion (spiking_deployment_calculus.md §13.2 L-A/L-B).

L-A: with ``origin_teacher_kd`` armed, every conversion tuner's KD anchor is
the pipeline's cached ORIGIN (reference) teacher — fail-loud when the
snapshot step never cached one; unarmed keeps the historical per-step
self-snapshot byte-path. L-B: with ``origin_anchored_compact`` armed, step
floors/targets anchor to the origin metric (reference-teacher metric, else
the retention envelope) and the target adjuster never relaxes on a miss.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from mimarsinan.tuning.adaptation_target_adjuster import AdaptationTargetAdjuster
from mimarsinan.tuning.orchestration.retention_envelope import (
    RETENTION_ENVELOPE_CACHE_KEY,
    origin_metric,
    resolve_step_anchor,
)
from mimarsinan.tuning.teacher import (
    find_reference_teacher,
    resolve_conversion_teacher,
)


class _DuckPipeline:
    def __init__(self, config=None, cache=None, target=None):
        self.config = config or {}
        self.cache = cache if cache is not None else {}
        self._target = target

    def get_target_metric(self):
        return self._target


class TestOriginTeacherResolver:
    def test_unarmed_returns_frozen_self_snapshot(self):
        model = nn.Linear(3, 2)
        pipe = _DuckPipeline(config={"device": "cpu"})
        teacher = resolve_conversion_teacher(pipe, model)
        assert teacher is not model
        assert all(not p.requires_grad for p in teacher.parameters())
        assert not teacher.training

    def test_armed_returns_the_cached_reference_teacher(self):
        reference = nn.Linear(3, 2)
        pipe = _DuckPipeline(
            config={"device": "cpu", "origin_teacher_kd": True},
            cache={"Reference Teacher Snapshot.reference_teacher_model": reference},
        )
        teacher = resolve_conversion_teacher(pipe, nn.Linear(3, 2))
        assert teacher is reference
        assert all(not p.requires_grad for p in teacher.parameters())

    def test_armed_without_cached_teacher_fails_loud(self):
        pipe = _DuckPipeline(config={"device": "cpu", "origin_teacher_kd": True})
        with pytest.raises(RuntimeError, match="Reference Teacher Snapshot"):
            resolve_conversion_teacher(pipe, nn.Linear(3, 2))

    def test_finder_scans_by_key_suffix(self):
        reference = nn.Linear(2, 2)
        pipe = _DuckPipeline(
            cache={"Any Step Name.reference_teacher_model": reference, "x": 1},
        )
        assert find_reference_teacher(pipe) is reference
        assert find_reference_teacher(_DuckPipeline()) is None


class TestOriginMetricAndAnchor:
    def test_origin_metric_prefers_reference_metric_over_envelope(self):
        pipe = _DuckPipeline(cache={
            "Reference Teacher Snapshot.reference_teacher_metric": 0.8678,
            RETENTION_ENVELOPE_CACHE_KEY: 0.852,
        })
        assert origin_metric(pipe) == pytest.approx(0.8678)

    def test_origin_metric_falls_back_to_envelope(self):
        pipe = _DuckPipeline(cache={RETENTION_ENVELOPE_CACHE_KEY: 0.852})
        assert origin_metric(pipe) == pytest.approx(0.852)
        assert origin_metric(_DuckPipeline()) is None

    def test_anchor_unarmed_is_the_rolling_target(self):
        pipe = _DuckPipeline(
            cache={RETENTION_ENVELOPE_CACHE_KEY: 0.852}, target=0.7711,
        )
        assert resolve_step_anchor(pipe) == pytest.approx(0.7711)

    def test_anchor_armed_is_the_origin(self):
        pipe = _DuckPipeline(
            config={"origin_anchored_compact": True},
            cache={
                "Reference Teacher Snapshot.reference_teacher_metric": 0.8678,
            },
            target=0.7711,
        )
        assert resolve_step_anchor(pipe) == pytest.approx(0.8678)

    def test_anchor_armed_without_origin_falls_back_to_rolling(self):
        pipe = _DuckPipeline(
            config={"origin_anchored_compact": True}, target=0.7711,
        )
        assert resolve_step_anchor(pipe) == pytest.approx(0.7711)


class TestFrozenTargetAdjuster:
    def test_frozen_adjuster_never_relaxes_on_miss(self):
        adjuster = AdaptationTargetAdjuster(0.8678, decay=0.99, frozen=True)
        for _ in range(50):
            adjuster.update_target(0.10)
        assert adjuster.get_target() == pytest.approx(0.8678)

    def test_unfrozen_adjuster_keeps_the_historical_decay(self):
        adjuster = AdaptationTargetAdjuster(0.8678, decay=0.99)
        adjuster.update_target(0.10)
        assert adjuster.get_target() < 0.8678


class TestSnapshotStepGating:
    def test_applies_when_origin_teacher_kd_armed(self):
        from mimarsinan.pipelining.pipeline_steps.training.reference_teacher_snapshot_step import (
            ReferenceTeacherSnapshotStep,
        )

        class _Plan:
            def __init__(self, config):
                self.config = config

        assert ReferenceTeacherSnapshotStep.applies_to(
            _Plan({"origin_teacher_kd": True})
        )
        assert not ReferenceTeacherSnapshotStep.applies_to(_Plan({}))
