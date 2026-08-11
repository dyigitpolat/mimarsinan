"""SanafeRunner host-op wall timing (W4.3).

``time_host_stages`` is the opt-in: each ``run()`` times its host ComputeOp
stages with a fresh ``StageTimer`` and surfaces the walls on the per-sample
``SanafeRunRecord.compute_stage_walls`` (additive field — default off keeps
the record's walls empty and the run byte-identical).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from mimarsinan.chip_simulation.sanafe import runner as runner_mod
from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner


def _compute_only_mapping(op_id=42):
    op = SimpleNamespace(id=op_id)
    stage = SimpleNamespace(
        kind="compute", name="host_op", hard_core_mapping=None,
        compute_op=op, input_map=[], output_map=[],
        schedule_segment_index=None, schedule_pass_index=None,
    )
    return SimpleNamespace(
        stages=[stage],
        get_neural_segments=lambda: [],
        get_compute_ops=lambda: [op],
        output_sources=np.array([], dtype=object),
        node_activation_scales={},
        node_input_activation_scales={},
    )


def _run(monkeypatch, **runner_kwargs):
    monkeypatch.setattr(
        runner_mod, "execute_compute_op_numpy",
        lambda op, original_input, state_buffer, *, device, in_scale,
        out_scale, dtype=np.float32: np.asarray([[3.0]], dtype=dtype),
    )
    runner = SanafeRunner(
        mapping=_compute_only_mapping(), simulation_length=8, **runner_kwargs,
    )
    return runner.run(np.asarray([[1.0, 2.0]], dtype=np.float32), sample_index=0)


def test_opt_in_surfaces_measured_walls_on_the_record(monkeypatch):
    rec = _run(monkeypatch, time_host_stages=True)
    (wall,) = rec.compute_stage_walls
    json.dumps(rec.compute_stage_walls)
    assert wall["stage_index"] == 0
    assert wall["name"] == "host_op"
    assert wall["invocations"] == 1
    assert wall["wall_s_total"] >= 0.0


def test_default_off_keeps_walls_empty(monkeypatch):
    rec = _run(monkeypatch)
    assert rec.compute_stage_walls == []


def test_each_run_gets_a_fresh_timer(monkeypatch):
    monkeypatch.setattr(
        runner_mod, "execute_compute_op_numpy",
        lambda op, original_input, state_buffer, *, device, in_scale,
        out_scale, dtype=np.float32: np.asarray([[3.0]], dtype=dtype),
    )
    runner = SanafeRunner(
        mapping=_compute_only_mapping(), simulation_length=8,
        time_host_stages=True,
    )
    sample = np.asarray([[1.0, 2.0]], dtype=np.float32)
    first = runner.run(sample, sample_index=0)
    second = runner.run(sample, sample_index=1)
    # Per-sample records never carry another sample's accumulation.
    assert first.compute_stage_walls[0]["invocations"] == 1
    assert second.compute_stage_walls[0]["invocations"] == 1
