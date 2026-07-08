"""Incremental Plotly streaming planner: append the tail, never double or lose.

The monitor's live curves (Overview target-metric line, per-step tuner curves)
grow one point at a time over the WebSocket. Re-plotting the whole trace on
every frame flashes and jumps; instead the client extends only the NEW tail via
``Plotly.extendTraces``. ``planTraceExtension`` is the pure decision that drives
that fast path — it decides extend vs full redraw vs no-op from the last-plotted
trace identities/counts and the desired full trace set.

Honesty: the planner only ever slices the TAIL of the caller-supplied REAL
points; it never fabricates or interpolates a sample. A structural change or a
SHRINK (a REST snapshot handing back fewer points after a reconnect) forces a
full redraw, so no point is ever doubled or lost. This test executes the real ES
module under Node.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_MODULE = (
    Path(__file__).resolve().parents[3]
    / "src" / "mimarsinan" / "gui" / "static" / "js" / "stream-plot.js"
)


def _plan(prev, nxt):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to execute the monitor's ES modules")
    script = (
        f"import {{ planTraceExtension }} from {json.dumps(_MODULE.as_uri())};\n"
        f"const prev = {json.dumps(prev)};\n"
        f"const next = {json.dumps(nxt)};\n"
        "process.stdout.write(JSON.stringify(planTraceExtension(prev, next)));\n"
    )
    proc = subprocess.run(
        [node, "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"node failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


def _trace(name, ys):
    return {"name": name, "x": list(range(len(ys))), "y": ys}


class TestPlanTraceExtension:
    def test_module_exists(self) -> None:
        assert _MODULE.is_file(), f"missing {_MODULE}"

    def test_no_prior_state_is_a_redraw(self) -> None:
        plan = _plan(None, [_trace("loss", [1.0, 0.9, 0.8])])
        assert plan["mode"] == "redraw"
        assert plan["names"] == ["loss"]
        assert plan["counts"] == [3]

    def test_unchanged_counts_is_a_noop(self) -> None:
        prev = {"names": ["loss"], "counts": [3]}
        plan = _plan(prev, [_trace("loss", [1.0, 0.9, 0.8])])
        assert plan["mode"] == "noop"
        assert plan["counts"] == [3]

    def test_growth_extends_only_the_tail(self) -> None:
        prev = {"names": ["loss"], "counts": [2]}
        plan = _plan(prev, [_trace("loss", [1.0, 0.9, 0.8, 0.7])])
        assert plan["mode"] == "extend"
        assert plan["indices"] == [0]
        # Exactly the two NEW real samples — no fabricated/interpolated values.
        assert plan["newYs"] == [[0.8, 0.7]]
        assert plan["newXs"] == [[2, 3]]
        assert plan["counts"] == [4]

    def test_structural_change_forces_redraw(self) -> None:
        prev = {"names": ["loss"], "counts": [2]}
        # A new trace joined the chart (e.g. Validation accuracy appears).
        plan = _plan(prev, [_trace("loss", [1.0, 0.9, 0.8]),
                            _trace("acc", [0.3, 0.4])])
        assert plan["mode"] == "redraw"
        assert plan["names"] == ["loss", "acc"]

    def test_renamed_trace_forces_redraw(self) -> None:
        prev = {"names": ["loss"], "counts": [1]}
        plan = _plan(prev, [_trace("accuracy", [0.3, 0.4])])
        assert plan["mode"] == "redraw"

    def test_shrink_reconciles_by_redraw_not_extend(self) -> None:
        # Reconnect / REST snapshot hands back FEWER points than we plotted:
        # extending would be wrong, so we redraw from the authoritative set.
        prev = {"names": ["loss"], "counts": [5]}
        plan = _plan(prev, [_trace("loss", [1.0, 0.9])])
        assert plan["mode"] == "redraw"
        assert plan["counts"] == [2]

    def test_multi_trace_extends_only_grown_traces(self) -> None:
        prev = {"names": ["loss", "acc"], "counts": [3, 2]}
        plan = _plan(prev, [_trace("loss", [1, 2, 3, 4, 5]),  # +2
                            _trace("acc", [0.1, 0.2])])         # unchanged
        assert plan["mode"] == "extend"
        assert plan["indices"] == [0]
        assert plan["newYs"] == [[4, 5]]
        assert plan["counts"] == [5, 2]

    def test_reconnect_replay_of_identical_snapshot_is_noop(self) -> None:
        # A REST reconcile that re-sends the exact same points must NOT double
        # them: the planner sees equal counts and does nothing.
        prev = {"names": ["m"], "counts": [4]}
        plan = _plan(prev, [_trace("m", [0.1, 0.2, 0.3, 0.4])])
        assert plan["mode"] == "noop"

    def test_empty_trace_set_against_empty_prior_is_noop(self) -> None:
        # Overview chart with only event-line markers and zero measured points:
        # no measured trace exists, so successive frames are no-ops (the markers
        # are relayout'd separately by the caller).
        plan = _plan({"names": [], "counts": []}, [])
        assert plan["mode"] == "noop"
        assert plan["counts"] == []


def _drive(frames):
    """Drive a synthetic sequential WS feed through the planner + a fake
    extendTraces (tail concat), returning the mode sequence and final plotted
    series. This is the live-mode exercise: it proves the streamed plot stays
    byte-for-byte reconciled with the source across growth, reconnect-replay,
    and shrink — no point doubled or lost."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to execute the monitor's ES modules")
    script = (
        f"import {{ planTraceExtension }} from {json.dumps(_MODULE.as_uri())};\n"
        f"const frames = {json.dumps(frames)};\n"
        "let state = null; let plotted = []; const modes = [];\n"
        "for (const ys of frames) {\n"
        "  const next = [{ name: 'm', x: ys.map((_, i) => i), y: ys }];\n"
        "  const plan = planTraceExtension(state, next);\n"
        "  modes.push(plan.mode);\n"
        "  if (plan.mode === 'redraw') plotted = ys.slice();\n"
        "  else if (plan.mode === 'extend') plotted = plotted.concat(plan.newYs[0]);\n"
        "  state = { names: plan.names, counts: plan.counts };\n"
        "}\n"
        "process.stdout.write(JSON.stringify({ modes, plotted }));\n"
    )
    proc = subprocess.run(
        [node, "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"node failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


class TestSequentialStream:
    def test_growth_then_reconnect_replay_then_shrink(self) -> None:
        result = _drive([
            [0.9],                 # first point -> redraw
            [0.9, 0.8],            # +1 -> extend
            [0.9, 0.8, 0.7],       # +1 -> extend
            [0.9, 0.8, 0.7],       # WS reconnect replays the full snapshot -> noop
            [0.9, 0.8, 0.7, 0.6],  # +1 -> extend
            [0.9, 0.8],            # REST reconcile hands back fewer -> redraw
        ])
        assert result["modes"] == [
            "redraw", "extend", "extend", "noop", "extend", "redraw",
        ]
        # After the reconcile the plotted series matches the authoritative set
        # exactly — and the identical-replay frame never doubled point 0.7.
        assert result["plotted"] == [0.9, 0.8]

    def test_streamed_series_never_duplicates_across_a_long_append(self) -> None:
        # 50 real points arriving one frame at a time must reproduce the source
        # series exactly on the plotted line (one redraw, then all extends).
        frames = [[round(1.0 - i * 0.01, 4) for i in range(n)] for n in range(1, 51)]
        result = _drive(frames)
        assert result["modes"][0] == "redraw"
        assert set(result["modes"][1:]) == {"extend"}
        assert result["plotted"] == frames[-1]
