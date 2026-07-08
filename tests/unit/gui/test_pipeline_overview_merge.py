"""The monitor's WS ``pipeline_overview`` applier must never drop server fields.

Two producers emit ``pipeline_overview`` frames with *different* key sets — the
in-process collector (``get_pipeline_overview``) and the active-run tailer
(``get_run_detail``) — and both are applied by the same client code. A field the
client forgets to copy is silently erased from the page state; that is how the
Configuration tab used to fall back to its raw table mid-run. The merge is
therefore key-agnostic, and this test executes the real ES module under Node.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_MODULE = (
    Path(__file__).resolve().parents[3]
    / "src" / "mimarsinan" / "gui" / "static" / "js" / "pipeline-state.js"
)


def _merge(prev, frame):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to execute the monitor's ES modules")
    script = (
        f"import {{ mergePipelineOverview }} from {json.dumps(_MODULE.as_uri())};\n"
        f"const prev = {json.dumps(prev)};\n"
        f"const frame = {json.dumps(frame)};\n"
        "process.stdout.write(JSON.stringify(mergePipelineOverview(prev, frame)));\n"
    )
    proc = subprocess.run(
        [node, "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"node failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


class TestMergePipelineOverview:
    def test_module_exists(self) -> None:
        assert _MODULE.is_file(), f"missing {_MODULE}"

    def test_frame_without_config_view_keeps_the_structured_view(self) -> None:
        """The Configuration-tab regression: a frame that omits ``config_view``
        must not erase the view the initial REST fetch installed."""
        prev = {"steps": [], "config_view": {"summary": {"experiment_name": "e"}}}
        merged = _merge(prev, {"type": "pipeline_overview", "steps": [{"name": "s1"}]})
        assert merged["config_view"] == {"summary": {"experiment_name": "e"}}
        assert merged["steps"] == [{"name": "s1"}]

    def test_frame_with_config_view_replaces_it(self) -> None:
        prev = {"config_view": {"summary": {"experiment_name": "old"}}}
        frame = {"type": "pipeline_overview", "config_view": {"summary": {"experiment_name": "new"}}}
        assert _merge(prev, frame)["config_view"]["summary"]["experiment_name"] == "new"

    def test_envelope_keys_never_leak_into_pipeline_state(self) -> None:
        merged = _merge({}, {"type": "pipeline_overview", "event_seq": 12, "steps": []})
        assert "type" not in merged
        assert "event_seq" not in merged

    def test_explicit_null_overwrites(self) -> None:
        """``current_step`` goes null when a step completes: null is data, not
        an omission, so it must win over the previous value."""
        merged = _merge({"current_step": "Pretraining"}, {"current_step": None})
        assert merged["current_step"] is None

    def test_active_run_liveness_and_error_survive(self) -> None:
        """Active-run frames carry ``is_alive``/``status``/``error``; hardcoding
        ``is_alive: true`` hid the failure banner for dead subprocess runs."""
        frame = {"type": "pipeline_overview", "is_alive": False, "status": "failed", "error": "boom"}
        merged = _merge({"is_alive": True}, frame)
        assert merged["is_alive"] is False
        assert merged["status"] == "failed"
        assert merged["error"] == "boom"

    def test_unknown_server_field_flows_through(self) -> None:
        """A field the client has never heard of must reach the page state, so
        adding one server-side never needs a matching client edit."""
        merged = _merge({}, {"type": "pipeline_overview", "future_field": [1, 2, 3]})
        assert merged["future_field"] == [1, 2, 3]

    def test_steps_defaults_to_a_list(self) -> None:
        assert _merge(None, {"type": "pipeline_overview"})["steps"] == []
