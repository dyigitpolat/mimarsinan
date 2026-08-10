"""Shape-aware remedy targeting: a server remedy op may carry a ``path``.

A retired key found NESTED inside a structural pc container (the legacy
hw-search shapes ``platform_constraints.user`` / ``.auto.fixed``) cannot be
cleared by scope routing — deleting at the pc ROOT leaves the nested
declaration alive and the error returns on the next resolve. The parse layer
records where it found the key and the remedy op carries that container
``path``; ``state.js`` walks it (creating containers on a ``set``, never on a
``clear``). These tests execute the ES module under Node.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_WIZARD_JS = (
    Path(__file__).resolve().parents[3]
    / "src" / "mimarsinan" / "gui" / "static" / "js" / "wizard"
)


def _apply(draft, calls):
    """Load state.js, seed ``state.draft``, run the given calls, return the
    draft. Each call is (fn, args...) with fn one of setKey/clearKey."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to execute the wizard's ES modules")
    script = (
        f"import * as s from {json.dumps((_WIZARD_JS / 'state.js').as_uri())};\n"
        f"s.state.draft = {json.dumps(draft)};\n"
    )
    for fn, *args in calls:
        script += f"s.{fn}({', '.join(json.dumps(a) for a in args)});\n"
    script += "process.stdout.write(JSON.stringify(s.state.draft));\n"
    proc = subprocess.run(
        [node, "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"node failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


_PC = "platform_constraints"


def _legacy_draft():
    return {
        "deployment_parameters": {},
        _PC: {"mode": "user", "user": {"allow_weight_reuse": True,
                                       "weight_bits": 5}},
    }


class TestClearWithPath:
    def test_clear_reaches_the_nested_container(self):
        draft = _apply(_legacy_draft(), [
            ("clearKey", "allow_weight_reuse", _PC, [_PC, "user"]),
        ])
        # the nested declaration is gone; its siblings and the shape survive
        assert draft[_PC] == {"mode": "user", "user": {"weight_bits": 5}}

    def test_scope_routing_without_a_path_misses_the_nested_key(self):
        # The pre-path behavior, pinned as the CONTRAST: scope routing edits
        # the pc root, so the nested legacy declaration survives.
        draft = _apply(_legacy_draft(), [
            ("clearKey", "allow_weight_reuse", _PC),
        ])
        assert draft[_PC]["user"]["allow_weight_reuse"] is True

    def test_clear_never_creates_missing_containers(self):
        draft = _apply({"deployment_parameters": {}, _PC: {}}, [
            ("clearKey", "allow_weight_reuse", _PC, [_PC, "user"]),
        ])
        assert draft[_PC] == {}

    def test_clear_stops_at_a_non_container_segment(self):
        start = {"deployment_parameters": {}, _PC: {"user": 7}}
        draft = _apply(start, [
            ("clearKey", "allow_weight_reuse", _PC, [_PC, "user"]),
        ])
        assert draft == start

    def test_auto_fixed_path_clears_the_search_shape(self):
        draft = _apply({
            "deployment_parameters": {},
            _PC: {"mode": "auto",
                  "auto": {"fixed": {"allow_weight_reuse": False}}},
        }, [
            ("clearKey", "allow_weight_reuse", _PC, [_PC, "auto", "fixed"]),
        ])
        assert draft[_PC]["auto"]["fixed"] == {}


class TestSetWithPath:
    def test_set_creates_the_container_walk(self):
        draft = _apply({"deployment_parameters": {}, _PC: {}}, [
            ("setKey", "weight_bits", 4, _PC, [_PC, "user"]),
        ])
        assert draft[_PC]["user"] == {"weight_bits": 4}

    def test_pathless_set_keeps_scope_routing(self):
        draft = _apply({"deployment_parameters": {}, _PC: {}}, [
            ("setKey", "weight_bits", 4, _PC),
        ])
        assert draft[_PC] == {"weight_bits": 4}


class TestRemedyWiring:
    def test_review_forwards_the_op_path_to_state(self):
        source = (_WIZARD_JS / "review.js").read_text(encoding="utf-8")
        actions = source[source.index("const REMEDY_ACTIONS"):]
        actions = actions[: actions.index("};") + 2]
        assert "setKey(remedy.key, remedy.value, remedy.scope, remedy.path)" in actions
        assert "clearKey(remedy.key, remedy.scope, remedy.path)" in actions
