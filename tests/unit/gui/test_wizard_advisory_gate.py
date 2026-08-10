"""The wizard's acknowledge-to-launch gate over deployment advisories.

Gating advisories — ``mandate_violation`` OR severity ``UNSUPPORTED`` — each
demand an explicit client-side acknowledgment before Launch enables, and every
stored acknowledgment is invalidated the moment a config edit changes the
resolved set of gating advisory ids (an ack given to one advisory set must
never carry over to a different one). The logic is pure and lives in the ES
module ``static/js/wizard/advisories.js``; this test executes it under Node.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_MODULE = (
    Path(__file__).resolve().parents[3]
    / "src" / "mimarsinan" / "gui" / "static" / "js" / "wizard" / "advisories.js"
)


def _node_eval(expr, bindings):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to execute the wizard's ES modules")
    script = f"import * as gate from {json.dumps(_MODULE.as_uri())};\n"
    for name, value in bindings.items():
        script += f"const {name} = {json.dumps(value)};\n"
    script += f"process.stdout.write(JSON.stringify({expr}));\n"
    proc = subprocess.run(
        [node, "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"node failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


def _adv(advisory_id, severity="INFO", mandate=False):
    """A resolve-payload advisory row (the schema_api ``as_payload`` shape)."""
    return {
        "id": advisory_id,
        "severity": severity,
        "title": "t",
        "detail": "d",
        "tentative": True,
        "mandate_violation": mandate,
        "suggested_levers": [],
    }


class TestGatingClassification:
    def test_module_exists(self):
        assert _MODULE.is_file(), f"missing {_MODULE}"

    def test_unsupported_severity_gates(self):
        assert _node_eval(
            "gate.isGatingAdvisory(row)", {"row": _adv("A", "UNSUPPORTED")},
        ) is True

    def test_mandate_violation_gates_regardless_of_severity(self):
        assert _node_eval(
            "gate.isGatingAdvisory(row)", {"row": _adv("A", "RISK", mandate=True)},
        ) is True
        assert _node_eval(
            "gate.isGatingAdvisory(row)", {"row": _adv("A", "INFO", mandate=True)},
        ) is True

    def test_plain_risk_and_info_do_not_gate(self):
        assert _node_eval(
            "gate.isGatingAdvisory(row)", {"row": _adv("A", "RISK")},
        ) is False
        assert _node_eval(
            "gate.isGatingAdvisory(row)", {"row": _adv("A", "INFO")},
        ) is False

    def test_gating_ids_are_sorted_and_unique(self):
        rows = [
            _adv("ADV-B", "UNSUPPORTED"),
            _adv("ADV-A", "RISK", mandate=True),
            _adv("ADV-B", "UNSUPPORTED"),
            _adv("ADV-C", "INFO"),
        ]
        assert _node_eval(
            "gate.gatingAdvisoryIds(rows)", {"rows": rows},
        ) == ["ADV-A", "ADV-B"]


class TestAckStateReset:
    def test_fresh_state_from_null(self):
        state = _node_eval(
            "gate.nextAckState(null, rows)", {"rows": [_adv("A", "UNSUPPORTED")]},
        )
        assert state == {"gatingIds": ["A"], "acked": []}

    def test_acks_survive_an_unchanged_gating_set(self):
        prev = {"gatingIds": ["A", "B"], "acked": ["A"]}
        rows = [_adv("B", "UNSUPPORTED"), _adv("A", "RISK", mandate=True)]
        state = _node_eval(
            "gate.nextAckState(prev, rows)", {"prev": prev, "rows": rows},
        )
        assert state == {"gatingIds": ["A", "B"], "acked": ["A"]}

    def test_non_gating_advisory_changes_never_reset_acks(self):
        """A new RISK/INFO row leaves the GATING id set unchanged, so
        acknowledgments stand."""
        prev = {"gatingIds": ["A"], "acked": ["A"]}
        rows = [_adv("A", "UNSUPPORTED"), _adv("NEW-INFO", "INFO")]
        state = _node_eval(
            "gate.nextAckState(prev, rows)", {"prev": prev, "rows": rows},
        )
        assert state["acked"] == ["A"]

    def test_a_new_gating_id_resets_every_ack(self):
        prev = {"gatingIds": ["A"], "acked": ["A"]}
        rows = [_adv("A", "UNSUPPORTED"), _adv("B", "UNSUPPORTED")]
        state = _node_eval(
            "gate.nextAckState(prev, rows)", {"prev": prev, "rows": rows},
        )
        assert state == {"gatingIds": ["A", "B"], "acked": []}

    def test_a_departed_gating_id_resets_every_ack(self):
        prev = {"gatingIds": ["A", "B"], "acked": ["A", "B"]}
        state = _node_eval(
            "gate.nextAckState(prev, rows)",
            {"prev": prev, "rows": [_adv("A", "UNSUPPORTED")]},
        )
        assert state == {"gatingIds": ["A"], "acked": []}

    def test_a_swapped_gating_id_resets_every_ack(self):
        prev = {"gatingIds": ["A"], "acked": ["A"]}
        state = _node_eval(
            "gate.nextAckState(prev, rows)",
            {"prev": prev, "rows": [_adv("B", "UNSUPPORTED")]},
        )
        assert state == {"gatingIds": ["B"], "acked": []}

    def test_all_advisories_clearing_empties_the_state(self):
        prev = {"gatingIds": ["A"], "acked": ["A"]}
        state = _node_eval(
            "gate.nextAckState(prev, rows)", {"prev": prev, "rows": []},
        )
        assert state == {"gatingIds": [], "acked": []}


class TestAckToggleAndPending:
    def test_ack_then_unack_round_trip(self):
        acked = _node_eval(
            "gate.withAck(state, 'A', true)",
            {"state": {"gatingIds": ["A", "B"], "acked": []}},
        )
        assert acked == {"gatingIds": ["A", "B"], "acked": ["A"]}
        unacked = _node_eval(
            "gate.withAck(state, 'A', false)", {"state": acked},
        )
        assert unacked == {"gatingIds": ["A", "B"], "acked": []}

    def test_unknown_id_never_enters_the_ack_set(self):
        state = _node_eval(
            "gate.withAck(state, 'GHOST', true)",
            {"state": {"gatingIds": ["A"], "acked": []}},
        )
        assert state == {"gatingIds": ["A"], "acked": []}

    def test_pending_lists_unacked_gating_ids(self):
        assert _node_eval(
            "gate.pendingAckIds(state)",
            {"state": {"gatingIds": ["A", "B"], "acked": ["B"]}},
        ) == ["A"]

    def test_pending_empty_once_every_gating_id_is_acked(self):
        assert _node_eval(
            "gate.pendingAckIds(state)",
            {"state": {"gatingIds": ["A", "B"], "acked": ["A", "B"]}},
        ) == []

    def test_pending_on_null_state_is_empty(self):
        assert _node_eval("gate.pendingAckIds(null)", {}) == []

    def test_launch_flow_two_gating_rows(self):
        """resolve → ack one → still pending → ack the other → launchable."""
        rows = [_adv("A", "UNSUPPORTED"), _adv("B", "RISK", mandate=True)]
        pending = _node_eval(
            "gate.pendingAckIds(gate.withAck(gate.nextAckState(null, rows), 'A', true))",
            {"rows": rows},
        )
        assert pending == ["B"]
        done = _node_eval(
            "gate.pendingAckIds("
            "gate.withAck(gate.withAck(gate.nextAckState(null, rows), 'A', true), 'B', true))",
            {"rows": rows},
        )
        assert done == []
