"""The wizard's live Mapping Performance panel must preview the DEPLOYED program.

``/api/hw_config_verify`` reads the chip's capability declaration off the request
body through ``ChipCapabilities.from_platform_constraints``. The client used to
send three permission bits and not ``max_schedule_passes``, so a scheduled
platform was previewed against a program composed under the wrong pass budget.

These execute the client's own request builder under Node and feed its output to
the real server-side reader, so a dropped key is a failing round trip rather than
a silently wrong panel.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities

_JS_DIR = (
    Path(__file__).resolve().parents[3]
    / "src" / "mimarsinan" / "gui" / "static" / "js" / "wizard"
)
_MODULE = _JS_DIR / "hw_request.js"


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to execute the wizard's ES modules")
    return node


def _node_eval(expr: str, bindings: dict):
    script = f"import * as hw from {json.dumps(_MODULE.as_uri())};\n"
    for name, value in bindings.items():
        script += f"const {name} = {json.dumps(value)};\n"
    script += f"process.stdout.write(JSON.stringify({expr}));\n"
    proc = subprocess.run(
        [_node(), "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"node failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


def _declaration(values: dict):
    """``capabilityDeclaration(effectiveValue)`` over a draft's resolved values."""
    return _node_eval("hw.capabilityDeclaration((k) => values[k])", {"values": values})


def _verify_body(values: dict, cores: list):
    return _node_eval(
        "hw.hwVerifyBody({model_type: 'm'}, cores, "
        "hw.capabilityDeclaration((k) => values[k]))",
        {"values": values, "cores": cores},
    )


class TestTheClientDeclaresEveryCapabilityTheServerReads:
    def test_the_two_key_sets_are_equal(self):
        """A capability added server-side must reach this client, or this fails."""
        assert set(_declaration({})) == set(ChipCapabilities().layout_kwargs())

    def test_an_unset_draft_declares_the_servers_own_defaults(self):
        assert _declaration({}) == ChipCapabilities().layout_kwargs()


class TestTheSchedulingDeclarationSurvivesTheRoundTrip:
    VALUES = {
        "allow_coalescing": True,
        "allow_neuron_splitting": False,
        "allow_scheduling": True,
        "max_schedule_passes": 3,
    }

    def test_the_server_reads_back_exactly_what_the_wizard_declared(self):
        body = _verify_body(self.VALUES, [{"max_axons": 32, "max_neurons": 32, "count": 2}])
        capabilities = ChipCapabilities.from_platform_constraints(body)
        assert capabilities.max_schedule_passes == 3
        assert capabilities.allow_scheduling is True
        assert capabilities.allow_coalescing is True

    def test_the_posted_body_carries_the_whole_layout_declaration(self):
        body = _verify_body(self.VALUES, [{"max_axons": 8, "max_neurons": 8, "count": 1}])
        missing = set(ChipCapabilities().layout_kwargs()) - set(body)
        assert not missing, missing
        assert body["core_types"] == [{"max_axons": 8, "max_neurons": 8, "count": 1}]
        assert body["model_repr_json"] == {"model_type": "m"}

    def test_an_undeclared_budget_posts_the_default(self):
        body = _verify_body({"allow_scheduling": True}, [])
        assert ChipCapabilities.from_platform_constraints(body).max_schedule_passes == 8


class TestTheModulesParse:
    @pytest.mark.parametrize("name", ["hw_request.js", "hw.js"])
    def test_node_checks_the_wizard_hardware_modules(self, name):
        proc = subprocess.run(
            [_node(), "--check", str(_JS_DIR / name)],
            capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
