"""Call the browser's own ES modules from python tests, under node.

A rule the python suite RE-IMPLEMENTS is a rule whose deletion in the browser
the suite cannot notice — the wizard's objective-chip filter was exactly that
until a mutation walked out of it alive. Tests that depend on a client-side
rule call it through here instead, so they exercise the code the browser runs.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

JS_ROOT = (
    Path(__file__).resolve().parent.parent
    / "src" / "mimarsinan" / "gui" / "static" / "js"
)

NODE_TIMEOUT_S = 60


def call_js(module: Path, function: str, *args: Any) -> Any:
    """Call *function*, exported by the ES module at *module*, on JSON *args*.

    Returns the JSON-decoded return value. Skips (never fails) when node is
    absent: the rule is still the browser's, and a machine without node cannot
    say anything about it either way.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to execute the browser's ES modules")
    assert module.is_file(), f"missing ES module {module}"
    script = (
        f"import {{ {function} }} from {json.dumps(module.as_uri())};\n"
        f"const args = {json.dumps(list(args))};\n"
        f"process.stdout.write(JSON.stringify({function}(...args)));\n"
    )
    proc = subprocess.run(
        [node, "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=NODE_TIMEOUT_S,
    )
    assert proc.returncode == 0, f"node failed running {function}:\n{proc.stderr}"
    return json.loads(proc.stdout)
