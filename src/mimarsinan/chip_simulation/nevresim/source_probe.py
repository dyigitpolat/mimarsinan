"""The two checks of a source-tree probe: declared sites, then the tree's own constexpr self-test.

A behaviour that lives in nevresim's sources rather than in mimarsinan's (W33's core-local stage-input read,
``chip_simulation/stage_input_clock``; the ODIN program's integration policies, ``nevresim/integration_policy_read``)
is established on a checkout only where the tree a run compiles against holds every declared site and the C++20
compiler the backend builds with constant-evaluates the tree's own self-test in a ``static_assert`` (syntax only).
Moved here from ``stage_input_clock`` unchanged [U-H], so both probes share one implementation without an import
cycle through the driver."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from mimarsinan.common.build_utils import find_cpp20_compiler

#: One site: its path under ``include/``, the text the site holds, and the older texts it replaced.
ReadSite = Tuple[str, str, Tuple[str, ...]]
SELF_TEST_TIMEOUT_S = 300.0


def site_problems(include: Path, sites: Sequence[ReadSite]) -> List[str]:
    """Per read site, a missing file, a missing text, or a global-cycle read the site still holds."""
    out: List[str] = []
    for path, marker, replaced in sites:
        file = include / path
        if not file.is_file():
            out.append(f"{path} is missing")
            continue
        text = file.read_text(errors="replace")
        if marker not in text:
            out.append(f"{path} does not hold {marker!r}")
        out.extend(f"{path} still holds the global-cycle read {r!r}" for r in replaced if r in text)
    return out


def self_test_problem(include: Path, self_test: Tuple[str, str],
                      compile_options: Callable[[str], List[str]]) -> Optional[str]:
    """Why the tree's self-test does not hold under the backend's compiler, or None when it holds."""
    header, function = self_test
    compiler, family = find_cpp20_compiler()
    if compiler is None or family is None:
        return f"no C++20 compiler is found to evaluate the self-test {function}"
    with tempfile.TemporaryDirectory() as work:
        probe = Path(work) / "stage_input_read_probe.cpp"
        probe.write_text(f'#include "{header}"\nstatic_assert({function}());\n')
        cmd = [compiler, "-std=c++20", "-fsyntax-only", *compile_options(family), "-I", str(include), str(probe)]
        try:
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=SELF_TEST_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return f"the self-test {function} did not finish within {SELF_TEST_TIMEOUT_S:.0f} s ({compiler})"
    if done.returncode != 0:
        return f"the self-test {function} does not hold ({compiler}: {' '.join(done.stderr.split())[-300:]})"
    return None
