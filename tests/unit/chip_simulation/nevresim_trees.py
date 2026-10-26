"""nevresim source trees for the tests of the stage-input read probe, each built from this checkout's tree.

- ``checkout_tree()``: the tree nevresim compiles against here (``nevresim_root()``).
- ``base_tree(tmp)``: base 90af7d2, the tree without W33's core-local stage-input read. It is this checkout's include
  tree with the four headers the change edits restored from ``tests/fixtures/nevresim_90af7d2`` and the two files it
  adds removed; on a checkout whose nevresim is the base plus ``nevresim.patch`` this is the base's include tree byte
  for byte.
- ``global_cycle_policy_tree(tmp)``: every read site of the change present, but the compute policy reads each stage
  input at the global cycle (``local_cycle = cycle``), so only the tree's own self-test can tell it from the change."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from mimarsinan.common.env import nevresim_root

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "nevresim_90af7d2" / "include"
#: The files W33's change adds to nevresim's include tree.
ADDED = ("simulator/spike_generation/stage_input_train.hpp", "_tests/stage_input_clock.hpp")
POLICY = "simulator/compute_policy/compute_policy_base.hpp"
_LOCAL_CYCLE = re.compile(r"local_cycle = cycle\s*-\s*chip\.get_cores\(\)\[core_id\]\.get_latency\(\) \* LatencyScale")


def checkout_tree() -> Path:
    return Path(nevresim_root())


def _copy(tmp, name: str) -> Path:
    root = Path(tmp) / name
    shutil.copytree(checkout_tree() / "include", root / "include")
    return root


def base_tree(tmp) -> Path:
    root = _copy(tmp, "nevresim_90af7d2")
    for path in ADDED:
        (root / "include" / path).unlink(missing_ok=True)
    for source in FIXTURE.rglob("*"):
        if source.is_file():
            shutil.copyfile(source, root / "include" / source.relative_to(FIXTURE))
    return root


def global_cycle_policy_tree(tmp) -> Path:
    root = _copy(tmp, "nevresim_global_cycle_policy")
    policy = root / "include" / POLICY
    text, n = _LOCAL_CYCLE.subn("local_cycle = cycle", policy.read_text())
    assert n == 1, "the checkout's compute policy holds no core-local read to undo"
    policy.write_text(text)
    return root
