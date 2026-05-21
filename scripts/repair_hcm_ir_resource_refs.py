"""Backfill ``step: "Soft Core Mapping"`` on IR-graph resource refs embedded
in the Hard Core Mapping step's snapshot for already-completed runs.

Why
---
Older runs were snapshotted before
``snapshot_ir_graph(source_step_name=...)`` existed: the Hardware tab's
embedded IR-graph snapshot duplicated every IR heatmap descriptor, the
``GUIHandle`` snapshot executor's 30 s drain timeout truncated the
persistence loop, and the Hardware tab now shows missing-image icons
for cores whose PNGs never landed in
``_GUI_STATE/resources/Hard Core Mapping/ir_core_heatmap/``.

The fix on the live path is to embed the IR summary with a ``step``
hint that points at the SCM step's resource folder. This script applies
the same rewrite to historical runs in-place so they recover without a
full pipeline rerun.

Usage
-----
    python scripts/repair_hcm_ir_resource_refs.py <run_dir> [<run_dir> ...]

Each ``<run_dir>`` is a generated-pipeline directory (the parent of
``_GUI_STATE/``). The script is idempotent: refs that already carry a
``step`` field are left untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


SOURCE_STEP = "Soft Core Mapping"
HARDWARE_STEP = "Hard Core Mapping"
RESOURCE_REF_KEYS = ("heatmap_resource", "pre_pruning_resource")


def _tag_ref(ref: dict) -> bool:
    """Add ``step`` hint to a resource ref if missing. Returns True iff
    the ref was modified."""
    if not isinstance(ref, dict) or "kind" not in ref or "rid" not in ref:
        return False
    if "step" in ref:
        return False
    ref["step"] = SOURCE_STEP
    return True


def repair_steps_json(steps_json_path: Path) -> int:
    """Tag IR resource refs in the HCM snapshot. Returns count of refs tagged."""
    with steps_json_path.open() as f:
        doc = json.load(f)

    hcm = doc.get("steps", {}).get(HARDWARE_STEP, {})
    snap = hcm.get("snapshot") or {}
    ir = snap.get("ir_graph") or {}
    tagged = 0

    for node in ir.get("nodes", []) or []:
        for key in RESOURCE_REF_KEYS:
            if _tag_ref(node.get(key)):
                tagged += 1

    for bank in (ir.get("weight_banks") or {}).values():
        if _tag_ref(bank.get("heatmap_resource")):
            tagged += 1

    if tagged == 0:
        return 0

    tmp = steps_json_path.with_suffix(steps_json_path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(doc, f)
    tmp.replace(steps_json_path)
    return tagged


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    total = 0
    for raw in argv:
        run_dir = Path(raw)
        steps_json = run_dir / "_GUI_STATE" / "steps.json"
        if not steps_json.is_file():
            print(f"[skip] {steps_json}: not a file")
            continue
        n = repair_steps_json(steps_json)
        print(f"[ok] {steps_json}: tagged {n} IR resource ref(s)")
        total += n
    print(f"Total refs tagged: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
