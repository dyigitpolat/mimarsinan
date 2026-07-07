#!/usr/bin/env python3
"""Regenerate the golden resolution snapshot; audit the diff before committing."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
HARNESS = ROOT / "tests" / "unit" / "config_schema" / "test_golden_resolution_snapshot.py"


def main() -> int:
    spec = importlib.util.spec_from_file_location("golden_resolution_harness", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    snapshot = module.build_snapshot()
    path = Path(module.SNAPSHOT_PATH)
    path.write_text(json.dumps(snapshot, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path} ({len(snapshot['configs'])} configs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
