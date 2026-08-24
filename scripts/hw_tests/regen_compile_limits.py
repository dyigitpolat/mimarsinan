#!/usr/bin/env python3
"""Regenerate hw/fpga/compile_limits.json + the study doc from fresh yosys runs.

Deliberate action: the [slow] gate `tests/integration/test_odin_compile_limits.py`
re-derives every non-stock configuration and requires an EXACT match against the
committed record, so run this only when the RTL, the generator or the studied
configuration set actually changed, and review the diff.

    env/bin/python scripts/hw_tests/regen_compile_limits.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mimarsinan.chip_simulation.odin_rtl.limits.artifacts import (  # noqa: E402
    LIMITS_JSON,
    STUDY_MD,
    load_committed,
    measure_all,
    write_record,
)
from mimarsinan.chip_simulation.odin_rtl.limits.report import render_study  # noqa: E402
from mimarsinan.chip_simulation.odin_rtl.synthesis import yosys_version  # noqa: E402


def main() -> int:
    if "--render-only" in sys.argv[1:]:
        record = load_committed()
        print("[limits] rendering the study from the COMMITTED record; no "
              "synthesis ran and no number moved")
    else:
        version = yosys_version()
        print(f"[limits] {version}")
        with tempfile.TemporaryDirectory(prefix="odin_limits_") as scratch:
            record = measure_all(workdir=Path(scratch), tool_version=version)
        write_record(record)
    STUDY_MD.write_text(render_study(record), encoding="utf-8")
    for row in record["configurations"]:
        census = row["census"]
        print(f"[limits] {row['key']}: LUTeq={census['lut_equivalent']:,} "
              f"FF={census['flip_flops']:,} CARRY={census['carry']:,} "
              f"LUTRAM={census['lutram']:,} BRAM36={census['bram36']:,} "
              f"URAM={census['uram']:,}")
    print(f"[limits] wrote {LIMITS_JSON}\n[limits] wrote {STUDY_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
