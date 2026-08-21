#!/usr/bin/env python3
"""Regenerate hw/fpga/SYNTH_REPORT.md + synth_resources.json from a fresh yosys run.

Deliberate action: the [slow] gate `tests/integration/test_odin_rtl_synth.py`
requires an EXACT match against the committed numbers, so run this only when the
RTL (or the pinned yosys) actually changed, and review the diff.

    env/bin/python scripts/hw_tests/regen_synth_report.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mimarsinan.chip_simulation.odin_rtl.synth_artifacts import (  # noqa: E402
    REPORT_MD,
    RESOURCES_JSON,
    build_record,
    write_artifacts,
)
from mimarsinan.chip_simulation.odin_rtl.synth_census import (  # noqa: E402
    memory_inferences,
)
from mimarsinan.chip_simulation.odin_rtl.synthesis import (  # noqa: E402
    run_synthesis,
    yosys_version,
)


def main() -> int:
    version = yosys_version()
    with tempfile.TemporaryDirectory(prefix="odin_synth_") as scratch:
        root = Path(scratch)
        print(f"[synth] {version}")
        overlay = run_synthesis(overlay=True, workdir=root / "overlay")
        print(f"[synth] vendor+overlay: "
              f"{'ok' if overlay.succeeded else 'FAILED'} in {overlay.seconds:.1f} s")
        if not overlay.succeeded:
            print(f"[synth] {overlay.error_line}\n{overlay.log_tail}", file=sys.stderr)
            return 1
        control = run_synthesis(overlay=False, workdir=root / "control")
        print(f"[synth] vendored memories alone (F17 control): "
              f"{'ok' if control.succeeded else 'refused'} in {control.seconds:.1f} s")
        record = build_record(
            attempt=overlay, control=control, tool_version=version)

    for finding in (m.finding() for m in memory_inferences(overlay.require_stat())):
        if finding is not None:
            print(f"[synth] {finding}", file=sys.stderr)
    write_artifacts(record)
    print(f"[synth] per-core: {record['per_core']}")
    print(f"[synth] wrote {RESOURCES_JSON}\n[synth] wrote {REPORT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
