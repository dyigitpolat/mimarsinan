"""[H3b] Calibrate THIS host's compute rate into declarable physics overrides.

Usage: PYTHONPATH=src env/bin/python scripts/calibrate_host.py [out.json]
Writes generated/host_calibration.json by default; paste (or reference) the
``platform_physics_overrides`` block into a run config to price host terms
with measured evidence.
"""

import json
import os
import sys

from mimarsinan.deployment_record.platform_physics.host_calibration import (
    run_calibration,
)


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "generated/host_calibration.json"
    artifact = run_calibration()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)
        fh.write("\n")
    print(f"machine: {artifact['machine']}")
    print(f"host_macs_per_s: {artifact['host_macs_per_s']:.3e} ({artifact['host_macs_per_s']/1e9:.2f} G/s)")
    print(f"p_host: {artifact['p_host_w']} W ({artifact['p_host_basis']})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
