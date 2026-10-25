"""[H3b] Calibrate THIS host's compute rate into declarable physics overrides.

Usage: PYTHONPATH=src env/bin/python scripts/calibrate_host.py [out.json] [--declared rows.json]
Writes generated/host_calibration.json by default. The artifact's
``platform_physics_overrides`` block is the named ``host`` block of
``merge_override_blocks`` (read it with ``host_calibration.load_host_block``).
``--declared`` adds host rows the calibration cannot measure (``p_host`` where
RAPL is unreadable), each a physics-constant declaration with its own evidence;
a row both measured and declared refuses by name.
"""

import argparse
import json
import os

from mimarsinan.deployment_record.platform_physics.host_calibration import (
    run_calibration,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", nargs="?", default="generated/host_calibration.json")
    parser.add_argument("--declared", default=None,
                        help="JSON object of declared host rows, e.g. {\"p_host\": {...}}")
    args = parser.parse_args()
    declared = None
    if args.declared is not None:
        with open(args.declared, "r", encoding="utf-8") as fh:
            declared = json.load(fh)
    artifact = run_calibration(declared=declared)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)
        fh.write("\n")
    block = artifact["platform_physics_overrides"]
    print(f"machine: {artifact['machine']} ({artifact['threads']} torch thread(s), "
          f"CPUs {artifact['cpu_affinity']})")
    print(f"host_macs_per_s: {artifact['host_macs_per_s']:.3e} ({artifact['host_macs_per_s']/1e9:.2f} G/s)")
    dispatch = block["t_host_op_overhead"]
    print(f"t_host_op_overhead: {dispatch['low']:.2f} / {dispatch['nominal']:.2f} / {dispatch['high']:.2f} us "
          f"(p10 / median / p90)")
    power = block.get("p_host")
    if power is None:
        print(f"p_host: absent ({artifact['p_host_basis']})")
    else:
        print(f"p_host: {power.get('low', power['nominal'])} / {power['nominal']} / "
              f"{power.get('high', power['nominal'])} {power.get('unit', 'W')} ({artifact['p_host_basis']})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
