#!/usr/bin/env python3
"""Emit (or CHECK) the RTL a generated chip configuration's fabric is built from.

The files land under ``hw/gen/chips/<chip>/`` and are COMMITTED, for the same
reason ``kernel.xml`` is frozen into the package: the build runs from a package
that ships no ``src/``, so the generator cannot run there. ``--check`` is the
gate that keeps the committed bytes equal to what the template would emit today.

    env/bin/python scripts/hacc/gen_chip_rtl.py --chip odin_wide_1024x256_mb16
    env/bin/python scripts/hacc/gen_chip_rtl.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from mimarsinan.chip_simulation.odin_fpga.chip_configs import (  # noqa: E402
    ChipConfig,
    chip_config_named,
    chip_configs,
)
from mimarsinan.mapping.export.odin_gen.fabric import (  # noqa: E402
    generate_fabric,
    write_generated_fabric,
)
from mimarsinan.mapping.export.odin_gen.generate import (  # noqa: E402
    DESCRIPTOR_FILENAME,
)


def emit(config: ChipConfig) -> int:
    fabric = generate_fabric(config.core_spec)
    written = write_generated_fabric(fabric, config.committed_rtl_root)
    for path in written:
        print(f"[chip-rtl] wrote {path.relative_to(REPO)}")
    return 0


def check(config: ChipConfig) -> int:
    fabric = generate_fabric(config.core_spec)
    root = config.committed_rtl_root
    drift = []
    descriptor = (
        json.dumps(fabric.core.descriptor, indent=1, sort_keys=True) + "\n"
    ).encode("utf-8")
    for name, payload in tuple(fabric.files) + ((DESCRIPTOR_FILENAME, descriptor),):
        path = root / name
        if not path.is_file():
            drift.append(f"{path.relative_to(REPO)} is missing")
        elif path.read_bytes() != payload:
            drift.append(f"{path.relative_to(REPO)} differs from the template")
    if drift:
        print(f"[chip-rtl] {config.name}: DRIFT\n  " + "\n  ".join(drift),
              file=sys.stderr)
        return 1
    print(f"[chip-rtl] {config.name}: {len(fabric.files) + 1} committed file(s) "
          f"byte-identical to the generator's output")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chip", default=None,
                        help="one configuration; default is every generated one")
    parser.add_argument("--check", action="store_true",
                        help="compare the committed files instead of writing them")
    options = parser.parse_args()
    if options.chip is not None:
        targets = [chip_config_named(options.chip)]
    else:
        targets = [c for c in chip_configs() if not c.is_stock]
    status = 0
    for config in targets:
        if config.is_stock:
            print(f"[chip-rtl] {config.name} is the VENDORED fabric: its kernel is "
                  f"the committed hw/fpga/kernel tree and nothing is generated "
                  f"for it.", file=sys.stderr)
            return 2
        status |= check(config) if options.check else emit(config)
    return status


if __name__ == "__main__":
    sys.exit(main())
