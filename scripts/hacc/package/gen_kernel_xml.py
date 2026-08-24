#!/usr/bin/env python3
"""Hand ``build_xclbn.sh`` the kernel.xml that was FROZEN from the host SSOT.

The repository's own ``scripts/hacc/gen_kernel_xml.py`` derives the register map
from ``mimarsinan.chip_simulation.odin_fpga.kernel_registers`` — the SSOT the
host driver also reads. That import needs the ``src/`` tree, which this package
deliberately does not ship. So ``scripts/hacc/make_package.py`` RUNS the real
generator here, freezes its output as ``scripts/hacc/kernel.xml`` next to this
file, and ships this stand-in with the same command line.

There is therefore still exactly one register table, and no copy of it lives in
this file: the answer is a file, and this script only hands it over.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

FROZEN = Path(__file__).resolve().parent / "kernel.xml"
KERNEL_NAME = "odin_fpga_kernel_top"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kernel", default=KERNEL_NAME)
    options = parser.parse_args()
    if options.kernel != KERNEL_NAME:
        print(
            f"REFUSING: this package's kernel.xml was frozen for "
            f"{KERNEL_NAME!r}, not {options.kernel!r}. The packaged kernel and "
            f"the host driver resolve the SAME name; renaming it here would "
            f"build a kernel nothing can open.",
            file=sys.stderr)
        return 2
    if not FROZEN.is_file():
        print(
            f"REFUSING: {FROZEN} is missing. It is written by "
            f"scripts/hacc/make_package.py from the host-side register SSOT; "
            f"re-download the package rather than hand-writing one.",
            file=sys.stderr)
        return 2
    shutil.copyfile(FROZEN, options.output)
    digest = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    print(f"[hacc-build] wrote {options.output} from the frozen kernel.xml "
          f"(sha256 {digest[:16]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
