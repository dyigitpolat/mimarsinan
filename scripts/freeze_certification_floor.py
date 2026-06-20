"""Freeze a per-cell regression floor for the Fix-B certification gate (Frontier E6).

Records the CURRENT deployed numbers for one (firing × sync × backend) cell into a
frozen floor book (JSON). This is the freezing step of the certification protocol:
run the matrix (a GPU run, NOT this script), read off the deployed-forward / wall-clock
numbers per cell, and freeze each cell here BEFORE the first Fix-B flip. The CI gate
(``mimarsinan.chip_simulation.certification.certify``) then certifies every new run
against this frozen floor instead of byte-identity.

This script does NOT run the matrix. It takes already-measured numbers and persists
the floor — so freezing is a pure, auditable, reviewable diff.

Usage:
    python scripts/freeze_certification_floor.py \\
        --book docs/certification/regression_floor.json \\
        --firing ttfs_cycle_based --sync cascaded --backend nevresim \\
        --deployed-accuracy 0.9540 --wall-clock-s 70.0 \\
        --eps 0.01 --wall-clock-slack 0.5 \\
        --commit "$(git rev-parse --short HEAD)"

See docs/CERTIFICATION_PROTOCOL.md for the full freeze + gate workflow.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mimarsinan.chip_simulation.certification import (  # noqa: E402
    CertificationCell,
    CertificationFloorBook,
    freeze_cell,
    load_floor_book,
    save_floor_book,
)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--book", required=True, help="path to the JSON floor book")
    p.add_argument("--firing", required=True, help="spiking mode (lif/ttfs/...)")
    p.add_argument(
        "--sync", default=None,
        help="ttfs_cycle schedule (cascaded/synchronized); omit for non-cycle modes",
    )
    p.add_argument("--backend", required=True, help="deployment backend (nevresim/...)")
    p.add_argument(
        "--deployed-accuracy", type=float, required=True,
        help="the deployed-forward, full-test, parity-gated accuracy (R6 number of record)",
    )
    p.add_argument(
        "--wall-clock-s", type=float, required=True,
        help="measured per-step wall-clock in seconds",
    )
    p.add_argument("--eps", type=float, default=0.0, help="accuracy slack (floor − eps)")
    p.add_argument(
        "--wall-clock-slack", type=float, default=0.0,
        help="fractional wall-clock budget slack (budget = wall × (1 + slack))",
    )
    p.add_argument(
        "--wall-clock-budget-s", type=float, default=None,
        help="absolute wall-clock budget override (wins over --wall-clock-slack)",
    )
    p.add_argument("--commit", default=None, help="commit the numbers were measured at")
    p.add_argument(
        "--samples", type=int, default=None,
        help="test-set sample count behind the deployed accuracy (provenance)",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    if os.path.isfile(args.book):
        book = load_floor_book(args.book)
    else:
        book = CertificationFloorBook()

    cell = CertificationCell(
        firing=args.firing, sync=args.sync, backend=args.backend
    )
    provenance = {
        "frozen_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if args.commit:
        provenance["commit"] = args.commit
    if args.samples is not None:
        provenance["samples"] = args.samples

    book = freeze_cell(
        book,
        cell,
        deployed_accuracy=args.deployed_accuracy,
        wall_clock_s=args.wall_clock_s,
        eps=args.eps,
        wall_clock_slack=args.wall_clock_slack,
        wall_clock_budget_s=args.wall_clock_budget_s,
        provenance=provenance,
    )
    save_floor_book(book, args.book)

    floor = book.floor_for(cell)
    assert floor is not None
    print(
        f"froze cell {cell.cell_key!r}: deployed_accuracy="
        f"{floor.deployed_accuracy:.4f} (floor {floor.accuracy_floor():.4f}), "
        f"wall_clock_s={floor.wall_clock_s:.1f} (budget {floor.wall_clock_budget():.1f}) "
        f"-> {args.book}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
