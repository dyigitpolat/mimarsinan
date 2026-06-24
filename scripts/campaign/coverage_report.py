"""Frontier E1 CLI — print the hypervolume coverage report over the campaign ledger.

Reads ``runs/campaign/ledger.jsonl`` (or ``--ledger``), GROUP-BYs the science-valid
rows into hypervolume cells keyed by the full deployment tuple, and prints:

  * the tier tally (VALID / VALID_FLAGGED / INVALID) over the covered cells,
  * the coverage FRACTION against a claimed sub-product (``--vehicle`` / ``--dataset``
    / ``--sync`` / … pin axes; unpinned axes take a single screened representative),
  * the named UNTESTED frontier (claimed cells with no row),
  * the RESEARCH-GAP frontier (the union of ``research_gap_ops`` over VALID_FLAGGED
    cells — the future-conversion targets),
  * the axis classification (orthogonal vs interacting; the collapsed axis).

Run:  python scripts/campaign/coverage_report.py [--ledger PATH] [--json]
                                                  [--vehicle deep_cnn ...]
                                                  [--dataset mnist fmnist ...]
                                                  [--sync cascaded synchronized ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

from mimarsinan.chip_simulation.coverage_ledger import (  # noqa: E402
    AXES,
    CoverageStatus,
    claimed_subproduct,
    coverage_report,
)


def _default_ledger() -> str:
    env = os.environ.get("MIM_CAMPAIGN_DIR")
    base = env if env else os.path.join(REPO, "runs", "campaign")
    return os.path.join(base, "ledger.jsonl")


def _read_ledger(path: str) -> list:
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _build_claim(args) -> list:
    pins = {}
    if args.vehicle:
        pins["vehicle"] = args.vehicle
    if args.dataset:
        pins["dataset"] = args.dataset
    if args.sync:
        pins["sync"] = args.sync
    if args.firing:
        pins["firing"] = args.firing
    if args.S:
        pins["S"] = args.S
    if args.depth:
        pins["depth"] = args.depth
    if not pins:
        return []
    return claimed_subproduct(**pins)


def _print_axes() -> None:
    print("\n=== AXIS CLASSIFICATION (orthogonal vs interacting) ===")
    for axis in AXES:
        tag = axis.kind.value.upper()
        flags = []
        if axis.collapsed:
            flags.append(f"COLLAPSED→{axis.representative!r}")
        if axis.interacts_with:
            flags.append("interacts_with=" + ",".join(axis.interacts_with))
        suffix = ("  [" + "; ".join(flags) + "]") if flags else ""
        print(f"  {axis.name:<18} {tag:<11}{suffix}")


def _print_report(report, claim) -> None:
    print("=== HYPERVOLUME COVERAGE REPORT ===")
    print(f"covered cells (science-valid ledger rows): {report.covered_cell_count}")
    tc = report.tier_counts
    print(
        f"  by tier:  VALID={tc[CoverageStatus.VALID]}  "
        f"VALID_FLAGGED={tc[CoverageStatus.VALID_FLAGGED]}  "
        f"INVALID={tc[CoverageStatus.INVALID]}"
    )
    if claim:
        print(
            f"\nclaimed sub-product: {report.claimed_cell_count} cells  |  "
            f"covered: {report.covered_claimed_count}  |  "
            f"coverage fraction: {report.coverage_fraction:.3f}"
        )
        if report.untested_frontier:
            print(f"\nUNTESTED frontier ({len(report.untested_frontier)} cells):")
            for cell in report.untested_frontier:
                print(f"  - {cell.cell_key}")
        else:
            print("\nUNTESTED frontier: (none — the claimed sub-product is fully covered)")
    else:
        print("\n(no claimed sub-product pinned; pass --vehicle/--dataset/--sync to "
              "measure a coverage fraction and an untested frontier)")

    if report.research_gap_frontier:
        print(f"\nRESEARCH-GAP frontier ({len(report.research_gap_frontier)} ops — "
              f"the future-conversion targets over VALID_FLAGGED cells):")
        for op in report.research_gap_frontier:
            print(f"  - {op}")
    else:
        print("\nRESEARCH-GAP frontier: (none — no unsupported-op flags)")

    if report.placement_fixable_frontier:
        print(f"\nPLACEMENT-FIXABLE frontier ({len(report.placement_fixable_frontier)} "
              f"ops — offloadable encoders that un-flag the cell; NOT research gaps):")
        for op in report.placement_fixable_frontier:
            print(f"  - {op}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="E1 hypervolume coverage report.")
    parser.add_argument("--ledger", default=_default_ledger(),
                        help="path to ledger.jsonl (default: runs/campaign/ledger.jsonl)")
    parser.add_argument("--vehicle", nargs="*", default=None, help="pin the vehicle axis")
    parser.add_argument("--dataset", nargs="*", default=None, help="pin the dataset axis")
    parser.add_argument("--sync", nargs="*", default=None, help="pin the sync axis")
    parser.add_argument("--firing", nargs="*", default=None, help="pin the firing axis")
    parser.add_argument("--S", nargs="*", default=None, help="pin the S axis")
    parser.add_argument("--depth", nargs="*", default=None, help="pin the depth axis")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument("--axes", action="store_true",
                        help="also print the axis classification")
    args = parser.parse_args(argv)

    rows = _read_ledger(args.ledger)
    claim = _build_claim(args)
    report = coverage_report(rows, claimed_subproduct=claim or None)

    if args.json:
        out = report.to_dict()
        out["ledger"] = args.ledger
        out["ledger_rows"] = len(rows)
        print(json.dumps(out, indent=2))
        return 0

    print(f"ledger: {args.ledger}  ({len(rows)} rows)")
    _print_report(report, claim)
    if args.axes:
        _print_axes()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
