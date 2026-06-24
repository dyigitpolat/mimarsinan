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
    AttributionFidelity,
    CoverageStatus,
    ScreeningStatus,
    claimed_subproduct,
    coverage_report,
    honest_claimed_subproduct,
)
from mimarsinan.chip_simulation.coverage_ci import (  # noqa: E402
    DEFAULT_FLAG_AGE_DAYS,
    audit_coverage_instrument,
    CoverageGuardError,
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
    builder = claimed_subproduct if args.legacy_denominator else honest_claimed_subproduct
    return builder(**pins)


def _print_axes() -> None:
    print("\n=== AXIS SCREENING STATUS (the coverage denominator's witness) ===")
    for axis in AXES:
        status = axis.screening_status.value.upper()
        flags = []
        if axis.collapsed:
            flags.append(f"COLLAPSED→{axis.representative!r}")
        if axis.interacts_with:
            flags.append("interacts_with=" + ",".join(axis.interacts_with))
        artifact = axis.screening_artifact.strip()
        if not artifact:
            if axis.screening_status is ScreeningStatus.ASSERTED_UNSCREENED:
                artifact = "ASSERTED-UNSCREENED — counted interacting (no screen yet)"
            else:
                artifact = "(enumerated interacting — no collapse claimed)"
        suffix = ("  [" + "; ".join(flags) + "]") if flags else ""
        print(f"  {axis.name:<18} {status:<22}{suffix}")
        print(f"    artifact: {artifact[:110]}")


def _print_attribution_fidelity(report) -> None:
    fidelity = report.attribution_fidelity
    if not fidelity:
        return
    cracked = [
        region
        for region, fid in fidelity.items()
        if fid is AttributionFidelity.VALUE_DOMAIN_ONLY
    ]
    if cracked:
        print(f"\nATTRIBUTION-FIDELITY: {len(cracked)} region(s) are VALUE_DOMAIN_ONLY "
              f"(deployed accuracy bit-exact, per-neuron ATTRIBUTION known-cracked):")
        for region in cracked:
            print(f"  - {region}")


def _print_flag_aging(report) -> None:
    meta = report.flag_metadata
    if not meta:
        return
    print(f"\nFLAG AGING ({len(meta)} flagged cell(s) — owner + age):")
    for m in meta:
        owner = m.owner or "(UNOWNED)"
        age = f"{m.age_days}d" if m.age_days is not None else "age?"
        print(f"  - {owner:<14} {age:<6} {m.cell_key}")


def _print_report(report, claim) -> None:
    print("=== HYPERVOLUME COVERAGE REPORT ===")
    print(f"covered cells (science-valid ledger rows): {report.covered_cell_count}")
    tc = report.tier_counts
    # VALID and VALID_FLAGGED are reported SEPARATELY — never merged into one total.
    print(
        f"  by tier:  VALID={tc[CoverageStatus.VALID]}  "
        f"VALID_FLAGGED={tc[CoverageStatus.VALID_FLAGGED]}  "
        f"INVALID={tc[CoverageStatus.INVALID]}"
    )
    if claim:
        # ALWAYS print the claimed sub-product SIZE next to the fraction (never a bare
        # 0.75 with no denominator).
        print(
            f"\nclaimed sub-product: {report.claimed_subproduct_size} cells  |  "
            f"covered: {report.covered_claimed_count}  |  "
            f"HONEST coverage fraction: {report.coverage_fraction:.4f}  "
            f"(of {report.claimed_subproduct_size})"
        )
        if report.untested_frontier:
            shown = report.untested_frontier[:20]
            print(f"\nUNTESTED frontier ({len(report.untested_frontier)} cells, "
                  f"showing {len(shown)}):")
            for cell in shown:
                print(f"  - {cell.cell_key}")
        else:
            print("\nUNTESTED frontier: (none — the claimed sub-product is fully covered)")
    else:
        print("\n(no claimed sub-product pinned; pass --vehicle/--dataset/--sync to "
              "measure a coverage fraction and an untested frontier)")

    _print_attribution_fidelity(report)
    _print_flag_aging(report)

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
                        help="also print the axis screening status")
    parser.add_argument("--legacy-denominator", action="store_true",
                        help="use the LEGACY single-default claim (an unpinned axis "
                             "collapses to one default) instead of the honest "
                             "enumerate-unscreened denominator")
    parser.add_argument("--now", default=None,
                        help="reference date (YYYY-MM-DD) for flag aging (default today)")
    parser.add_argument("--max-flag-age-days", type=int, default=DEFAULT_FLAG_AGE_DAYS,
                        help="CI threshold: an UNOWNED flag older than this fails --ci")
    parser.add_argument("--ci", action="store_true",
                        help="run the self-audit guards and exit non-zero on violation")
    args = parser.parse_args(argv)

    rows = _read_ledger(args.ledger)
    claim = _build_claim(args)
    report = coverage_report(rows, claimed_subproduct=claim or None, now_ts=args.now)

    if args.ci:
        try:
            audit_coverage_instrument(AXES, report, max_age_days=args.max_flag_age_days)
        except CoverageGuardError as exc:
            print(f"COVERAGE CI FAILED: {exc}")
            return 1
        print("COVERAGE CI PASSED: axes screening sound; tiers separate; no aged "
              "unowned flags.")
        return 0

    if args.json:
        out = report.to_dict()
        out["ledger"] = args.ledger
        out["ledger_rows"] = len(rows)
        out["denominator"] = "legacy" if args.legacy_denominator else "honest"
        print(json.dumps(out, indent=2))
        return 0

    print(f"ledger: {args.ledger}  ({len(rows)} rows)")
    print(f"denominator: {'LEGACY single-default' if args.legacy_denominator else 'HONEST enumerate-unscreened'}")
    _print_report(report, claim)
    if args.axes:
        _print_axes()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
