#!/usr/bin/env python
"""Do the declared physics profiles reproduce the measurements their papers publish?

Prices each shipped reference case — a published operating point whose workload census
comes from the paper's own structural statements — with the profile it names, and
compares axis by axis against the published number.

    python scripts/silicon_correlation.py
    python scripts/silicon_correlation.py --cases odin_biological_time
    python scripts/silicon_correlation.py --json correlation.json

Exits non-zero if any case falls outside its declared tolerance, so this is runnable
as a gate. A miss is a finding about the constants, never a reason to widen the band.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mimarsinan.deployment_record.correlation import (  # noqa: E402
    available_cases,
    correlate_all,
    render_correlation,
)


def _payload(results) -> dict:
    return {
        "cases": [
            {
                "name": result.case.name,
                "profile": result.case.profile,
                "measurement_kind": result.measurement_kind,
                "independence": result.case.independence,
                "citation": result.case.citation,
                "overrides_applied": dict(result.overrides_applied),
                "passed": result.passed,
                "axes": [
                    {
                        "axis": axis.name,
                        "predicted": axis.predicted,
                        "published": axis.published,
                        "error_pct": None if axis.predicted is None else axis.error_pct,
                        "refusal": axis.refusal,
                    }
                    for axis in result.axes
                ],
            }
            for result in results
        ]
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", default=",".join(available_cases()),
        help="comma-separated case names (default: every shipped case)")
    parser.add_argument("--json", help="also write the correlation here")
    args = parser.parse_args(argv)

    names = [name.strip() for name in args.cases.split(",") if name.strip()]
    results = correlate_all(names)
    print(render_correlation(results))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(_payload(results), handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\nwrote {args.json}")

    missed = [result.case.name for result in results if not result.passed]
    if missed:
        print(f"\nOUTSIDE TOLERANCE: {', '.join(missed)}")
        return 1
    print(f"\nall {len(results)} cases within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
