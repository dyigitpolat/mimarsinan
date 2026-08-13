#!/usr/bin/env python
"""Price one deployment across every declared target — the cross-platform study.

Takes a SEALED deployment record (or a hand-declared census) and answers the
chip-designer's question for each shipped physics profile: what would this
deployment cost on that chip?

    python scripts/cross_platform_study.py generated_files/<run>/deployment_record.json
    python scripts/cross_platform_study.py --profiles truenorth,loihi <record.json>

Every row discloses its evidence, an axis a target cannot back is absent WITH its
reason rather than zero, and a table mixing measurement bases says so.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mimarsinan.deployment_record.platform_physics import (  # noqa: E402
    available_profiles,
)
from mimarsinan.deployment_record.quantities import from_record  # noqa: E402
from mimarsinan.deployment_record.schema.record import (  # noqa: E402
    load_deployment_record,
)
from mimarsinan.deployment_record.study import (  # noqa: E402
    compare_platforms,
    render_comparison,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", help="a sealed deployment_record.json")
    parser.add_argument(
        "--profiles",
        default=",".join(available_profiles()),
        help="comma-separated profile names (default: every shipped profile)",
    )
    parser.add_argument("--json", help="also write the comparison here")
    args = parser.parse_args(argv)

    record = load_deployment_record(args.record)
    census = from_record(record)
    comparison = compare_platforms(
        census, [name.strip() for name in args.profiles.split(",") if name.strip()]
    )

    print(f"deployment: {record.identity.cell_key}  ({record.identity.run_dir})")
    print()
    print(render_comparison(comparison))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(comparison.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
