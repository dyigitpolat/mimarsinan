#!/usr/bin/env python3
"""[W6b] Rebuild the softcore-elimination table from a stored pruned IR graph.

The per-run artifact is emitted by the soft-core mapping seam
(``softcore_elimination.json`` / ``.md``). This probe is the RETROSPECTIVE
path: point it at a cached ``*.ir_graph.pickle`` from an old run and it
reconstructs the same table from the elimination masks that graph retained.

Only the realized arm is recoverable this way (the weaker arms left no trace),
and cores the liveness pass deleted are absent from the denominator.

Usage::

    python scripts/softcore_elimination_from_ir.py <ir_graph.pickle> \
        [--geometry pre_elimination|as_stored] [--out DIR]
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mimarsinan.mapping.softcore_elimination import (  # noqa: E402
    GEOMETRY_AS_STORED,
    GEOMETRY_PRE_ELIMINATION,
    render_softcore_elimination_markdown,
    report_from_pruned_ir_graph,
    summarize_softcore_elimination,
    write_softcore_elimination_markdown,
    write_softcore_elimination_record,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ir_graph_pickle")
    parser.add_argument(
        "--geometry",
        default=GEOMETRY_PRE_ELIMINATION,
        choices=(GEOMETRY_PRE_ELIMINATION, GEOMETRY_AS_STORED),
    )
    parser.add_argument("--out", default=None, help="write JSON + markdown here")
    args = parser.parse_args(argv)

    with open(args.ir_graph_pickle, "rb") as f:
        graph = pickle.load(f)

    report = report_from_pruned_ir_graph(graph, geometry=args.geometry)
    print(summarize_softcore_elimination(report))
    print(render_softcore_elimination_markdown(report))
    if args.out:
        print(write_softcore_elimination_record(report, args.out))
        print(write_softcore_elimination_markdown(report, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
