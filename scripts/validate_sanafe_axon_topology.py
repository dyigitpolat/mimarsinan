#!/usr/bin/env python3
"""Validate HCM cross-tile connectivity vs SANA-FE network wiring for a run.

Counts static LIF cross-tile edges from the HardCoreMapping and compares
them to the connectivity overlay used by SanafeRunner (same graph source).

Usage::

    PYTHONPATH=src python scripts/validate_sanafe_axon_topology.py \\
        generated/mnist_hard_all_lif_ca60_phased_deployment_run_20260522_101111
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


def _load_hcm_from_run(run_dir: Path):
    p = run_dir / "SANA-FE Simulation.sanafe_simulation_results.pickle"
    if not p.is_file():
        raise FileNotFoundError(p)
    with open(p, "rb") as f:
        report = pickle.load(f)
    rec = report.per_sample[0]
    seg = list(rec.segments.values())[0]
    return seg, seg.connectivity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--cores-per-tile", type=int, default=8)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    from mimarsinan.chip_simulation.sanafe.analysis import (
        _compute_connectivity_edges,
        _count_cross_tile_connectivity_edges,
    )

    seg, connectivity = _load_hcm_from_run(run_dir)
    cpt = max(int(args.cores_per_tile), 1)
    cross_edges = _count_cross_tile_connectivity_edges(
        connectivity, cores_per_tile=cpt,
    )
    mapped = int(getattr(seg, "mapped_cross_tile_axons", cross_edges))

    print(f"run: {run_dir.name}")
    print(f"stage: {seg.stage_name}")
    print(f"cross_tile_connectivity_edges (HCM LIF): {cross_edges}")
    print(f"mapped_cross_tile_axons (recorded): {mapped}")
    print(f"inter_tile_packets (observed): {getattr(seg, 'inter_tile_packets', 0)}")

    if cross_edges == 0:
        print("NOTE: no static cross-tile LIF edges in connectivity overlay.")
        return 0
    if mapped == 0:
        print("WARN: static cross-tile edges exist but mapped axon count is zero.")
        return 1
    if getattr(seg, "inter_tile_packets", 0) == 0:
        print(
            "INFO: mapped cross-tile routes exist but message_trace has no "
            "inter-tile deliveries (check TTFS event emission / re-sim)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
