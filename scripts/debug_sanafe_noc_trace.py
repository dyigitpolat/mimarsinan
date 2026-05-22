#!/usr/bin/env python3
"""Inspect SANA-FE NoC / spike-trace / TTFS activity for a finished run directory.

Usage::

    PYTHONPATH=src python scripts/debug_sanafe_noc_trace.py \\
        generated/mnist_hard_all_lif_ca60_phased_deployment_run_20260522_101111

Optional second argument: reference run for side-by-side comparison.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, List


def _load_segment(run_dir: Path):
    p = run_dir / "SANA-FE Simulation.sanafe_simulation_results.pickle"
    if not p.is_file():
        raise FileNotFoundError(p)
    with open(p, "rb") as f:
        report = pickle.load(f)
    rec = report.per_sample[0]
    seg = list(rec.segments.values())[0]
    return seg


def _core_has_activation(arr: Any) -> bool:
    if arr is None:
        return False
    try:
        import numpy as np
        a = np.asarray(arr, dtype=float)
        return a.size > 0 and bool((a > 0).any())
    except Exception:
        return False


def _contract_active_cores(seg) -> int:
    n = 0
    for c in getattr(seg, "contract_ttfs_cores", ()) or ():
        if _core_has_activation(getattr(c, "output_activation", None)):
            n += 1
    return n


def _hardware_active_cores(seg) -> int:
    return sum(
        1 for c in seg.per_core
        if _core_has_activation(getattr(c, "output_activation", None))
    )


def _event_active_cores(seg) -> int:
    return sum(1 for c in seg.per_core if int(getattr(c, "spikes_fired", 0)) > 0)


def _ttfs_mismatches(seg, limit: int = 8) -> List[str]:
    contract = {
        int(c.core_index): getattr(c, "output_activation", None)
        for c in (getattr(seg, "contract_ttfs_cores", ()) or ())
    }
    lines: List[str] = []
    for rec in seg.per_core:
        ci = int(rec.core_index)
        c_act = contract.get(ci)
        c_has = _core_has_activation(c_act)
        h_has = _core_has_activation(getattr(rec, "output_activation", None))
        e_has = int(getattr(rec, "spikes_fired", 0)) > 0
        if (c_has or h_has) and not e_has:
            lines.append(f"  core{ci}: contract={c_has} hardware={h_has} events={e_has}")
        if len(lines) >= limit:
            break
    return lines


def _summarize(seg, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"stage: {seg.stage_name}")
    print(f"packets_sent: {seg.packets_sent}")
    print(
        f"message taxonomy: inter={getattr(seg, 'inter_tile_packets', 0)} "
        f"intra={getattr(seg, 'intra_tile_packets', 0)} "
        f"input_path={getattr(seg, 'input_path_packets', 0)}"
    )
    print(f"cross_tile_connectivity_edges: {getattr(seg, 'cross_tile_connectivity_edges', 0)}")
    print(f"mapped_cross_tile_axons: {getattr(seg, 'mapped_cross_tile_axons', 0)}")
    print(f"noc_links: {len(seg.noc_links)}")
    print(
        f"spikes: chip={getattr(seg, 'chip_spike_count', seg.spikes)} "
        f"lif={getattr(seg, 'lif_spike_count', 0)} "
        f"seg.spikes={seg.spikes} parse_skipped="
        f"{getattr(seg, 'spike_trace_parse_skipped', 0)}"
    )
    contract_act = getattr(seg, "ttfs_contract_active_cores", None)
    if contract_act is None:
        contract_act = _contract_active_cores(seg)
    hw_act = getattr(seg, "ttfs_hardware_active_cores", None)
    if hw_act is None:
        hw_act = _hardware_active_cores(seg)
    evt_act = getattr(seg, "ttfs_event_active_cores", None)
    if evt_act is None:
        evt_act = _event_active_cores(seg)
    mismatch = getattr(seg, "ttfs_activation_event_mismatch_count", None)
    print(
        f"TTFS activity: contract_active={contract_act} "
        f"hardware_active={hw_act} event_active={evt_act} "
        f"mismatch={mismatch if mismatch is not None else 'n/a'}"
    )
    per_core_sum = sum(c.spikes_fired for c in seg.per_core)
    input_sum = sum(getattr(c, "input_neuron_spikes_fired", 0) for c in seg.per_core)
    print(f"per_core spikes_fired sum: {per_core_sum}")
    print(f"per_core input_neuron_spikes_fired sum: {input_sum}")
    mat = seg.per_neuron_spike_trace
    if mat is not None:
        print(f"spike matrix nonzero rows: {int((mat.sum(axis=1) > 0).sum())}")
    if seg.spike_capture_warning:
        print(f"warning: {seg.spike_capture_warning}")
    mismatches = _ttfs_mismatches(seg)
    if mismatches:
        print("TTFS activation without hardware events (first cores):")
        for line in mismatches:
            print(line)
    if seg.message_trace:
        groups = Counter(
            str(ev.get("src_neuron_group_id", "?"))
            for ev in seg.message_trace
        )
        print(f"top message sources: {groups.most_common(8)}")
        cross = sum(
            1 for ev in seg.message_trace
            if int(ev.get("src_tile_id", -1)) != int(ev.get("dest_tile_id", -1))
        )
        print(f"raw cross-tile messages in flat trace: {cross}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Path to generated run folder")
    parser.add_argument(
        "reference_run_dir", type=Path, nargs="?", default=None,
        help="Optional reference run for comparison",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    _summarize(_load_segment(run_dir), run_dir.name)
    if args.reference_run_dir is not None:
        ref = args.reference_run_dir.resolve()
        _summarize(_load_segment(ref), ref.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
