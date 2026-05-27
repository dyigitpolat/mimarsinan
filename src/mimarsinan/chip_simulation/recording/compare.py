"""Spike record comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from mimarsinan.chip_simulation.recording.records import CoreSpikeCounts, RunRecord, SegmentSpikeRecord

# Diffing


@dataclass
class Diff:
    """A single comparison failure between a reference and an actual record."""

    stage_index: int
    stage_name: str
    schedule_segment_index: Optional[int]
    schedule_pass_index: Optional[int]
    layer: str       # one of: "seg_input", "core_input", "core_output", "seg_output"
    core_index: Optional[int]
    has_hardware_bias: Optional[bool]
    n_always_on_axons: Optional[int]
    core_latency: Optional[int]
    expected: np.ndarray
    actual: np.ndarray
    suggested_cause: str


_LAYER_TO_CAUSE = {
    "seg_input": "encoding drift (uniform rate encoder mismatch)",
    "core_input": "axon span / routing wiring (or upstream output diverged)",
    "core_output": "LIF dynamics: threshold, weights, hardware bias, or reset semantics",
    "seg_output": "output_sources gather (compress_spike_sources path)",
}


def _diff(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    seg: SegmentSpikeRecord,
    layer: str,
    core: Optional[CoreSpikeCounts] = None,
) -> Optional[Diff]:
    """Return a ``Diff`` if ``expected`` and ``actual`` disagree, else None."""
    if expected.shape != actual.shape:
        # Shape mismatch is a structural bug, not a count bug.  Surface it
        # with a synthetic empty-array diff so the report still names the
        # site clearly.
        return Diff(
            stage_index=seg.stage_index,
            stage_name=seg.stage_name,
            schedule_segment_index=seg.schedule_segment_index,
            schedule_pass_index=seg.schedule_pass_index,
            layer=layer,
            core_index=core.core_index if core is not None else None,
            has_hardware_bias=core.has_hardware_bias if core is not None else None,
            n_always_on_axons=core.n_always_on_axons if core is not None else None,
            core_latency=core.core_latency if core is not None else None,
            expected=np.asarray(expected.shape),
            actual=np.asarray(actual.shape),
            suggested_cause=f"shape mismatch — {_LAYER_TO_CAUSE.get(layer, layer)}",
        )
    if np.array_equal(expected, actual):
        return None
    return Diff(
        stage_index=seg.stage_index,
        stage_name=seg.stage_name,
        schedule_segment_index=seg.schedule_segment_index,
        schedule_pass_index=seg.schedule_pass_index,
        layer=layer,
        core_index=core.core_index if core is not None else None,
        has_hardware_bias=core.has_hardware_bias if core is not None else None,
        n_always_on_axons=core.n_always_on_axons if core is not None else None,
        core_latency=core.core_latency if core is not None else None,
        expected=expected,
        actual=actual,
        suggested_cause=_LAYER_TO_CAUSE.get(layer, layer),
    )


def compare_records(ref: RunRecord, actual: RunRecord) -> List[Diff]:
    """Walk every segment in stage order, comparing in this layer order:

      1. ``seg_input_spike_count`` — encoder/uniform-rate disagreement.
      2. Per-core ``input_spike_count`` — axon span/routing/always-on bug.
      3. Per-core ``output_spike_count`` — LIF dynamics, threshold,
         weights, hardware bias.
      4. ``seg_output_spike_count`` — output_sources gather bug.

    Returns the list of all diffs found, in the order they were
    detected.  An empty list means the two records agree exactly.
    """
    diffs: List[Diff] = []

    if ref.T != actual.T:
        # T mismatch is a global config bug; reported once, not per-stage.
        diffs.append(
            Diff(
                stage_index=-1, stage_name="<root>",
                schedule_segment_index=None, schedule_pass_index=None,
                layer="T",
                core_index=None, has_hardware_bias=None,
                n_always_on_axons=None, core_latency=None,
                expected=np.asarray(ref.T), actual=np.asarray(actual.T),
                suggested_cause="simulation_length mismatch between reference and actual",
            )
        )
        return diffs

    common = sorted(set(ref.segments.keys()) & set(actual.segments.keys()))
    only_ref = set(ref.segments.keys()) - set(actual.segments.keys())
    only_act = set(actual.segments.keys()) - set(ref.segments.keys())
    if only_ref or only_act:
        diffs.append(
            Diff(
                stage_index=-1, stage_name="<root>",
                schedule_segment_index=None, schedule_pass_index=None,
                layer="segment_set",
                core_index=None, has_hardware_bias=None,
                n_always_on_axons=None, core_latency=None,
                expected=np.asarray(sorted(only_ref), dtype=np.int64),
                actual=np.asarray(sorted(only_act), dtype=np.int64),
                suggested_cause="segments present in one record but missing in the other",
            )
        )

    for stage_index in common:
        rseg = ref.segments[stage_index]
        aseg = actual.segments[stage_index]

        d = _diff(rseg.seg_input_spike_count, aseg.seg_input_spike_count,
                  seg=rseg, layer="seg_input")
        if d is not None:
            diffs.append(d)

        # Walk cores in the segment.  Reference and actual must enumerate
        # cores in the same allocation order — they share the mapping.
        if len(rseg.cores) != len(aseg.cores):
            diffs.append(
                Diff(
                    stage_index=stage_index, stage_name=rseg.stage_name,
                    schedule_segment_index=rseg.schedule_segment_index,
                    schedule_pass_index=rseg.schedule_pass_index,
                    layer="core_count",
                    core_index=None, has_hardware_bias=None,
                    n_always_on_axons=None, core_latency=None,
                    expected=np.asarray(len(rseg.cores)),
                    actual=np.asarray(len(aseg.cores)),
                    suggested_cause="core count mismatch — recorder built different segment views",
                )
            )
        else:
            for rcore, acore in zip(rseg.cores, aseg.cores):
                d = _diff(rcore.input_spike_count, acore.input_spike_count,
                          seg=rseg, layer="core_input", core=rcore)
                if d is not None:
                    diffs.append(d)
                d = _diff(rcore.output_spike_count, acore.output_spike_count,
                          seg=rseg, layer="core_output", core=rcore)
                if d is not None:
                    diffs.append(d)

        d = _diff(rseg.seg_output_spike_count, aseg.seg_output_spike_count,
                  seg=rseg, layer="seg_output")
        if d is not None:
            diffs.append(d)

    return diffs


def format_first_diff(diffs: List[Diff]) -> str:
    """Pretty-print the first diff with enough context for triage."""
    if not diffs:
        return "no diffs"

    d = diffs[0]
    parts = [
        "",
        f"  stage           : {d.stage_index} ({d.stage_name!r})",
    ]
    if d.schedule_segment_index is not None:
        parts.append(
            f"  schedule        : segment={d.schedule_segment_index} "
            f"pass={d.schedule_pass_index}"
        )
    parts.append(f"  layer           : {d.layer}")
    if d.core_index is not None:
        parts.append(f"  core            : index={d.core_index} latency={d.core_latency}")
        parts.append(
            f"  bias / on-axons : has_hardware_bias={d.has_hardware_bias} "
            f"n_always_on_axons={d.n_always_on_axons}"
        )
    parts.append(f"  suggested cause : {d.suggested_cause}")

    exp = d.expected
    act = d.actual
    if exp.shape == act.shape and exp.ndim <= 1:
        n_total = int(exp.size)
        diff_mask = exp != act
        n_diff = int(diff_mask.sum())
        first_idx = int(np.argmax(diff_mask)) if n_diff > 0 else -1
        parts.append(
            f"  divergence      : {n_diff}/{n_total} positions differ; "
            f"first at index {first_idx}"
        )
        if first_idx >= 0:
            parts.append(
                f"    expected[{first_idx}]={exp[first_idx]}  "
                f"actual[{first_idx}]={act[first_idx]}"
            )
            # Also show the absolute totals so we know whether it's a
            # systematic over/undercount or a localised flip.
            parts.append(
                f"    Σ expected={int(np.asarray(exp).sum())}  "
                f"Σ actual={int(np.asarray(act).sum())}"
            )
    else:
        parts.append(f"  expected shape  : {tuple(exp.shape)}")
        parts.append(f"  actual shape    : {tuple(act.shape)}")

    if len(diffs) > 1:
        parts.append(f"  (+{len(diffs) - 1} more diffs not shown)")

    return "\n".join(parts)
