"""THE continuity golden test (schema §6): projection ≡ legacy extraction.

A realistic ``SanafeStepReport``-shaped snapshot is fed to BOTH surfaces —
the legacy ``extract_cost_record`` and the record path (``from_simulators``
converters → ``DeploymentRecord`` → ``cost_record_from_deployment_record``) —
and every LIVE ``CostRecord`` field is pinned equal, field by field. The
formerly-dead fields (``reprogram_passes``, ``reuse_passes``,
``params_reloaded``, ``max_ft_pass_wall_s``, ``ft_pass_walls``) go live from
the record's schedule/adaptation fragments; ``activation_bytes_moved`` stays 0
(no producer exists — the §6 stated disposition).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from unit.deployment_record.record_fixtures import (
    make_full_record,
    make_sanafe_snapshot,
)

from mimarsinan.chip_simulation.certification import CertificationCell
from mimarsinan.chip_simulation.cost_extraction import (
    COST_RECORD_FORMAT_VERSION,
    extract_cost_record,
)
from mimarsinan.deployment_record.build.from_simulators import (
    depth_from_sanafe,
    energy_record_from_sanafe,
    s_global_from_sanafe,
    segment_timings_from_sanafe,
)
from mimarsinan.deployment_record.cost import cost_record_from_deployment_record
from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_assembly import (
    timing_record,
)

RUN_DIR = "/runs/t_continuity"
DEPLOYED_ACCURACY = 0.97  # the fixture record's deployed read metric

make_snapshot = make_sanafe_snapshot


def make_record_from_snapshot(snapshot: dict):
    """The record path: converters + fixture fragments (consistent census)."""
    base = make_full_record()
    return replace(
        base,
        identity=replace(base.identity, run_dir=RUN_DIR),
        energy=energy_record_from_sanafe(snapshot),
        timing=timing_record(
            s_global=s_global_from_sanafe(snapshot),
            depth=depth_from_sanafe(snapshot),
            per_segment=segment_timings_from_sanafe(snapshot),
            host_ops_s=None,
        ),
    )


def _legacy(snapshot: dict, **kwargs):
    cell = CertificationCell.from_key("lif_streamed@sanafe")
    return extract_cost_record(
        cell=cell,
        deployed_accuracy=DEPLOYED_ACCURACY,
        sanafe_snapshot=snapshot,
        provenance={"run_dir": RUN_DIR},
        **kwargs,
    )


LIVE_FIELDS = (
    "cell_key", "mode", "backend", "acc_deploy", "mj_per_sample", "spikes",
    "latency_steps", "cores", "s_global", "depth",
    "energy_proxy_neuron_steps", "activation_bytes_moved", "provenance",
    "format_version",
)


class TestContinuityGolden:
    def test_live_fields_value_identical_field_by_field(self):
        snapshot = make_snapshot()
        legacy = _legacy(snapshot)
        projected = cost_record_from_deployment_record(
            make_record_from_snapshot(snapshot)
        )
        for field in LIVE_FIELDS:
            assert getattr(projected, field) == getattr(legacy, field), field
        # Pin the actual numbers so the fixture can never degenerate silently.
        assert projected.mj_per_sample == 2.0  # 4.0 mJ over 2 samples
        assert projected.spikes == 987
        assert projected.latency_steps == 64
        assert projected.cores == 3
        assert projected.s_global == 32
        assert projected.depth == 2
        assert projected.energy_proxy_neuron_steps == (40 + 25) * 32 + 30 * 32
        assert projected.provenance == {"run_dir": RUN_DIR}
        assert projected.format_version == COST_RECORD_FORMAT_VERSION

    def test_single_sample_keeps_legacy_division_rule(self):
        snapshot = make_snapshot(sample_count=1)
        legacy = _legacy(snapshot)
        projected = cost_record_from_deployment_record(
            make_record_from_snapshot(snapshot)
        )
        assert legacy.mj_per_sample == 4.0  # sample_count 1: NO division
        assert projected.mj_per_sample == legacy.mj_per_sample

    def test_dead_fields_now_live_from_the_record(self):
        snapshot = make_snapshot()
        legacy = _legacy(snapshot)
        record = make_record_from_snapshot(snapshot)
        projected = cost_record_from_deployment_record(record)
        # Legacy emission left these dead (defaults).
        assert (legacy.reprogram_passes, legacy.reuse_passes,
                legacy.params_reloaded) == (0, 0, 0)
        assert legacy.max_ft_pass_wall_s == 0.0
        assert legacy.ft_pass_walls == ()
        # The projection populates them from the record's fragments.
        assert projected.reprogram_passes == record.schedule.reprogram_passes == 1
        assert projected.reuse_passes == record.schedule.reuse_passes == 1
        assert projected.params_reloaded == record.schedule.params_reloaded == 100
        adaptation = record.adaptation
        assert adaptation is not None
        assert projected.max_ft_pass_wall_s == adaptation.max_ft_pass_wall_s == 12.5
        assert projected.ft_pass_walls == tuple(
            {"label": w.label, "wall_s": w.wall_s}
            for w in adaptation.ft_pass_walls
        )
        # Stated disposition: no producer exists anywhere in the codebase.
        assert projected.activation_bytes_moved == 0

    def test_full_dict_identical_when_legacy_gets_the_live_inputs(self):
        """extract_cost_record with the record's figures == projection, byte-for-byte."""
        snapshot = make_snapshot()
        record = make_record_from_snapshot(snapshot)
        adaptation = record.adaptation
        assert adaptation is not None
        legacy = _legacy(
            snapshot,
            reprogram_passes=record.schedule.reprogram_passes,
            reuse_passes=record.schedule.reuse_passes,
            params_reloaded=record.schedule.params_reloaded,
            max_ft_pass_wall_s=adaptation.max_ft_pass_wall_s,
            ft_pass_walls=[
                {"label": w.label, "wall_s": w.wall_s}
                for w in adaptation.ft_pass_walls
            ],
        )
        projected = cost_record_from_deployment_record(record)
        assert projected.to_dict() == legacy.to_dict()

    def test_projection_refuses_a_record_without_sanafe_fragments(self):
        snapshot = make_snapshot()
        record = make_record_from_snapshot(snapshot)
        with pytest.raises(ValueError, match="energy fragment"):
            cost_record_from_deployment_record(replace(record, energy=None))
        no_segments = replace(
            record, timing=replace(record.timing, per_segment=()),
        )
        with pytest.raises(ValueError, match="per_segment"):
            cost_record_from_deployment_record(no_segments)


def test_energy_breakdown_terms_are_joules_converted_to_mj():
    # Pins the J -> mJ conversion on per-plane breakdown terms: the snapshot
    # carries energy_breakdown_j in JOULES; EnergyTermRecord.mj must be *1e3.
    snapshot = make_sanafe_snapshot()
    record = energy_record_from_sanafe(snapshot)
    breakdown_j = snapshot["aggregate"]["energy_breakdown_j"]
    assert breakdown_j, "fixture must carry a non-empty breakdown"
    from mimarsinan.deployment_record.build.from_simulators import (
        _ENERGY_BREAKDOWN_TERMS,
    )
    by_name = {t.name: t for t in record.breakdown}
    pinned = [t for t in _ENERGY_BREAKDOWN_TERMS if t in breakdown_j]
    assert pinned, "fixture must cover at least one converter term"
    for term in pinned:
        rec = by_name[f"sanafe_{term}"]
        assert rec.mj == pytest.approx(float(breakdown_j[term]) * 1000.0)
        assert rec.kind == "measured"
