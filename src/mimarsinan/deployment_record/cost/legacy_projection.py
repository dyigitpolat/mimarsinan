"""Legacy ``cost_record.json`` continuity: a format-v3 projection of the record.

THE CONTINUITY CONTRACT (docs/deployment_record_schema.md §6): the terminal
Deployment Record step writes ``cost_record.json`` as a PROJECTION of the
sealed record, same directory, same SANA-FE conditional as the deleted
``SanafeSimulationStep._emit_cost_record``. Live fields are value-identical
to ``extract_cost_record`` on the same SANA-FE snapshot (pinned field-by-field
by the golden-fixture test); the formerly-dead fields — ``reprogram_passes``,
``reuse_passes``, ``params_reloaded``, ``max_ft_pass_wall_s``,
``ft_pass_walls`` — go LIVE from the record's schedule/adaptation fragments.

``activation_bytes_moved`` stays 0 deliberately: no producer exists anywhere
in the codebase; activation movement is modeled via the record's traffic
terms instead (§6 disposition, owner-signed).
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from mimarsinan.chip_simulation.certification import CertificationCell
from mimarsinan.chip_simulation.cost_extraction import CostRecord
from mimarsinan.deployment_record.schema import DeploymentRecord


def cost_record_from_deployment_record(record: DeploymentRecord) -> CostRecord:
    """Project a sealed :class:`DeploymentRecord` onto the legacy ``CostRecord``.

    Requires the record's SANA-FE-backed fragments (``energy`` + measured
    ``timing.per_segment``): a record without them has no legacy cost surface —
    exactly the runs that never wrote ``cost_record.json`` before.
    """
    energy = record.energy
    if energy is None:
        raise ValueError(
            "cost_record projection requires the energy fragment "
            "(SANA-FE disabled runs never wrote cost_record.json)"
        )
    timing = record.timing
    if not timing.per_segment:
        raise ValueError(
            "cost_record projection requires measured timing.per_segment "
            "(the SANA-FE per-segment census)"
        )
    identity = record.identity
    cell = CertificationCell.from_key(identity.cell_key)
    schedule = record.schedule
    cores = sum(len(segment.cores) for segment in schedule.segments())
    latency_steps = sum(
        segment.timesteps_executed for segment in timing.per_segment
    )
    max_ft_pass_wall_s, ft_pass_walls = _adaptation_walls(record)
    return CostRecord(
        cell_key=identity.cell_key,
        mode=identity.mode,
        backend=cell.backend,
        acc_deploy=float(record.accuracy.deployed.metric),
        mj_per_sample=float(energy.mj_per_sample),
        spikes=int(energy.total_spikes),
        latency_steps=int(latency_steps),
        cores=int(cores),
        s_global=int(timing.s_global),
        depth=int(timing.depth),
        energy_proxy_neuron_steps=int(energy.energy_proxy_neuron_steps),
        max_ft_pass_wall_s=max_ft_pass_wall_s,
        ft_pass_walls=ft_pass_walls,
        reprogram_passes=int(schedule.reprogram_passes),
        reuse_passes=int(schedule.reuse_passes),
        params_reloaded=int(schedule.params_reloaded),
        # No producer exists (§6): stays 0, modeled via traffic terms instead.
        activation_bytes_moved=0,
        provenance={"run_dir": identity.run_dir},
    )


def _adaptation_walls(
    record: DeploymentRecord,
) -> Tuple[float, Tuple[Mapping[str, Any], ...]]:
    """The FT-pass wall bundle in the exact legacy ``ft_pass_walls.json`` shape."""
    adaptation = record.adaptation
    if adaptation is None:
        return 0.0, ()
    return (
        float(adaptation.max_ft_pass_wall_s),
        tuple(
            {"label": wall.label, "wall_s": wall.wall_s}
            for wall in adaptation.ft_pass_walls
        ),
    )
