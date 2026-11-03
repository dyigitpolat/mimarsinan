"""[U-S] The opt-in event census step: the deployed program's synaptic events, measured on validation inputs."""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.chip_simulation.deployed_host import on_deployed_host
from mimarsinan.chip_simulation.spiking_semantics import is_cascaded_ttfs
from mimarsinan.chip_simulation.synaptic_events import EventCensusRefusal, census_by_unit
from mimarsinan.data_handling.sample_loader import load_validation_samples_by_index
from mimarsinan.mapping.crossbar_utilization import CrossbarUtilizationReport
from mimarsinan.models.nn.lif_kernels import measurement_plane
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.simulation_factory import build_spiking_hybrid_flow
from mimarsinan.pipelining.core.steps.pipeline_step import METRIC_CARRIED, PipelineStep

STEP_NAME = "Event Census"
CENSUS_SPLIT = "validation"


def require_recorded_spikes(plan) -> None:
    """The census reads the per-cycle executor's spike record (LIF, cascaded TTFS); other semantics refuse by name."""
    if plan.spiking_mode == "lif" or is_cascaded_ttfs(plan.spiking_mode, plan.ttfs_cycle_schedule):
        return
    raise EventCensusRefusal(
        f"event census declared (event_census_samples={plan.event_census_samples}) on spiking_mode "
        f"{plan.spiking_mode!r} under core semantics {plan.core_semantics!r}: only the per-cycle executor records "
        "spikes (lif, cascaded ttfs_cycle_based); a value-domain program's events are its static MAC census")


class EventCensusStep(PipelineStep):
    """Run the first ``event_census_samples`` validation samples through the deployed program on the Hard Core
    Mapping executor, one sample at a time (its recorder runs at batch 1), and persist each sample's synaptic-event
    census (``census_by_unit``) with their mean per inference, which the Deployment Record step seals as
    ``EnergyRecord.synaptic_events``. Never the test split; metric-neutral."""

    REQUIRES = ("model", "hard_core_mapping")
    PROMISES = ("event_census",)

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)
        require_recorded_spikes(DeploymentPlan.of(pipeline))
        self.metric = None

    @classmethod
    def applies_to(cls, plan) -> bool:
        return int(plan.event_census_samples) > 0

    def validate(self):
        return self.metric if self.metric is not None else self.pipeline.get_target_metric()

    def validate_metric_kind(self) -> str:
        return METRIC_CARRIED

    def process(self):
        plan = DeploymentPlan.of(self.pipeline)
        model = self.get_entry("model")
        mapping = self.get_entry("hard_core_mapping")
        samples = load_validation_samples_by_index(
            self.pipeline.data_provider_factory, range(int(plan.event_census_samples)),
            num_workers=int(self.pipeline.config.get("num_workers", 4)))
        # [U-HDA] the deployed program on the deployed host (the CPU, where the deployed read runs).
        with on_deployed_host("Event Census (the deployed program's spiking executor)", self.pipeline.config["device"],
                              programs=(mapping,), model=model,
                              reporter=getattr(self.pipeline, "reporter", None)) as host:
            flow = build_spiking_hybrid_flow(
                self.pipeline, mapping, preprocessor=getattr(model, "preprocessor", None)).to(host.device).eval()
            with measurement_plane(), torch.no_grad():
                _out, records = flow.forward_with_recordings(torch.cat(samples, dim=0).to(host.device))
        by_unit = [census_by_unit(record, mapping) for record in records]
        per_sample = [int(sum(units.values())) for units in by_unit]
        mean = float(np.mean(per_sample))
        steps = int(flow.simulation_length)
        cells = int(CrossbarUtilizationReport.from_hybrid_mapping(mapping).cells_used)
        census = {
            "samples": len(per_sample), "split": CENSUS_SPLIT, "simulation_steps": steps,
            "spiking_mode": plan.spiking_mode, "per_sample": per_sample, "mean_per_inference": mean,
            "per_unit_mean": {str(i): float(np.mean([u[i] for u in by_unit])) for i in sorted(by_unit[0])},
            "cells_used": cells, "activity": mean / (cells * steps) if cells * steps > 0 else None,
        }
        self.add_entry("event_census", census, "basic")
        print(f"[EventCensusStep] {len(per_sample)} {CENSUS_SPLIT} sample(s): {mean:.6g} synaptic events per "
              f"inference over {cells} used cells at T={steps} (activity {census['activity']})")
        self.metric = self.pipeline.get_target_metric()
        self._verdict = {"status": "pass", "rule": "event census measured on validation inputs",
                         "detail": {k: census[k] for k in ("samples", "mean_per_inference", "activity")}}
