import numpy as np
import torch

from mimarsinan.certification.spike_certificate import certify_spike_counts
from mimarsinan.chip_simulation.simulation_runner import SimulationRunner
from mimarsinan.chip_simulation.spiking_semantics import is_lif
from mimarsinan.config_schema.registry import effective_value as _effective
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.simulation_factory import (
    build_spiking_hybrid_flow,
)
from mimarsinan.pipelining.core.steps.pipeline_step import (
    METRIC_CARRIED,
    PipelineStep,
)


def _certify_nevresim_counts(*, pipeline, mapping, captured, samples):
    """[§17 Edge B] nevresim raw stage counts vs the HCM streaming executor on
    the same inputs — matching-discipline chip-math cell, atol from the
    ``nevresim`` backend class; fatal on failure."""
    nev = {
        ("stage", k): torch.as_tensor(np.asarray(raw), dtype=torch.float64)
        for k, (_stage, raw) in enumerate(captured)
    }
    flow_captured: list = []
    flow = build_spiking_hybrid_flow(pipeline, mapping, model=None)
    flow.stage_count_recorder = (
        lambda stage, counts: flow_captured.append(counts.detach().cpu())
    )
    # nevresim executes the per-cycle chip program: the matching cell is
    # streaming, regardless of the configured metric discipline.
    flow.lif_execution_synchronized = False
    try:
        with torch.no_grad():
            flow(samples)
    finally:
        flow.stage_count_recorder = None
    hcm = {
        ("stage", k): c.to(torch.float64) for k, c in enumerate(flow_captured)
    }
    cert = certify_spike_counts(
        lambda _b: nev, lambda _b: hcm, [samples], backend="nevresim",
    )
    print(f"[SpikeCountCertificate] nevresim/streaming: {cert.summary()}")
    for dv in cert.divergent:
        print(f"[SpikeCountCertificate] nevresim divergent {dv}")
    if not cert.passed:
        raise RuntimeError(
            f"nevresim spike-count certificate FAILED: {cert.summary()}"
        )
    return cert


class SimulationStep(PipelineStep):
    REQUIRES = ("hard_core_mapping",)

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

        self.probe_accuracy = None

    def validate(self):
        # nevresim is a decision-parity PROBE on a small subsample; its
        # binomial-noise accuracy is reported, never the pipeline metric —
        # the accuracy verdict is the SCM identity read (retention-gated
        # there). Loihi and SANA-FE follow the same metric-neutral contract.
        return self.pipeline.get_target_metric()

    def validate_metric_kind(self) -> str:
        return METRIC_CARRIED

    def _report_probe(self, accuracy) -> None:
        print("Simulation accuracy: ", accuracy)
        self.pipeline.reporter.report("nevresim_probe_accuracy", float(accuracy))

    def process(self):
        mapping = self.get_entry('hard_core_mapping')
        runner = SimulationRunner(
            self.pipeline,
            mapping,
            int(self.pipeline.config["simulation_steps"]),
        )

        plan = DeploymentPlan.of(self.pipeline)
        n = int(_effective(self.pipeline.config, "spike_count_parity_samples"))
        captured: list = []
        if is_lif(str(plan.spiking_mode)) and n > 0:
            # Raw pre-decode counts, first n samples only (Edge B cell).
            runner.stage_count_recorder = (
                lambda stage, raw: captured.append((stage, raw[:n]))
            )

        self.probe_accuracy = runner.run()
        self._report_probe(self.probe_accuracy)

        cert_summary = None
        if captured:
            samples = torch.stack(runner.test_input[:n])
            cert = _certify_nevresim_counts(
                pipeline=self.pipeline, mapping=mapping,
                captured=captured, samples=samples,
            )
            cert_summary = cert.summary()
        self._verdict = {
            "status": "pass",
            "rule": "nevresim decision-parity probe (metric-neutral)",
            "detail": {
                "probe_accuracy": float(self.probe_accuracy),
                "spike_count_certificate": cert_summary,
            },
        }
