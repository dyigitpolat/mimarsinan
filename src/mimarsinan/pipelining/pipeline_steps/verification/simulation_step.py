import numpy as np
import torch

from mimarsinan.certification.spike_certificate import certify_spike_counts
from mimarsinan.chip_simulation.hybrid_run.stage_timing import StageTimer
from mimarsinan.chip_simulation.simulation_runner import SimulationRunner
from mimarsinan.config_schema.registry import effective_value as _effective
from mimarsinan.deployment_record.schema import AccuracyReadRecord
from mimarsinan.models.nn.lif_kernels import measurement_plane
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.simulation_factory import (
    build_spiking_hybrid_flow,
)
from mimarsinan.pipelining.core.steps.pipeline_step import (
    METRIC_CARRIED,
    PipelineStep,
)


def _certify_nevresim_counts(*, pipeline, mapping, captured, samples):
    """[§17 Edge T'] nevresim WINDOW stage counts vs the HCM streaming
    executor on the same inputs.

    Root-caused 2026-08-09 (closes the 2026-07-21 'timing alignment' lever):
    the divergence was never independent timing — it was (a) fractional
    thresholds truncated by the chip's integer theta register (fixed: the
    NAPQ grid is integer-lattice) and (b) the stdout readout integrating
    bias through the pipeline tail cycles (fixed: lif segments consume the
    window-gated record counts). On the integer chip (weight_quantization)
    the comparison is all-integer arithmetic and is gated EXACT — FATAL on
    any mismatch. Float chips keep report-only status: knife-edge threshold
    comparisons under different float summation orders are inherent."""
    nev = {
        ("stage", k): torch.as_tensor(np.asarray(raw), dtype=torch.float64)
        for k, (_stage, raw) in enumerate(captured)
    }
    # [V9 tie class, root-caused 2026-08-11] the twin runs on the PIPELINE
    # device and inside the measurement plane, like the runner that fed the
    # chip: host ComputeOps decide exact-lattice ties (pre-activation ==
    # theta/T tread, which LSQ exact-QAT trains ONTO) by snapped values, not
    # by the GEMM dust of whatever device/batch the samples arrived on
    # (t0_05: the twin's encoder ran on CPU at batch-2 vs the runner's CUDA
    # batch-25 — one tread tie flipped and 17 windows cascaded).
    samples = samples.to(pipeline.config["device"])
    flow_captured: list = []
    flow = build_spiking_hybrid_flow(pipeline, mapping, model=None)
    flow.stage_count_recorder = (
        lambda stage, counts: flow_captured.append(counts.detach().cpu())
    )
    # nevresim executes the per-cycle chip program: the matching cell is
    # streaming, regardless of the configured metric discipline.
    flow.lif_execution_synchronized = False
    try:
        with measurement_plane(), torch.no_grad():
            flow(samples)
    finally:
        flow.stage_count_recorder = None
    hcm = {
        ("stage", k): c.to(torch.float64) for k, c in enumerate(flow_captured)
    }
    cert = certify_spike_counts(
        lambda _b: nev, lambda _b: hcm, [samples], backend="nevresim",
    )
    integer_chip = bool(DeploymentPlan.of(pipeline).weight_quantization)
    if integer_chip:
        print(
            f"[SpikeCountCertificate] nevresim windows vs HCM-streaming: "
            f"exact={cert.exact_match_fraction:.6f} "
            f"max|dcount|={cert.max_abs_delta:g} over "
            f"{cert.neuron_windows_compared} neuron-windows (integer chip — "
            f"FATAL at any mismatch)"
        )
        if cert.exact_match_fraction != 1.0 or cert.max_abs_delta != 0:
            raise AssertionError(
                f"nevresim↔HCM window-count exactness violated on the integer "
                f"chip: exact={cert.exact_match_fraction:.6f} "
                f"max|dcount|={cert.max_abs_delta:g} over "
                f"{cert.neuron_windows_compared} neuron-windows. All-integer "
                f"arithmetic admits NO tolerance — check the theta lattice "
                f"(quantize_ir_graph), the window record path, or comb drift."
            )
    else:
        print(
            f"[SpikeTransientReport] nevresim vs HCM-streaming: "
            f"exact={cert.exact_match_fraction:.6f} "
            f"max|dcount|={cert.max_abs_delta:g} over "
            f"{cert.neuron_windows_compared} neuron-windows "
            "(float chip: knife-edge threshold comparisons under different "
            "float summation orders are inherent; the decision-parity probe "
            "is this cell's arbiter)"
        )
    return cert


class SimulationStep(PipelineStep):
    REQUIRES = ("hard_core_mapping",)
    PROMISES = ("deployment_record_nevresim",)

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

    def _emit_deployment_record_fragment(
        self, runner, stage_timer, probe_accuracy: float,
    ) -> None:
        """[W4.3] persist the ``deployment_record_nevresim`` fragment: the
        probe read (schema ``AccuracyReadRecord`` shape), the driver's measured
        total-output-spikes figure (flat path; ``None`` when the run's decode
        path never produced it), and the measured host-op walls (empty when
        the program has no host ComputeOps)."""
        read = AccuracyReadRecord(
            metric=float(probe_accuracy),
            backend="nevresim",
            samples=len(runner.test_data),
            kind="measured",
            step=self.name,
        )
        self.add_entry("deployment_record_nevresim", {
            "accuracy_reads": [read.to_dict()],
            "total_spikes": runner.nevresim_total_spikes,
            "compute_stage_walls": stage_timer.compute_stage_walls(),
        }, "basic")

    def process(self):
        mapping = self.get_entry('hard_core_mapping')
        # [W4.3] host-op walls: opt into the shared stage loop's timer (the
        # nevresim run is otherwise byte-identical; flat runs simply record
        # no compute stages).
        stage_timer = StageTimer()
        runner = SimulationRunner(
            self.pipeline,
            mapping,
            int(self.pipeline.config["simulation_steps"]),
            stage_timer=stage_timer,
        )

        plan = DeploymentPlan.of(self.pipeline)
        n = int(_effective(self.pipeline.config, "spike_count_parity_samples"))
        observable, _skip = plan.mode_policy().certification_observable()
        captured: list = []
        if observable == "counts" and n > 0:
            # Raw pre-decode counts, first n samples only (Edge B cell).
            runner.stage_count_recorder = (
                lambda stage, raw: captured.append((stage, raw[:n]))
            )

        probe_accuracy = float(runner.run())
        self.probe_accuracy = probe_accuracy
        self._report_probe(probe_accuracy)

        cert_summary = None
        if captured:
            samples = torch.stack(runner.test_input[:n])
            cert = _certify_nevresim_counts(
                pipeline=self.pipeline, mapping=mapping,
                captured=captured, samples=samples,
            )
            cert_summary = cert.summary()
        self._emit_deployment_record_fragment(runner, stage_timer, probe_accuracy)
        self._verdict = {
            "status": "pass",
            "rule": "nevresim decision-parity probe (metric-neutral)",
            "detail": {
                "probe_accuracy": probe_accuracy,
                "spike_count_certificate": cert_summary,
            },
        }
