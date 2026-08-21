"""The physical ODIN backend step: deployed counts from a device, certified vs HCM."""

from __future__ import annotations

from typing import Any, Dict, List

from mimarsinan.certification.record_certificates import certify_run_records
from mimarsinan.chip_simulation.odin_fpga.factory import build_transport
from mimarsinan.chip_simulation.odin_fpga.records import (
    BACKEND_NAME,
    OdinFpgaRunRecord,
    aggregate_walls,
)
from mimarsinan.chip_simulation.odin_fpga.runner import OdinFpgaRunner
from mimarsinan.data_handling.sample_loader import load_test_samples_by_index
from mimarsinan.deployment_record.schema.accuracy import AccuracyReadRecord
from mimarsinan.mapping.platform.platform_constraints import (
    resolve_platform_mapping_params,
)
from mimarsinan.models.spiking.hybrid.carry import run_pass_transfer
from mimarsinan.pipelining.core.engine.pipeline_helpers import (
    require_spiking_mode_supported,
)
from mimarsinan.pipelining.core.simulation_factory import (
    assert_spike_parity_or_raise,
    build_deployment_contract,
    record_hcm_reference,
)
from mimarsinan.pipelining.core.steps.pipeline_step import (
    METRIC_CARRIED,
    PipelineStep,
)

#: The ODIN membrane starts every sample at zero — the CLEAR stage the exporter
#: emits rewrites exactly those bytes, and the software twins zero theirs.
MEMBRANE_INIT = 0


class OdinFpgaDeploymentStep(PipelineStep):
    """Deploy every neural segment onto ODIN cores and certify the counts."""

    REQUIRES = ("model", "hard_core_mapping", "platform_constraints_resolved")
    PROMISES = ("odin_fpga_deployment_results",)

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)
        self.metric = None
        # The per-sample device records, retained so a gate can read the
        # per-(sample, cycle, core, neuron) counts the device actually emitted
        # (the promised report carries the JSON-safe projection only).
        self.deployed_records: List[OdinFpgaRunRecord] = []

    def validate(self):
        if self.metric is not None:
            return self.metric
        return self.pipeline.get_target_metric()

    def validate_metric_kind(self) -> str:
        # self.metric is the stored previous pipeline metric (metric-neutral gate).
        return METRIC_CARRIED

    def process(self):
        self.get_entry("model")
        hard_core_mapping = self.get_entry("hard_core_mapping")
        platform = self.get_entry("platform_constraints_resolved")
        config = self.pipeline.config
        require_spiking_mode_supported(
            self.pipeline, "OdinFpgaDeploymentStep", backend=BACKEND_NAME)

        contract = build_deployment_contract(self.pipeline)
        params = resolve_platform_mapping_params(platform["cores"])
        sample_count = int(config.get("odin_fpga_sample_count", 1))
        if sample_count <= 0:
            raise ValueError("odin_fpga_sample_count must be >= 1")
        samples = load_test_samples_by_index(
            self.pipeline.data_provider_factory, range(sample_count),
            num_workers=int(config.get("num_workers", 4)))

        runner = OdinFpgaRunner(
            hard_core_mapping, contract.simulation_steps,
            contract=contract, transport=build_transport(config),
            weight_bits=int(platform.get("weight_bits", config["weight_bits"])),
            effective_max_axons=int(params.effective_max_axons),
            membrane_init=MEMBRANE_INIT,
            weight_sign_granularity=str(config["weight_sign_granularity"]),
            pass_transfer=run_pass_transfer(config),
        )

        deployed: List[OdinFpgaRunRecord] = []
        certificates: List[Any] = []
        device = config["device"]
        for index, sample in enumerate(samples):
            _flow, reference = record_hcm_reference(
                self.pipeline, hard_core_mapping, sample,
                sample_index=index, device=device)
            actual = runner.run(
                sample.detach().cpu().numpy().reshape(1, -1), sample_index=index)
            assert_spike_parity_or_raise(reference, actual.to_hcm_subset())
            certificate = certify_run_records(
                reference, actual.to_hcm_subset(), backend=BACKEND_NAME)
            print(f"[SpikeCountCertificate] {certificate.summary()}")
            if not certificate.passed:
                raise AssertionError(
                    f"{BACKEND_NAME}: {certificate.summary()} — the device did "
                    f"not reproduce the reference counts")
            certificates.append(certificate)
            deployed.append(actual)

        self.deployed_records = deployed
        walls = aggregate_walls(deployed)
        report = {
            "backend": BACKEND_NAME,
            "transport": deployed[0].transport if deployed else "none",
            "samples": len(deployed),
            "walls": walls,
            "accuracy_read": self._accuracy_fragment(len(deployed)),
            "timing_fragment": _timing_fragment(deployed, walls),
            "certificates": [c.summary() for c in certificates],
        }
        self.add_entry("odin_fpga_deployment_results", report, "pickle")

        self.metric = self.pipeline.get_target_metric()
        self.pipeline.reporter.report("ODIN FPGA Spike Parity", 1.0)
        self.pipeline.reporter.report(
            "ODIN FPGA Programming (s)", walls["programming_s"])
        self.pipeline.reporter.report(
            "ODIN FPGA Execution (s)", walls["execution_s"])
        self._verdict = {
            "status": "pass",
            "rule": "HCM vs ODIN device spike parity (exact counts)",
            "detail": {
                "samples": len(deployed),
                "transport": report["transport"],
                "programming_s": walls["programming_s"],
                "execution_s": walls["execution_s"],
                "device_cycles": walls["device_cycles"],
                "spike_count_certificate": certificates[-1].summary()
                if certificates else "",
            },
        }
        print(
            f"ODIN FPGA deployment: {len(deployed)} sample(s) on "
            f"{report['transport']!r}, programming {walls['programming_s']:.3f} s, "
            f"execution {walls['execution_s']:.3f} s, "
            f"{int(walls['device_cycles'])} device cycles")

    def _accuracy_fragment(self, samples: int) -> Dict[str, Any]:
        """The MEASURED accuracy read this backend contributes to the record."""
        return AccuracyReadRecord(
            metric=float(self.pipeline.get_target_metric()),
            backend=BACKEND_NAME, samples=int(samples), kind="measured",
            step=self.name,
        ).to_dict()


def _timing_fragment(deployed: List[OdinFpgaRunRecord], walls: Dict[str, float]
                     ) -> Dict[str, Any]:
    """The timing fragment: per-segment device walls plus the reprogram physics."""
    per_segment: List[Dict[str, Any]] = []
    for run in deployed:
        for timing in run.timings:
            per_segment.append({
                "stage_index": timing.stage_index,
                "sample_index": run.sample_index,
                "cores": timing.cores,
                "program_bytes": timing.program_bytes,
                "programming_s": timing.program_wall_s,
                "programming_basis": timing.program_basis,
                "execution_s": timing.run_wall_s,
                "device_cycles": timing.device_cycles,
            })
    return {
        "per_segment": per_segment,
        "programming_s": walls["programming_s"],
        "execution_s": walls["execution_s"],
        "host_ops_s": float(sum(
            row["wall_s_total"] for run in deployed
            for row in run.compute_stage_walls)),
        "note": (
            "measured on the declared transport; programming_s is the per-pass "
            "reprogramming cost, never folded into execution_s"
        ),
    }
