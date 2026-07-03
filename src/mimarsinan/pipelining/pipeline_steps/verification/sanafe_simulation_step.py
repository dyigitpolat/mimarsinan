"""Optional SANA-FE detailed-stats + HCM spike-parity step."""

from __future__ import annotations

import logging
from typing import List

from mimarsinan.chip_simulation.certification import CertificationCell
from mimarsinan.chip_simulation.cost_extraction import (
    extract_cost_record,
    save_cost_record,
)
from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner
from mimarsinan.chip_simulation.sanafe.records import SanafeCoreDiff, SanafeRunRecord
from mimarsinan.chip_simulation.sanafe.stats import SanafeStepReport
from mimarsinan.chip_simulation.spiking_semantics import requires_ttfs_firing
from mimarsinan.chip_simulation.ttfs.ttfs_recorder import (
    compare_ttfs_contract_records,
    compare_ttfs_hardware_records,
    format_first_ttfs_diff,
)

logger = logging.getLogger("mimarsinan.chip_simulation")


def _attach_per_core_deltas(ref: object, sanafe_rec: SanafeRunRecord) -> None:
    """Stamp per-core HCM↔SF deltas on each segment so the GUI floorplan diff overlay can render spatial drift."""
    ref_segs = getattr(ref, "segments", {}) or {}
    for stage_idx, seg in sanafe_rec.segments.items():
        ref_seg = ref_segs.get(stage_idx)
        if ref_seg is None or not ref_seg.cores or not seg.per_core:
            continue
        ref_by_idx = {c.core_index: c for c in ref_seg.cores}
        deltas: List[SanafeCoreDiff] = []
        for sf in seg.per_core:
            ref_core = ref_by_idx.get(sf.core_index)
            if ref_core is None:
                continue
            in_sum_ref = int(ref_core.input_spike_count.sum())
            in_sum_sf = int(sf.input_spike_count.sum())
            out_sum_ref = int(ref_core.output_spike_count.sum())
            out_sum_sf = int(sf.output_spike_count.sum())
            deltas.append(SanafeCoreDiff(
                core_index=int(sf.core_index),
                # SF − HCM: positive = SF over-reports, negative = under.
                input_delta_sum=in_sum_sf - in_sum_ref,
                output_delta_sum=out_sum_sf - out_sum_ref,
            ))
        seg.hcm_diff = deltas
from mimarsinan.data_handling.test_sample_loader import load_test_samples_by_index
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.engine.pipeline_helpers import (
    require_spiking_mode_supported,
)
from mimarsinan.pipelining.core.steps.pipeline_step import PipelineStep
from mimarsinan.pipelining.core.simulation_factory import (
    assert_spike_parity_or_raise,
    build_deployment_contract,
    preprocess_hybrid_sample,
    record_hcm_reference,
    record_ttfs_hcm_reference,
)


class SanafeSimulationStep(PipelineStep):
    """Run SANA-FE on N deterministic samples; collect rich stats + parity-check."""

    REQUIRES = ("model", "hard_core_mapping")
    PROMISES = ("sanafe_simulation_results",)

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)
        self.metric = None

    def validate(self):
        if self.metric is not None:
            return self.metric
        return self.pipeline.get_target_metric()


    def process(self):
        self.get_entry("model")
        hard_core_mapping = self.get_entry("hard_core_mapping")
        T = int(self.pipeline.config["simulation_steps"])
        spiking_mode = DeploymentPlan.of(self.pipeline).spiking_mode
        require_spiking_mode_supported(
            self.pipeline, "SanafeSimulationStep", backend="sanafe",
        )
        is_ttfs = requires_ttfs_firing(spiking_mode)

        parity_check = True
        arch_preset = self.pipeline.config.get("sanafe_arch_preset", "loihi")
        custom_arch_path = self.pipeline.config.get("sanafe_custom_arch_path") or None
        log_potential = bool(self.pipeline.config.get("sanafe_log_potential_trace", False))
        log_messages = True
        device = self.pipeline.config["device"]

        n = int(self.pipeline.config.get("sanafe_sample_count", 1))
        if n <= 0:
            raise ValueError("sanafe_sample_count must be >= 1")
        samples = load_test_samples_by_index(
            self.pipeline.data_provider_factory,
            range(n),
            num_workers=int(self.pipeline.config.get("num_workers", 4)),
        )
        per_sample: list[SanafeRunRecord] = []

        for sample_idx, sample in enumerate(samples):
            ref = None
            ttfs_ref = None
            if parity_check:
                if is_ttfs:
                    _flow, ttfs_ref = record_ttfs_hcm_reference(
                        self.pipeline,
                        hard_core_mapping,
                        sample,
                        sample_index=sample_idx,
                    )
                else:
                    _flow, ref = record_hcm_reference(
                        self.pipeline,
                        hard_core_mapping,
                        sample,
                        sample_index=sample_idx,
                        device=device,
                    )

            contract = build_deployment_contract(self.pipeline)
            runner = SanafeRunner(
                mapping=hard_core_mapping,
                simulation_length=T,
                contract=contract,
                arch_preset=arch_preset,
                custom_arch_path=custom_arch_path,
                log_potential_trace=log_potential,
                log_message_trace=log_messages,
            )
            if is_ttfs:
                sample_np = preprocess_hybrid_sample(
                    self.pipeline, hard_core_mapping, sample, device=device,
                )
            else:
                sample_np = sample.detach().cpu().numpy().reshape(1, -1)
            sanafe_rec = runner.run(sample_np, sample_index=sample_idx)
            per_sample.append(sanafe_rec)

            if ref is not None:
                _attach_per_core_deltas(ref, sanafe_rec)

            if parity_check and is_ttfs and ttfs_ref is not None:
                contract_diffs = compare_ttfs_contract_records(
                    ttfs_ref,
                    sanafe_rec.to_ttfs_contract_subset(spiking_mode=spiking_mode),
                )
                if contract_diffs:
                    raise AssertionError(
                        format_first_ttfs_diff(contract_diffs, layer="contract"),
                    )
                hw_diffs = compare_ttfs_hardware_records(
                    ttfs_ref,
                    sanafe_rec.to_ttfs_hardware_subset(spiking_mode=spiking_mode),
                )
                if hw_diffs:
                    raise AssertionError(
                        format_first_ttfs_diff(hw_diffs, layer="hardware"),
                    )
            elif parity_check and ref is not None:
                assert_spike_parity_or_raise(ref, sanafe_rec.to_hcm_subset())

        report = SanafeStepReport.from_records(arch_preset, per_sample)
        self.add_entry("sanafe_simulation_results", report, "pickle")

        if parity_check:
            self.pipeline.reporter.report("SANA-FE Parity", 1.0)
        self.pipeline.reporter.report(
            "SANA-FE Total Energy (mJ)", report.aggregate["total_energy_mj"]
        )
        self.pipeline.reporter.report(
            "SANA-FE Max Sim Time (s)", report.aggregate["max_sim_time_s"]
        )
        self.pipeline.reporter.report(
            "SANA-FE Total Spikes", report.aggregate["total_spikes"]
        )
        self.pipeline.reporter.report(
            "SANA-FE Total NoC Packets", report.aggregate["total_packets"]
        )

        self.metric = self.pipeline.get_target_metric()
        print(
            f"SANA-FE simulation: {len(per_sample)} sample(s), "
            f"E={report.aggregate['total_energy_mj']:.3f} mJ, "
            f"t_sim_max={report.aggregate['max_sim_time_s']:.3e} s, "
            f"{report.aggregate['total_spikes']} spikes, "
            f"{report.aggregate['total_packets']} packets"
        )

        self._emit_cost_record(report)

    def _emit_cost_record(self, report: SanafeStepReport) -> None:
        """Drop a measured ``cost_record.json`` next to the run artifacts.

        Pure additive side-effect, exception-isolated: any failure is logged and
        swallowed so cost emission can never crash or alter the deployment.
        """
        try:
            plan = DeploymentPlan.of(self.pipeline)
            cell = CertificationCell.from_mode_policy(
                plan.mode_policy(), backend="sanafe",
            )
            record = extract_cost_record(
                cell=cell,
                deployed_accuracy=float(self.pipeline.get_target_metric()),
                sanafe_snapshot=report.to_snapshot_dict(),
                provenance={"run_dir": self.pipeline.working_directory},
            )
            save_cost_record(record, self.pipeline.working_directory)
        except Exception:
            logger.exception("SANA-FE cost-record emission failed; skipping")
