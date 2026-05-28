"""Optional SANA-FE detailed-stats + hard-parity step.

Added to the pipeline when ``enable_sanafe_simulation`` is true in
``deployment_parameters``.  Unlike ``LoihiSimulationStep`` (parity-only),
this step's primary value is the rich per-tile, per-core, per-neuron
output SANA-FE provides — energy decomposition, latency, NoC packet
traces, spike + potential traces.

When ``sanafe_parity_check`` (default ``True``) is on, the step also
builds an HCM reference for each sample and compares spike counts via
the shared ``compare_records`` diff machinery; any divergence fails the
pipeline with ``format_first_diff`` for triage.
"""

from __future__ import annotations

from typing import List

import torch

from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner
from mimarsinan.chip_simulation.sanafe.records import SanafeCoreDiff, SanafeRunRecord
from mimarsinan.chip_simulation.sanafe.stats import SanafeStepReport


def _attach_per_core_deltas(ref: object, sanafe_rec: SanafeRunRecord) -> None:
    """Stamp per-core HCM↔SF deltas on each segment of ``sanafe_rec``.

    The GUI floorplan "HCM diff" overlay reads ``seg.hcm_diff`` and
    renders a diverging-colormap layer over the chip floorplan, so
    spatial drift is visible at a glance — even on runs that pass the
    parity gate (where all deltas are zero, the overlay just renders
    transparent everywhere).  Skips segments that don't appear in both
    records (no common ground to diff against).
    """
    ref_segs = getattr(ref, "segments", {}) or {}
    for stage_idx, seg in sanafe_rec.segments.items():
        ref_seg = ref_segs.get(stage_idx)
        if ref_seg is None or not ref_seg.cores or not seg.per_core:
            continue
        # Align by ``core_index`` so split / coalesced cores still match.
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
from mimarsinan.pipelining.core.engine.pipeline_helpers import (
    require_lif_spiking_mode,
    require_spiking_mode_supported,
)
from mimarsinan.pipelining.core.steps.pipeline_step import PipelineStep
from mimarsinan.pipelining.core.simulation_factory import (
    assert_spike_parity_or_raise,
    build_neural_behavior_config,
    preprocess_hybrid_sample,
    record_hcm_reference,
    record_ttfs_hcm_reference,
)


class SanafeSimulationStep(PipelineStep):
    """Run SANA-FE on N deterministic samples; collect rich stats + parity-check."""

    def __init__(self, pipeline):
        requires = ["model", "hard_core_mapping"]
        promises = ["sanafe_simulation_results"]
        updates: List[str] = []
        clears: List[str] = []
        super().__init__(requires, promises, updates, clears, pipeline)
        self.metric = None

    def validate(self):
        if self.metric is not None:
            return self.metric
        return self.pipeline.get_target_metric()


    def process(self):
        # Read the contract surface and config knobs first so a misconfigured
        # step fails before any expensive setup.
        self.get_entry("model")
        hard_core_mapping = self.get_entry("hard_core_mapping")
        T = int(self.pipeline.config["simulation_steps"])
        spiking_mode = self.pipeline.config.get("spiking_mode", "lif")
        require_spiking_mode_supported(
            self.pipeline, "SanafeSimulationStep", backend="sanafe",
        )
        is_ttfs = spiking_mode in ("ttfs", "ttfs_quantized")

        parity_check = bool(self.pipeline.config.get("sanafe_parity_check", True))
        arch_preset = self.pipeline.config.get("sanafe_arch_preset", "loihi")
        custom_arch_path = self.pipeline.config.get("sanafe_custom_arch_path") or None
        log_potential = bool(self.pipeline.config.get("sanafe_log_potential_trace", False))
        log_messages = bool(self.pipeline.config.get("sanafe_log_message_trace", True))
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
            # Build HCM reference for the parity gate (skipped when disabled).
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

            # Run SANA-FE on the same sample.
            behavior = build_neural_behavior_config(self.pipeline)
            runner = SanafeRunner(
                mapping=hard_core_mapping,
                simulation_length=T,
                behavior=behavior,
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

            # When a HCM reference is available, attach per-core deltas to
            # each segment so the GUI floorplan "HCM diff" overlay shows
            # spatial drift even on runs that didn't fail the parity gate.
            # Always-zero overlays just mean the chip simulates HCM faithfully.
            if ref is not None:
                _attach_per_core_deltas(ref, sanafe_rec)

            if parity_check and is_ttfs and ttfs_ref is not None:
                from mimarsinan.chip_simulation.ttfs.ttfs_recorder import (
                    compare_ttfs_contract_records,
                    compare_ttfs_hardware_records,
                    format_first_ttfs_diff,
                )

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

        # Persist the report for the GUI snapshot builder.
        report = SanafeStepReport.from_records(arch_preset, per_sample)
        self.add_entry("sanafe_simulation_results", report, "pickle")

        # Report headline metrics.
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
