"""HACC NUS - ODIN Deployment: freeze the mapped network into a board bundle."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from mimarsinan.certification.record_certificates import certify_run_records
from mimarsinan.chip_simulation.odin_hacc.artifact import (
    BUNDLE_BASENAME,
    CAPTURE_BASENAME,
    bundle_stats,
    campaign_indices,
    kernel_table,
    render_bundle,
)
from mimarsinan.chip_simulation.odin_hacc.freeze import build_bundle
from mimarsinan.chip_simulation.odin_hacc.program_freeze import (
    ProgramFreezer,
    entry_rasters,
    readout_core_of,
)
from mimarsinan.chip_simulation.odin_hacc.witness import TwinWitness
from mimarsinan.data_handling.sample_loader import load_test_pairs_by_index
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.platform.platform_constraints import (
    resolve_platform_mapping_params,
)
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.engine.pipeline_helpers import (
    require_spiking_mode_supported,
)
from mimarsinan.pipelining.core.simulation_factory import (
    assert_spike_parity_or_raise,
    build_deployment_contract,
    hcm_reference_flow,
    record_on_hcm_flow,
)
from mimarsinan.pipelining.core.steps.pipeline_step import (
    METRIC_CARRIED,
    PipelineStep,
)

#: The name every registry, capability table and certificate uses for the
#: EXPORT target. It shares the crossbar's capability row with the physical
#: backend it produces work for.
BACKEND_NAME = "odin_hacc"

#: The ODIN membrane starts every sample at zero — the CLEAR stage the exporter
#: emits rewrites exactly those bytes, and the software twins zero theirs.
MEMBRANE_INIT = 0

#: The nevresim fragment: its presence means this pipeline already gated the
#: nevresim<->HCM window counts (FATAL at any mismatch on an integer chip).
NEVRESIM_FRAGMENT = "deployment_record_nevresim"


class OdinHaccDeploymentStep(PipelineStep):
    """Export the deployed network as one sealed, board-executable bundle."""

    REQUIRES = ("model", "hard_core_mapping", "platform_constraints_resolved")
    PROMISES = ("odin_hacc_deployment_bundle",)

    @classmethod
    def applies_to(cls, plan) -> bool:
        return bool(plan.enable_odin_hacc_export)

    def __init__(self, pipeline):
        plan = DeploymentPlan.of(pipeline)
        requires = list(self.REQUIRES)
        if plan.enable_nevresim_simulation:
            # The third arm is not re-run here: it is REQUIRED to have run, so
            # a bundle can never be frozen from a network whose nevresim leg
            # was never gated against the HCM reference.
            requires.append(NEVRESIM_FRAGMENT)
        super().__init__(requires, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)
        self.metric = None
        self.document: Dict[str, Any] | None = None

    def validate(self):
        return self.metric if self.metric is not None \
            else self.pipeline.get_target_metric()

    def validate_metric_kind(self) -> str:
        return METRIC_CARRIED

    def process(self):
        self.get_entry("model")
        mapping = self.get_entry("hard_core_mapping")
        platform = self.get_entry("platform_constraints_resolved")
        config = self.pipeline.config
        plan = DeploymentPlan.of(self.pipeline)
        require_spiking_mode_supported(
            self.pipeline, "OdinHaccDeploymentStep", backend=BACKEND_NAME)
        arms = ["hcm"]
        if plan.enable_nevresim_simulation:
            self.get_entry(NEVRESIM_FRAGMENT)
            arms.append("nevresim")

        contract = build_deployment_contract(self.pipeline)
        params = resolve_platform_mapping_params(platform["cores"])
        indices = campaign_indices(config)
        pairs = load_test_pairs_by_index(
            self.pipeline.data_provider_factory, indices,
            num_workers=int(config.get("num_workers", 4)))

        freezer = ProgramFreezer(
            mapping, contract=contract, timesteps=contract.simulation_steps,
            weight_bits=int(platform.get("weight_bits", config["weight_bits"])),
            effective_max_axons=int(params.effective_max_axons),
            membrane_init=MEMBRANE_INIT,
            weight_sign_granularity=str(config["weight_sign_granularity"]))

        # ONE reference executor for the whole campaign: rebuilding it per
        # sample cost more than the freezing did, and it carries no state
        # between samples (every window is recorded from a cleared membrane).
        reference_flow = hcm_reference_flow(
            self.pipeline, mapping, device=config["device"])
        frozen = [
            self._freeze_one(reference_flow, freezer, sample, label, index)
            for index, (sample, label) in enumerate(pairs)
        ]
        document, capture = self._build(
            frozen, config=config, contract=contract, freezer=freezer,
            params=params, platform=platform, arms=arms, plan=plan)
        paths = self._write(document, capture)
        self.document = document

        stats = bundle_stats(document, capture, paths)
        self.add_entry("odin_hacc_deployment_bundle", stats, "basic")
        self.metric = self.pipeline.get_target_metric()
        self.pipeline.reporter.report(
            "ODIN HACC Bundle Accuracy", stats["frozen_accuracy"])
        self._verdict = {
            "status": "pass",
            "rule": "HCM reference vs ODIN cycle-accurate twin (exact counts), "
                    "then the shipped bundle reader vs the repository encoder",
            "detail": stats,
        }
        print(f"ODIN HACC deployment bundle: {stats['samples']} sample(s), "
              f"{stats['cores']} pass(es), frozen accuracy "
              f"{stats['frozen_accuracy']:.6f} -> {paths['bundle']}")

    def _freeze_one(self, reference_flow, freezer, sample, label, index: int):
        """One sample, with the HCM reference gated against the twin exactly."""
        reference = record_on_hcm_flow(
            reference_flow, sample, sample_index=index,
            device=self.pipeline.config["device"])
        actual = freezer.freeze(
            sample.detach().cpu().numpy().reshape(1, -1),
            sample_index=index, label=int(label.reshape(-1)[0]))
        assert_spike_parity_or_raise(reference, actual.record)
        certificate = certify_run_records(
            reference, actual.record, backend=BACKEND_NAME)
        if not certificate.passed:
            raise AssertionError(
                f"{BACKEND_NAME}: sample {index}: {certificate.summary()} — the "
                f"cycle-accurate ODIN twin did not reproduce the HCM reference, "
                f"so its counts are not this network's and no bundle is written")
        return actual

    def _build(self, frozen, *, config, contract, freezer, params, platform,
               arms, plan):
        segment = frozen[0].plan.hcm
        rasters = entry_rasters(frozen)
        traces = [sample.plan.trace for sample in frozen]
        certification = list(range(min(
            len(frozen), int(config["odin_hacc_certification_samples"]))))
        return build_bundle(
            name=str(config.get("odin_hacc_bundle_name")
                     or _default_name(self.pipeline)),
            title=f"HACC NUS - ODIN Deployment: {config.get('model_type')} as "
                  f"{len(segment.cores)} host-mediated NC=1 passes",
            description=(
                "The deployed network's neural segment, frozen one core per "
                "pass. Every expectation below is the cycle-accurate twin's, "
                "gated sample by sample against the HCM torch reference at "
                "exact counts; the entry raster of each sample is what this "
                "program's HOST compute stages produced for it, which is why "
                "the bundle is executable for these samples and no others."),
            mapping=segment,
            rasters=rasters,
            labels=[sample.label for sample in frozen],
            simulation_length=int(contract.simulation_steps),
            chip_latency=int(ChipLatency(segment).calculate()),
            soma_law=freezer.soma_law,
            weight_bits=freezer.weight_bits,
            effective_max_axons=int(params.effective_max_axons),
            weight_sign_granularity=freezer.weight_sign_granularity,
            membrane_init=MEMBRANE_INIT,
            readout_core=readout_core_of(segment),
            certification=certification,
            provenance=self._provenance(arms),
            kernel_table=kernel_table(),
            model=self._model_table(config, platform, len(frozen), plan),
            witness=TwinWitness(),
            traces=traces,
        )

    def _provenance(self, arms: List[str]) -> Dict[str, Any]:
        return {
            "generator": "OdinHaccDeploymentStep",
            "derivation": TwinWitness.derivation,
            "agreement_arms": list(arms),
            "agreement": (
                "every shipped sample's twin counts were gated EXACT against "
                "the HCM torch reference in this run; the nevresim arm, when "
                "enabled, was gated against the same reference by the "
                "Simulation step before this step could run"),
            "experiment": str(self.pipeline.config.get("experiment_name", "")),
        }

    def _model_table(self, config, platform, samples: int, plan) -> Dict[str, Any]:
        return {
            "name": str(config.get("model_type", "")),
            "input_lines": int(config.get("input_size", 0)),
            "classes": int(config.get("num_classes", 0)),
            "timesteps": int(config["simulation_steps"]),
            "samples": int(samples),
            "cores_declared": platform["cores"],
            "spiking_mode": str(plan.spiking_mode),
        }

    def _write(self, document, capture) -> Dict[str, str]:
        directory = os.path.join(self.pipeline.working_directory, "odin_hacc")
        os.makedirs(directory, exist_ok=True)
        paths = {
            "bundle": os.path.join(directory, BUNDLE_BASENAME),
            "capture": os.path.join(directory, CAPTURE_BASENAME),
        }
        for key, payload in (("bundle", document), ("capture", capture)):
            with open(paths[key], "w", encoding="utf-8") as handle:
                handle.write(render_bundle(payload))
        return paths


def _default_name(pipeline) -> str:
    experiment = str(pipeline.config.get("experiment_name", "")).strip()
    return experiment or "odin_hacc_deployment"
