"""[calculus §17/PR44] the deployed spike-count certificate gate."""

from __future__ import annotations

import torch

from mimarsinan.certification.count_alignment import certify_flow_counts
from mimarsinan.chip_simulation.spiking_semantics import is_lif
from mimarsinan.config_schema.registry import effective_value as _effective
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.simulation_factory import (
    build_spiking_hybrid_flow,
)


def _certificate_samples(pipeline, model, n: int) -> "torch.Tensor | None":
    trainer = BasicTrainer(
        model, pipeline.config["device"],
        DataLoaderFactory.for_pipeline(pipeline), None,
    )
    try:
        batches = list(trainer.iter_validation_batches(1))
    finally:
        trainer.close()
    xs = [pair[0] for pair in batches]
    if not xs:
        return None
    return xs[0][:n].to(pipeline.config["device"])


def run_spike_count_certificate_gate(pipeline, model, ir_graph, hybrid_mapping):
    """Certify deployed per-neuron window counts against the NF oracle; fatal.

    LIF-only: the synchronized count executor is the exact cell and the
    staircase theorem extends equality to streaming [calculus §16-17].
    ``spike_count_parity_samples <= 0`` disables the gate."""
    plan = DeploymentPlan.of(pipeline)
    if not is_lif(str(plan.spiking_mode)):
        return None
    n = int(_effective(pipeline.config, "spike_count_parity_samples"))
    if n <= 0:
        return None
    samples = _certificate_samples(pipeline, model, n)
    if samples is None:
        return None
    flow = build_spiking_hybrid_flow(pipeline, hybrid_mapping, model=model)
    cert, detail = certify_flow_counts(
        model.get_mapper_repr(), ir_graph, flow, samples, backend="hcm",
    )
    print(f"[SpikeCountCertificate] {cert.summary()}")
    print(f"[SpikeCountCertificate] {detail}")
    if not cert.passed:
        raise RuntimeError(
            f"spike-count certificate FAILED: {cert.summary()} | {detail}"
        )
    return cert
