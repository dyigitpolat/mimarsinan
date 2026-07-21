"""[calculus §17/PR44] the deployed spike-count certificate gate."""

from __future__ import annotations

import torch

from mimarsinan.certification.count_alignment import certify_twin_flow_counts
from mimarsinan.chip_simulation.spiking_semantics import is_lif
from mimarsinan.config_schema.registry import effective_value as _effective
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.simulation_factory import (
    build_identity_mapping_for_pipeline,
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
    # The exact edge is chip-grid twin <-> packed program (same core
    # matrices); the model<->grid edge carries the honest WQ residual and is
    # governed by the nf_scm_parity atol gate, not this certificate.
    identity = build_identity_mapping_for_pipeline(
        ir_graph, pipeline_config=pipeline.config,
    )
    reference_flow = build_spiking_hybrid_flow(pipeline, identity, model=model)
    backend_flow = build_spiking_hybrid_flow(
        pipeline, hybrid_mapping, model=model,
    )
    # The fatal cell: synchronized identity-twin vs packed program — exact by
    # construction (same matrices, canonical schedule). The streaming cell is
    # a REPORT: streaming-vs-sync count deltas are the measured per-cycle
    # transient physics [§15-16]; the streaming census accuracy (read moments
    # later by the metric run) is that cell's arbiter, not per-window counts.
    cert, detail = certify_twin_flow_counts(
        ir_graph, reference_flow, backend_flow, samples, backend="hcm",
        discipline="synchronized",
    )
    print(f"[SpikeCountCertificate] synchronized: {cert.summary()}")
    print(f"[SpikeCountCertificate] synchronized: {detail}")
    for dv in cert.divergent:
        print(f"[SpikeCountCertificate] synchronized: divergent {dv}")
    if not cert.passed:
        raise RuntimeError(
            f"spike-count certificate FAILED (synchronized): "
            f"{cert.summary()} | {detail}"
        )
    stream_cert, _ = certify_twin_flow_counts(
        ir_graph, backend_flow, backend_flow, samples, backend="hcm",
        discipline="streaming", reference_discipline="synchronized",
    )
    print(
        "[SpikeTransientReport] streaming vs synchronized: "
        f"exact={stream_cert.exact_match_fraction:.6f} "
        f"max|dcount|={stream_cert.max_abs_delta:g} over "
        f"{stream_cert.neuron_windows_compared} neuron-windows "
        "(per-cycle transient physics; the streaming census accuracy is the "
        "arbiter for this cell)"
    )
    return cert
