"""[calculus §17/PR44] the deployed spike-count certificate gate."""

from __future__ import annotations

import torch

from mimarsinan.certification.count_alignment import certify_twin_flow_counts
from mimarsinan.certification.twin_schedule import twin_schedule_diagnostic
from mimarsinan.chip_simulation.certification import CertificationCell
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


def certificate_gate_armed(pipeline) -> bool:
    """[§17] whether the streaming-twin certificate will run for this config:
    a counts-observable mode with a nonzero sample budget. Shared by the gate
    and the SCM rung-2 derivation (identity ≡ packed counts ⇒ the identity
    accuracy is derived, not re-measured)."""
    plan = DeploymentPlan.of(pipeline)
    observable, _reason = plan.mode_policy().certification_observable()
    if observable != "counts":
        return False
    return int(_effective(pipeline.config, "spike_count_parity_samples")) > 0


def run_spike_count_certificate_gate(pipeline, model, ir_graph, hybrid_mapping):
    """Certify deployed per-neuron window counts against the NF oracle; fatal.

    LIF-only: the synchronized count executor is the exact cell and the
    staircase theorem extends equality to streaming [calculus §16-17].
    ``spike_count_parity_samples <= 0`` disables the gate."""
    plan = DeploymentPlan.of(pipeline)
    observable, skip_reason = plan.mode_policy().certification_observable()
    if observable != "counts":
        print(f"[SpikeCountCertificate] SKIP ({plan.spiking_mode}): {skip_reason}")
        return None
    n = int(_effective(pipeline.config, "spike_count_parity_samples"))
    if n <= 0:
        return None
    assert certificate_gate_armed(pipeline)
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
    # The FATAL cell is cycle-based LIF on BOTH sides [user law: deployed
    # semantics = per-cycle LIF, full stop]: the identity-IR twin run as a
    # genuine streaming (fire-during-integrate) program — the torch-side
    # cycle-based simulation of the deployed matrices — must match the packed
    # program's streaming counts per neuron-window, atol=0. The analytic
    # synchronized executor is a training/tuning-side surrogate and is never
    # load-bearing here. The streaming-vs-sync delta stays a REPORT (the §16
    # gauge diagnostic; the census accuracy is that cell's arbiter).
    cert, detail = certify_twin_flow_counts(
        ir_graph, reference_flow, backend_flow, samples, backend="hcm",
        discipline="streaming", reference_discipline="streaming",
    )
    print(f"[SpikeCountCertificate] streaming-twin: {cert.summary()}")
    print(f"[SpikeCountCertificate] streaming-twin: {detail}")
    for dv in cert.divergent:
        print(f"[SpikeCountCertificate] streaming-twin: divergent {dv}")
    if not cert.passed:
        print(twin_schedule_diagnostic(identity, hybrid_mapping))
        raise RuntimeError(
            f"spike-count certificate FAILED (streaming twin): "
            f"{cert.summary()} | {detail}"
        )
    gauge_cert, _ = certify_twin_flow_counts(
        ir_graph, backend_flow, backend_flow, samples, backend="hcm",
        discipline="streaming", reference_discipline="synchronized",
    )
    print(
        "[SpikeTransientReport] streaming vs synchronized-gauge: "
        f"exact={gauge_cert.exact_match_fraction:.6f} "
        f"max|dcount|={gauge_cert.max_abs_delta:g} over "
        f"{gauge_cert.neuron_windows_compared} neuron-windows "
        "(per-cycle transient physics vs the analytic gauge; the streaming "
        "census accuracy is the arbiter)"
    )
    cell = CertificationCell.from_mode_policy(plan.mode_policy(), backend="hcm")
    print(f"[SpikeCountCertificate] cell {cell.cell_key}: streaming-twin PASS "
          f"({cert.neuron_windows_compared} windows, {cert.samples} sample(s))")
    return cert
