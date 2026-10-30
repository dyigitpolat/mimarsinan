"""[calculus §17/PR44] the deployed spike-count certificate gate."""

from __future__ import annotations

import gc

import torch

from mimarsinan.certification.count_alignment import certify_twin_flow_counts
from mimarsinan.certification.twin_schedule import twin_schedule_diagnostic
from mimarsinan.chip_simulation.certification import CertificationCell
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.config_schema.registry import effective_value as _effective
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.deployment_record.build.from_certificates import (
    boundary_traffic_from_node_counts,
)
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


def _stage_indices_by_node(hybrid_mapping) -> "dict[int, int]":
    """Producing top-level stage index per node id, from the stage output maps."""
    indices: dict[int, int] = {}
    for stage_index, stage in enumerate(getattr(hybrid_mapping, "stages", []) or []):
        for io_slice in getattr(stage, "output_map", []) or []:
            indices.setdefault(int(io_slice.node_id), int(stage_index))
    return indices


def _report_synchronized_gauge(pipeline, ir_graph, backend_flow, samples):
    """[§16] the streaming-vs-analytic-gauge transient REPORT (never a gate).

    The gauge is the closed-form staircase, whose hypothesis a per-event point
    denies outright — there is no analytic gauge to compare against there, so
    the report says so by name instead of running a refused executor.
    """
    if SomaLaw.resolve(pipeline.config).is_per_event:
        print(
            "[SpikeTransientReport] SKIP under firing_granularity='per_event':"
            " the analytic staircase gauge assumes a window's count depends "
            "only on the total integrated charge, which the per-event law "
            "denies — there is no gauge to measure the transient against."
        )
        return
    gauge_cert, _, _ = certify_twin_flow_counts(
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


def run_spike_count_certificate_gate(pipeline, model, ir_graph, hybrid_mapping):
    """Certify deployed per-neuron window counts against the NF oracle; fatal.

    LIF-only: the synchronized count executor is the exact cell and the
    staircase theorem extends equality to streaming [calculus §16-17].
    ``spike_count_parity_samples <= 0`` disables the gate.

    Returns ``None`` on every skip path; on a PASS returns
    ``(certificate, boundary_traffic)`` where ``boundary_traffic`` is the
    per-node reduction of the packed program's counts (reduced EAGERLY here —
    the raw ``(B, n)`` tensors never leave the gate scope)."""
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
    cert, detail, backend_counts = certify_twin_flow_counts(
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
    # Boundary-traffic reduction, eager and in-scope: scalars only survive.
    boundary_traffic = boundary_traffic_from_node_counts(
        backend_counts,
        stage_index_by_node=_stage_indices_by_node(hybrid_mapping),
    )
    del backend_counts
    _report_synchronized_gauge(
        pipeline, ir_graph, backend_flow, samples,
    )
    cell = CertificationCell.from_mode_policy(plan.mode_policy(), backend="hcm")
    print(f"[SpikeCountCertificate] cell {cell.cell_key}: streaming-twin PASS "
          f"({cert.neuron_windows_compared} windows, {cert.samples} sample(s))")
    # [16.13] the twin flows cache tens of GB of segment tensors and nn.Module
    # graphs are cyclic: free them before the step's metric read or it OOMs.
    del reference_flow, backend_flow, identity
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cert, boundary_traffic
