"""Hard Core Mapping's deployment-record fragment emission (W4 stage 2).

Persists the HCM fragments as JSON-safe dicts through the pipeline cache
(the ``deployment_record_hcm`` promise): schedule/placement/utilization
mirrors via ``deployment_record.build.from_mapping``, the weight-programming
totals (the seal's independent cross-check figure), the gate-reduced boundary
traffic (when the spike-count gate ran), and the step's own accuracy read.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mimarsinan.deployment_record.build.partition import compute_partition_record
from mimarsinan.models.spiking.hybrid.carry import pass_transfer_for_backend
from mimarsinan.deployment_record.build.from_mapping import (
    placement_record_from_mapping,
    schedule_record_from_mapping,
    utilization_record_from_mapping,
)
from mimarsinan.deployment_record.schema import (
    AccuracyReadRecord,
    CertificateRecord,
)


def emit_deployment_record_hcm(
    step: Any,
    hybrid_mapping: Any,
    *,
    platform_constraints: Any,
    scm_fragment: Any,
    programming: Any,
    crossbar_report: Any,
    spike_gate_result: Any,
    accuracy: float,
    observes_values: bool,
    model: Any,
    ir_graph: Any,
    input_shape: Any,
    num_classes: int,
    encoding_placement: str,
) -> None:
    """Build the fragments from the live objects and ``add_entry`` them."""
    schedule = schedule_record_from_mapping(
        hybrid_mapping,
        weight_bits=platform_constraints.get("weight_bits"),
        params_reloaded=int(scm_fragment["reuse_plan"]["params_reloaded"]),
        timesteps=platform_constraints.get("simulation_steps"),
        # The HCM executor is what produced this record's numbers, and it carries
        # the raster; a backend that collapses records its own discipline.
        pass_transfer=pass_transfer_for_backend("hcm"),
    )
    placement = placement_record_from_mapping(hybrid_mapping)
    # The logical on-chip/host split is an ungated CENSUS sealed on every run
    # (the validity gate reads the same SSOT pair, so the two can never disagree).
    partition = compute_partition_record(
        ir_graph,
        model,
        input_shape,
        int(num_classes),
        encoding_placement=encoding_placement,
    )
    utilization = utilization_record_from_mapping(
        hybrid_mapping,
        crossbar_report=crossbar_report,
        relay_cores_inserted=int(scm_fragment["relay_cores_inserted"]),
        partition=partition,
    )
    read = AccuracyReadRecord(
        metric=float(accuracy),
        backend="value_census" if observes_values else "hcm",
        # The metric run does not report its sample count at this producer;
        # 0 = untracked (the terminal assembly step refines it).
        samples=0,
        kind="measured",
        step="Hard Core Mapping",
    )
    boundary_traffic: Optional[List[Dict[str, Any]]] = None
    certificates: List[Dict[str, Any]] = []
    if spike_gate_result is not None:
        cert, boundaries = spike_gate_result
        boundary_traffic = [record.to_dict() for record in boundaries]
        certificates.append(CertificateRecord(
            name="spike_count_streaming_twin",
            backend=str(cert.backend),
            passed=bool(cert.passed),
            neuron_windows_compared=int(cert.neuron_windows_compared),
            exact_match_fraction=float(cert.exact_match_fraction),
            max_abs_delta=float(cert.max_abs_delta),
            detail=cert.summary(),
        ).to_dict())
    step.add_entry("deployment_record_hcm", {
        "schedule": schedule.to_dict(),
        "placement": placement.to_dict(),
        "utilization": utilization.to_dict(),
        "weight_programming": {
            "neural_stages": int(programming.neural_stages),
            "programming_events": int(programming.programming_events),
            "params_programmed": int(programming.params_programmed),
            "params_unique": int(programming.params_unique),
        },
        "boundary_traffic": boundary_traffic,
        "accuracy_reads": [read.to_dict()],
        "certificates": certificates,
    }, "basic")
