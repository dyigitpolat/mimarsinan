"""[W49] Theorem 1's host condition at Hard Core Mapping, read by the sampled run.

Hard Core Mapping runs its sampled certificates (the spike-count and value twins, the operand program certificate)
under ``host_kernel_replay``: every host kernel they evaluate is replayed on each sampled input alone and once more on
the batch (``certification.host_condition``). The condition is printed and its record goes into the deployment record
beside the static program check's; a batch-dependent or nondeterministic kernel is named there and puts the deployment
outside the per-sample certified scope by name. Like the static check, it never refuses a run."""

from __future__ import annotations

from typing import Any, Dict, List

from mimarsinan.certification.host_condition import HostKernelReplay, host_kernel_replay

__all__ = ["host_condition_records", "host_kernel_replay"]


def _program_host_ops(hybrid_mapping: Any) -> int:
    return len(hybrid_mapping.get_compute_ops()) if hasattr(hybrid_mapping, "get_compute_ops") else 0


def host_condition_records(replay: HostKernelReplay, hybrid_mapping: Any) -> List[Dict[str, Any]]:
    """Print the host condition the replay read over the deployed program's host ops and return its record."""
    certificate = replay.certificate(program_host_ops=_program_host_ops(hybrid_mapping))
    print(f"[HostCondition] Theorem 1 host condition: {certificate.verdict()}")
    return [certificate.record()]
