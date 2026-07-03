"""Per-process registry mapping ``workload_id`` to the live ``JointArchHwProblem`` the backend delegates to."""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from compilagent import (
    BenchmarkBudget,
    DtypePolicy,
    ShapePolicy,
    ToleranceConfig,
    WorkloadKind,
    WorkloadSpec,
)


_LOCK = threading.Lock()
_PROBLEMS: Dict[str, Any] = {}


def register_problem(workload_id: str, problem: Any) -> None:
    """Bind ``workload_id`` to a live ``JointArchHwProblem``; re-registering a different instance under a used id raises ``ValueError``."""

    with _LOCK:
        existing = _PROBLEMS.get(workload_id)
        if existing is not None and existing is not problem:
            raise ValueError(
                f"workload_id {workload_id!r} already registered with a "
                f"different JointArchHwProblem instance"
            )
        _PROBLEMS[workload_id] = problem


def unregister_problem(workload_id: str) -> None:
    """Drop a previously registered problem; safe to call if absent."""

    with _LOCK:
        _PROBLEMS.pop(workload_id, None)


def lookup_problem(workload_id: str) -> Any:
    """Return the live ``JointArchHwProblem`` bound to ``workload_id``; raises ``KeyError`` if none is registered."""

    with _LOCK:
        try:
            return _PROBLEMS[workload_id]
        except KeyError as exc:
            known = sorted(_PROBLEMS.keys())
            raise KeyError(
                f"No JointArchHwProblem registered under {workload_id!r}; "
                f"known ids: {known or '(none)'}. "
                f"Did the optimizer call `register_problem(...)`?"
            ) from exc


def build_workload_spec(
    workload_id: str,
    *,
    title: str = "Mimarsinan layout-mapping workload",
    description: str = (
        "Joint NAS + hardware co-search for spiking neural networks. "
        "Each candidate is a (model_config, platform_constraints) "
        "configuration evaluated through mimarsinan's `JointArchHwProblem`."
    ),
    metadata: Optional[Dict[str, Any]] = None,
) -> WorkloadSpec:
    """Build a ``WorkloadSpec`` describing the bound problem, with a small budget and permissive tolerance."""

    return WorkloadSpec(
        id=workload_id,
        title=title,
        description=description,
        kind=WorkloadKind.FULL_MODEL,
        backend_id="mimarsinan_layout",
        dtype_policy=DtypePolicy(activation_dtype="fp32", param_dtype="fp32"),
        shape_policy=ShapePolicy(),
        tolerance=ToleranceConfig(atol=1.0, rtol=1.0, notes="not enforced"),
        budget=BenchmarkBudget(warmup=0, repetitions=1, max_seconds=10.0),
        metadata={"workload_id": workload_id, **(metadata or {})},
    )


__all__ = [
    "build_workload_spec",
    "lookup_problem",
    "register_problem",
    "unregister_problem",
]
