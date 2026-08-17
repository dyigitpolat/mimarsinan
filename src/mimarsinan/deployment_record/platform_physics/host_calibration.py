"""[H3b] Measure THIS host's compute rate, as operator-declared physics.

The host terms were the 100x wildcard in every e2e/energy comparison: priced
from ESTIMATED rates while the host wall is 99% of the measured e2e. The
deployment host is right here, so its rate is measurable — the result is an
override-ready ``platform_physics_overrides`` block with ``evidence_kind:
measured`` and the machine's identity in the note, never a silent default.

``p_host`` is measured only where RAPL exposes package energy; otherwise it is
NOT invented — the block simply omits it and says so.
"""

from __future__ import annotations

import platform
import time
from typing import Any, Dict, Optional, Tuple

#: ComputeOp-shaped fp32 matmuls: (batch, in_features, out_features) — the
#: encoder/softmax-scale work a host actually runs in this framework's flows.
CALIBRATION_SHAPES: Tuple[Tuple[int, int, int], ...] = (
    (1, 784, 500),
    (1, 500, 100),
    (1, 1024, 256),
)


def macs_of(shape: Tuple[int, int, int]) -> int:
    batch, in_features, out_features = shape
    return int(batch) * int(in_features) * int(out_features)


def measure_host_macs_per_s(
    *,
    shapes: Tuple[Tuple[int, int, int], ...] = CALIBRATION_SHAPES,
    repeats: int = 200,
    warmup: int = 20,
) -> float:
    """Sustained fp32 MACs/s over deployment-shaped ops, best-of-shapes.

    Best (not mean) of the per-shape sustained rates: the host runs each
    ComputeOp as a single matmul, so the honest rate is what the BLAS path
    sustains at that shape — averaging in a cold shape would understate it.
    """
    import torch

    torch.manual_seed(0)
    best = 0.0
    for shape in shapes:
        batch, in_features, out_features = shape
        x = torch.randn(batch, in_features)
        w = torch.randn(in_features, out_features)
        for _ in range(warmup):
            x @ w
        start = time.perf_counter()
        for _ in range(repeats):
            x @ w
        wall = time.perf_counter() - start
        best = max(best, macs_of(shape) * repeats / wall)
    return best


def measure_host_op_overhead_s(
    *, repeats: int = 400, warmup: int = 40,
) -> Tuple[float, float, float]:
    """[R2] Per-invocation wall of one host ComputeOp, through the
    deployment's OWN dispatch path.

    Drives ``execute_compute_op_torch`` — gather from the state buffer,
    dtype resolution, module dispatch — on a tiny op. Returns measured
    (p10, median, p90) walls. ESTIMAND, stated: the size-INDEPENDENT dispatch
    floor. The record's per-op walls also carry data marshalling that scales
    with boundary tensor sizes; that residual belongs to a per-byte term if
    fidelity shows it dominating, never inside this constant.
    """
    import numpy as np
    import torch

    from mimarsinan.chip_simulation.hybrid_run.host_compute import (
        execute_compute_op_torch,
    )
    from mimarsinan.mapping.ir import ComputeOp, IRSource

    torch.manual_seed(0)
    op = ComputeOp(
        id=1, name="calibration_op",
        input_sources=np.array([IRSource(0, j) for j in range(8)], dtype=object),
        op_type="Identity",
        params={"module": torch.nn.Identity()},
    )
    state = {0: torch.zeros(1, 8)}
    original = torch.zeros(1, 8)
    for _ in range(warmup):
        execute_compute_op_torch(op, original, state)
    walls = []
    for _ in range(repeats):
        start = time.perf_counter()
        execute_compute_op_torch(op, original, state)
        walls.append(time.perf_counter() - start)
    walls.sort()
    return (
        walls[len(walls) // 10],
        walls[len(walls) // 2],
        walls[(len(walls) * 9) // 10],
    )


def read_rapl_package_j() -> Optional[float]:
    """The RAPL package energy counter (J), or None where unreadable."""
    path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return int(fh.read().strip()) / 1e6
    except (OSError, ValueError):
        return None


def measure_p_host_w(*, seconds: float = 2.0) -> Optional[float]:
    """Package power under the calibration load (W), or None without RAPL."""
    import torch

    before = read_rapl_package_j()
    if before is None:
        return None
    x = torch.randn(64, 1024)
    w = torch.randn(1024, 1024)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        x @ w
    after = read_rapl_package_j()
    if after is None or after <= before:
        return None  # counter wrapped or vanished: no number beats a wrong one
    return (after - before) / seconds


def machine_identity() -> str:
    return f"{platform.node()} ({platform.machine()}, {platform.processor() or 'cpu'})"


def calibration_overrides(
    *, macs_per_s: float, op_overhead_s: Tuple[float, float, float],
    p_host_w: Optional[float],
    identity: str,
) -> Dict[str, Dict[str, Any]]:
    """The ``platform_physics_overrides`` block a run config declares."""
    note = (
        f"Measured on the deployment host {identity} over ComputeOp-shaped "
        f"fp32 matmuls (best sustained shape). Re-run "
        f"scripts/calibrate_host.py when the host changes."
    )
    overrides: Dict[str, Dict[str, Any]] = {
        "t_host_op_overhead": {
            # The vocabulary declares us; the band is measured percentiles.
            "low": op_overhead_s[0] * 1e6,
            "nominal": op_overhead_s[1] * 1e6,
            "high": op_overhead_s[2] * 1e6,
            "unit": "us",
            "evidence_kind": "measured",
            "note": (
                f"Per-invocation host ComputeOp dispatch wall on {identity}, "
                f"measured through execute_compute_op_torch (the deployment's "
                f"own path; median of repeats). The rate term prices the "
                f"arithmetic; this prices the dispatch."
            ),
        },
        "host_macs_per_s": {
            # The vocabulary declares G/s.
            "nominal": macs_per_s / 1e9,
            "unit": "G/s",
            "evidence_kind": "measured",
            "note": note,
        },
    }
    if p_host_w is not None:
        overrides["p_host"] = {
            "nominal": p_host_w,
            "unit": "W",
            "evidence_kind": "measured",
            "note": (
                f"RAPL package power under the calibration load on {identity}. "
                f"Whole-package: an upper bound on the deployment share."
            ),
        }
    return overrides


def run_calibration(**measure_kwargs: Any) -> Dict[str, Any]:
    """Measure and assemble the full calibration artifact."""
    macs_per_s = measure_host_macs_per_s(**measure_kwargs)
    op_overhead_s = measure_host_op_overhead_s()
    p_host = measure_p_host_w()
    identity = machine_identity()
    return {
        "machine": identity,
        "host_macs_per_s": macs_per_s,
        "t_host_op_overhead_s": list(op_overhead_s),
        "p_host_w": p_host,
        "p_host_basis": (
            "RAPL package counter" if p_host is not None
            else "unavailable: no readable RAPL counter — p_host stays declared"
        ),
        "platform_physics_overrides": calibration_overrides(
            macs_per_s=macs_per_s, op_overhead_s=op_overhead_s,
            p_host_w=p_host, identity=identity,
        ),
    }
