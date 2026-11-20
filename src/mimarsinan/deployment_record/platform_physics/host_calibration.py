"""[H3b] Measure THIS machine's compute rate and dispatch wall: [U-HOST] a labelled diagnostic of the development
machine, never a host block.

The machine that runs the toolchain is not the deployment host (the owner's rule: host metrics are priced from a
Jetson-class device's published data, never from the development machine or a GPU), so its measured rows price
nothing: the artifact's ``diagnostic`` carries them under ``DIAGNOSTIC_LABEL`` (``evidence_kind: measured``, the
machine's identity in each note), each judged as a constant of the vocabulary, and no ``platform_physics_overrides``
block. A priced host is an identified host description (``hosts/<name>.json``), the only source the admission rule
admits a host row from (``hosts.admission``); ``host_block`` and ``load_host_block`` refuse a calibration artifact
by that rule's name.

``p_host`` is measured only where RAPL exposes package energy; otherwise it is
NOT invented: the diagnostic omits it and says so, or carries the operator's
DECLARED row (a band with its own evidence) beside the measured ones. A row both
measured and declared refuses by name.
"""

from __future__ import annotations

import json
import os
import platform
import time
from typing import Any, Dict, Mapping, NoReturn, Optional, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.platform_physics.constants import constant_from_declaration
from mimarsinan.deployment_record.platform_physics.hosts.admission import HostRowAdmissionError

#: [U-HOST] The label of a calibration's rows: what they are, and that they price nothing.
DIAGNOSTIC_LABEL = ("diagnostic: host rows measured on the development machine (the machine running the toolchain); "
                    "never a host block or a priced term (a priced host is an identified host description's rows)")


class HostRowConflictError(ToolchainRefusal, ValueError):
    """A host row the calibration measured is also declared: two values for one host."""


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


def measure_host_mac_rates(
    *,
    shapes: Tuple[Tuple[int, int, int], ...] = CALIBRATION_SHAPES,
    repeats: int = 200,
    warmup: int = 20,
) -> Dict[Tuple[int, int, int], float]:
    """Sustained fp32 MACs/s per deployment-shaped op."""
    import torch

    torch.manual_seed(0)
    rates: Dict[Tuple[int, int, int], float] = {}
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
        rates[shape] = macs_of(shape) * repeats / wall
    return rates


def measure_host_macs_per_s(**measure_kwargs: Any) -> float:
    """Best (not mean) of the per-shape sustained rates: the host runs each
    ComputeOp as a single matmul, so the honest rate is what the BLAS path
    sustains at that shape — averaging in a cold shape would understate it.
    """
    return max(measure_host_mac_rates(**measure_kwargs).values())


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
    from mimarsinan.mapping.ir.deployment_dtype import SPIKING_ARITHMETIC

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
        execute_compute_op_torch(op, original, state, arithmetic=SPIKING_ARITHMETIC)
    walls = []
    for _ in range(repeats):
        start = time.perf_counter()
        execute_compute_op_torch(op, original, state, arithmetic=SPIKING_ARITHMETIC)
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


def cpu_model() -> str:
    """The processor's model name (Linux ``/proc/cpuinfo``), else what ``platform`` reports."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return " ".join(line.split(":", 1)[1].split())
    except OSError:
        pass
    return platform.processor() or "cpu"


def machine_identity() -> str:
    return f"{platform.node()} ({platform.machine()}, {cpu_model()})"


def _affinity() -> Tuple[int, ...]:
    getter = getattr(os, "sched_getaffinity", None)
    return tuple(sorted(getter(0))) if getter is not None else ()


def calibration_diagnostic(
    *, macs_per_s: float, op_overhead_s: Tuple[float, float, float],
    p_host_w: Optional[float],
    identity: str,
    declared: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """[U-HOST] The development machine's measured host rows (``DIAGNOSTIC_LABEL``) plus the operator's ``declared``
    rows (each a physics constant with its own evidence, copied verbatim), each judged as a constant of the
    vocabulary: a diagnostic, never an override block (no physics is resolved from it)."""
    note = (
        f"Measured on the development machine {identity} over ComputeOp-shaped "
        f"fp32 matmuls (best sustained shape): a diagnostic, never a host price. Re-run "
        f"scripts/calibrate_host.py when the machine changes."
    )
    rows: Dict[str, Dict[str, Any]] = {
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
                f"own path; median of repeats): a diagnostic of the development machine."
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
        rows["p_host"] = {
            "nominal": p_host_w,
            "unit": "W",
            "evidence_kind": "measured",
            "note": (
                f"RAPL package power under the calibration load on {identity}. "
                f"Whole-package: an upper bound on the machine's share (a diagnostic)."
            ),
        }
    for key, declaration in (declared or {}).items():
        if key in rows:
            raise HostRowConflictError(
                f"host row {key!r} is both measured on {identity} and declared; declare it only "
                "where the calibration cannot measure it")
        rows[key] = dict(declaration)
    for key, row in rows.items():  # every row a valid constant, evidence included
        constant_from_declaration(key, row)
    return rows


def run_calibration(
    *, declared: Optional[Mapping[str, Mapping[str, Any]]] = None, **measure_kwargs: Any,
) -> Dict[str, Any]:
    """Measure and assemble the full calibration artifact: [U-HOST] a labelled diagnostic."""
    import torch

    rates = measure_host_mac_rates(**measure_kwargs)
    macs_per_s = max(rates.values())
    op_overhead_s = measure_host_op_overhead_s()
    p_host = measure_p_host_w()
    threads, affinity = torch.get_num_threads(), _affinity()
    machine = machine_identity()
    identity = f"{machine}, {threads} torch thread(s) on CPUs {list(affinity)}"
    declared_power = (declared or {}).get("p_host")
    if p_host is not None:
        basis = "RAPL package counter"
    elif declared_power is not None:
        basis = (f"declared: {declared_power.get('evidence_kind')}: {declared_power.get('citation', '')} "
                 "(no readable RAPL counter)")
    else:
        basis = "unavailable: no readable RAPL counter — p_host stays declared"
    return {
        "machine": machine,
        "threads": threads,
        "cpu_affinity": list(affinity),
        "torch_version": str(torch.__version__),
        "host_macs_per_s": macs_per_s,
        "host_macs_per_s_by_shape": {"x".join(map(str, shape)): rate for shape, rate in rates.items()},
        "t_host_op_overhead_s": list(op_overhead_s),
        "p_host_w": p_host,
        "p_host_basis": basis,
        "diagnostic": {"label": DIAGNOSTIC_LABEL, "rows": calibration_diagnostic(
            macs_per_s=macs_per_s, op_overhead_s=op_overhead_s,
            p_host_w=p_host, identity=identity, declared=declared,
        )},
    }


def host_block(artifact: Mapping[str, Any]) -> NoReturn:
    """[U-HOST] A calibration artifact is never a host block: refused by the admission rule's name."""
    raise HostRowAdmissionError(
        f"a host calibration artifact ({artifact.get('machine', 'the development machine')}) is a labelled "
        f"diagnostic of the machine running the toolchain, never a host block: host rows are admitted only from an "
        "identified host description (hosts/<name>.json, hosts.description.host_description_block)")


def load_host_block(path: str) -> NoReturn:
    """``host_block`` of the calibration artifact at ``path``: refused by name."""
    with open(path, "r", encoding="utf-8") as fh:
        host_block(json.load(fh))
