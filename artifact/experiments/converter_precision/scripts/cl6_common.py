""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

WT = Path(os.environ.get("WT", "/work/worktrees/w2-explore-g7"))
C = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get(
    "CL6_RUNS", "/tmp/work/cl6_runs"))
DATASETS = os.environ.get("CL6_DATASETS", "/work/research/mimarsinan/datasets")


READOUT_LAW = {"adc_bits": 32, "range": "calibrated", "mult_bits": 16}


ADC_SWEEP = (3, 4, 5, 6, 8, 32)
AT_LEAST_A = 32


BUDGETS = {"resident": 1.0, "half": 0.5, "eighth": 0.125}


ARMS = {"tuned": {}, "untrained": {"wq_endpoint_recovery_steps": 0}}


SWEEP_DECLARATIONS = {"scm_degradation_tolerance": 1.0}


SWEPT_PLATFORM_KEYS = ("cores", "readout_law")
SWEPT_DEPLOYMENT_KEYS = tuple(sorted({k for arm in ARMS.values() for k in arm}))


def _t0_43v() -> Dict[str, Any]:
    ""
    cfg = json.loads(next((WT / "templates" / "tier_0").glob("t0_43_*.json")).read_text())
    cfg["experiment_name"] = "t0_43v"
    pc = cfg["platform_constraints"]
    pc["activation_bits"] = 8
    pc["max_schedule_passes"] = 128
    pc["readout_law"] = dict(READOUT_LAW)
    cfg["deployment_parameters"]["allow_scheduling"] = True
    return cfg


def _fixture() -> Dict[str, Any]:
    return json.loads((WT / "tests" / "fixtures" / "integer_value" / "mvm_int_readout_lenet5.json").read_text())


def _cifar10_resnet20(seed: int):
    def build() -> Dict[str, Any]:
        cfg = json.loads((C / "cifar10" / "cl6_cifar10_resnet20.json").read_text())
        cfg["seed"] = int(seed)
        cfg["experiment_name"] = f"cl6_cifar10_resnet20_s{seed}"
        if os.environ.get("CL6_SMOKE_EPOCHS"):
            cfg["deployment_parameters"]["training_epochs"] = int(os.environ["CL6_SMOKE_EPOCHS"])
        return cfg
    return build


MEMBERS = {
    "mvm_int_readout_lenet5": (_fixture, "Readout Law"),
    "t0_43v": (_t0_43v, "Boundary Quantization"),
    **{f"cifar10_resnet20_s{s}": (_cifar10_resnet20(s), "Readout Law") for s in (0, 1, 2)},
}

MNIST_MEMBERS = ("mvm_int_readout_lenet5", "t0_43v")


def base_config(member: str) -> Dict[str, Any]:
    ""
    cfg = copy.deepcopy(MEMBERS[member][0]())
    dp = cfg["deployment_parameters"]
    dp.update(SWEEP_DECLARATIONS)
    dp["num_workers"] = int(os.environ.get("CL6_NUM_WORKERS", "0"))
    cfg["datasets_path"] = DATASETS
    return cfg


def apply_gpu_cap() -> Optional[float]:
    ""
    cap = os.environ.get("CL6_GPU_CAP_GIB")
    if not cap:
        return None
    import torch
    if not torch.cuda.is_available():
        return None
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = min(1.0, float(cap) * 1024 ** 3 / total)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    return fraction


def gpu_peak(fraction: Optional[float]) -> Optional[Dict[str, Any]]:
    ""
    try:
        import importlib
        import torch
        if not torch.cuda.is_available():
            return None
        last = {"peak_reserved_gib": torch.cuda.max_memory_reserved(0) / 1024 ** 3,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(0) / 1024 ** 3}
        rec: Dict[str, Any] = {"fraction": fraction, "device": torch.cuda.get_device_name(0)}
        try:
            reader = getattr(importlib.import_module("mimarsinan.common.diagnostics"), "lifetime_cuda_peaks", None)
        except ImportError:
            reader = None
        life = reader(0) if reader is not None else None
        if life is None or life.absent is not None:
            reason = "LIFETIME_PEAK_ABSENT" if life is None else life.absent
            return {**last, "peak_scope": f"last phase only ({reason})", **rec}
        return {"peak_reserved_gib": life.max_memory_reserved_bytes / 1024 ** 3,
                "peak_allocated_gib": life.max_memory_allocated_bytes / 1024 ** 3, "peak_scope": "process lifetime",
                **{f"last_phase_{k}": v for k, v in last.items()}, **rec}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def start_step(member: str) -> str:
    return MEMBERS[member][1]


def budget_cores(cores: list, footprint: Dict[str, int], fraction: float) -> list:
    ""
    import math
    out = []
    for ct in cores:
        f = int(footprint.get(geometry_key(ct), 0))
        out.append({**ct, "count": max(1, math.ceil(fraction * f))})
    return out


def geometry_key(core: Dict[str, Any]) -> str:
    return f"{int(core['max_axons'])}x{int(core['max_neurons'])}"


def place_cores(pc: Dict[str, Any], cores: list, point: str) -> None:
    ""
    override = (pc.get("target_overrides") or {}).get("cores")
    if isinstance(override, dict):
        if json.dumps(override.get("value"), sort_keys=True) != json.dumps(cores, sort_keys=True):
            pc["target_overrides"]["cores"] = {
                "value": copy.deepcopy(cores),
                "note": f"the cores of CL6 point {point} (cl6_common.point_config), in place of the member's "
                        f"declared {json.dumps(override.get('value'), sort_keys=True)} ({override.get('note')})"}
        if "cores" not in pc:
            return
    pc["cores"] = copy.deepcopy(cores)


def point_config(member: str, arm: str, adc_bits: int, cores: Optional[list], *,
                 allow_scheduling: bool = True, name: Optional[str] = None) -> Dict[str, Any]:
    ""
    cfg = base_config(member)
    pc = cfg["platform_constraints"]
    point = name or f"{member}__{arm}__adc{adc_bits}"
    if cores is not None:
        place_cores(pc, cores, point)
    pc["readout_law"] = {**pc["readout_law"], "adc_bits": int(adc_bits)}
    dp = cfg["deployment_parameters"]
    dp.update(ARMS[arm])
    dp["allow_scheduling"] = bool(allow_scheduling)
    cfg["experiment_name"] = point
    return cfg


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def src_digest(wt: Path = WT) -> str:
    ""
    h = hashlib.sha256()
    for p in sorted((wt / "src").rglob("*.py")):
        h.update(str(p.relative_to(wt)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


_PARITY = re.compile(r"\[ValueParityGate\] (PASS|FAIL) (\S+) \(max\|delta\|=([0-9.e+-]+), atol=([0-9.e+-]+), n=(\d+)\)")
_TWIN = re.compile(r"\[ValueTwinCertificate\] (PASS|FAIL) backend=(\S+) samples=(\d+) windows=(\d+) "
                   r"within_atol=([0-9.]+) max\|delta\|=([0-9.e+-]+) atol=([0-9.e+-]+) passed=(\w+)")


def certificates_from_log(text: str) -> list:
    ""
    out = []
    for m in _PARITY.finditer(text):
        out.append({"edge": "R", "gate": "ValueParityGate", "verdict": m.group(1), "compares": m.group(2),
                    "max_delta": float(m.group(3)), "atol": float(m.group(4)), "samples": int(m.group(5))})
    for m in _TWIN.finditer(text):
        out.append({"edge": "C", "gate": "ValueTwinCertificate", "verdict": m.group(1), "backend": m.group(2),
                    "samples": int(m.group(3)), "windows": int(m.group(4)), "within_atol": float(m.group(5)),
                    "max_delta": float(m.group(6)), "atol": float(m.group(7)), "passed": m.group(8) == "True"})
    return out


def atol0_certified(certs: list) -> bool:
    ""
    edges = {c["edge"] for c in certs}
    return edges >= {"R", "C"} and all(c["verdict"] == "PASS" and c["atol"] == 0.0 and c["max_delta"] == 0.0
                                       for c in certs)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    tmp.replace(path)
