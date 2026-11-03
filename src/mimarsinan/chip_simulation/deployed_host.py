"""[U-HDA] The deployed host's arithmetic: where, and in which dtype, every evaluation of a deployed program runs.

The deployed read (K9, the harvests' executor) evaluates the deployed program on the CPU: a value program as
``ValueHybridCoreFlow(mapping, device="cpu", dtype=float64)``; a spiking program as its spiking executor
(``build_spiking_hybrid_flow``) on the CPU, its cores at float64 (on the integer membrane lattice under weight
quantization) and each host kernel at its deployment dtype (the stored module's), one sample at a time where the
program declares it (``host_kernel_evaluation``). [U-DAR] That arithmetic is written once, per domain, in the record
``mapping/ir/deployment_dtype.py`` (``VALUE_ARITHMETIC``, ``SPIKING_ARITHMETIC``); this module names it per evaluation
(``deployed_arithmetic`` selects a plan's domain by its spiking mode) and places every evaluation on the deployed host.
Every executor of a deployed program and both twins of every certificate read it:

- the censuses (``pipelining/core/census``) and the event census;
- the spike-count certificate's identity and packed twins (K7) and the NF twin of every parity gate (the model placed
  on the deployed host for the gate, handed back after);
- the HCM reference of every chip backend and the chip simulators' host kernels (``SpikingDeploymentContract``'s
  ``host_compute_device``): nevresim (K8), Lava (Loihi), SANA-FE and the two ODIN twins;
- the value certificates (K4, K5, the operand program's).

A row of record therefore certifies the arithmetic its printed accuracy comes from, whatever device the pipeline trained
on. A CUDA device holds neither the float32 host kernels (cuDNN and cuBLAS choose other algorithms and reduction
orders) nor float64 reductions bit for bit, so an evaluation there would agree with the deployed read by measurement,
never by construction (the tuned C5 B0 program classes 7 of 5,000 E01 images differently on a GPU, U-CEN's records).
Training, tuning and model metrics stay on the pipeline's device: they are not evaluations of the deployed program.

Each certificate prints its device and dtype in the step's log (``[HostArithmetic] ...``) and emits a
``host_arithmetic`` reporter event. An evaluation moved off the device its modules live on hands back every module it
borrows (the model, each host op's module) where it was (``borrowed_host_placement``); on a CPU pipeline nothing moves
and nothing is restored, so every evaluation there is the base's.
"""

from __future__ import annotations

import contextlib
import copy
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Iterator

import torch
import torch.nn as nn

from mimarsinan.chip_simulation.core_semantics import INERT_SPIKING_MODE
from mimarsinan.common.reporter import emit_reporter_event
from mimarsinan.mapping.ir.deployment_dtype import (  # [U-DAR] the record; the device rule is its own (re-exported)
    DEPLOYED_HOST_DEVICE as DEPLOYED_HOST_DEVICE,
    SPIKING_ARITHMETIC,
    VALUE_ARITHMETIC,
    DeployedArithmetic,
    deployed_host_device as deployed_host_device,
    dtype_name,
)
from mimarsinan.mapping.support.device_placement import preserved_module_placement


def deployed_arithmetic(spiking_mode: Any) -> DeployedArithmetic:
    """[U-DAR] The record of a plan's domain, keyed by its spiking mode: the value record exactly when the plan resolves
    ``INERT_SPIKING_MODE`` (``core_semantics`` mvm, ``DeploymentPlan``), the spiking record otherwise."""
    return VALUE_ARITHMETIC if spiking_mode == INERT_SPIKING_MODE else SPIKING_ARITHMETIC


def _name(dtype: Any) -> str:
    return dtype_name(dtype)


@dataclass(frozen=True)
class HostArithmetic:
    """Where and in which dtype one evaluation of the deployed program runs, beside the device the pipeline asked for."""

    evaluation: str
    device: str
    dtype: str
    pipeline_device: str

    @property
    def moved(self) -> bool:
        """Whether the evaluation runs off the device the pipeline asked for."""
        return torch.device(self.pipeline_device).type != torch.device(self.device).type

    def line(self) -> str:
        moved = (f"; pipeline device {self.pipeline_device}: the deployed read runs on the CPU, where its arithmetic "
                 f"is held exactly" if self.moved else "")
        return f"[HostArithmetic] {self.evaluation}: device={self.device} dtype={self.dtype}{moved}"


def program_host_ops(program: Any) -> list:
    """The host ops of a deployed program: an IR graph's (``get_compute_ops``) or a packed program's (its stages'
    ``compute_op``); none for anything else."""
    if program is None:
        return []
    if hasattr(program, "get_compute_ops") and not hasattr(program, "stages"):
        return list(program.get_compute_ops())
    stages = getattr(program, "stages", None) or ()
    return [stage.compute_op for stage in stages if getattr(stage, "compute_op", None) is not None]


def _asked(asked: Any) -> str:
    return str(asked if asked is not None else DEPLOYED_HOST_DEVICE)


def spiking_host_arithmetic(evaluation: str, asked: Any, *programs: Any) -> HostArithmetic:
    """A spiking program's arithmetic on the deployed host: the record's (``SPIKING_ARITHMETIC``): cores at its core
    dtype, each host kernel at its deployment dtype (named from the programs' host ops)."""
    host = sorted({_name(SPIKING_ARITHMETIC.host_kernel_dtype(op)) for program in programs
                   for op in program_host_ops(program)})
    dtype = f"cores {_name(SPIKING_ARITHMETIC.core_dtype)}, host kernels {'/'.join(host) if host else 'none'}"
    return HostArithmetic(evaluation=evaluation, device=deployed_host_device(asked), dtype=dtype,
                          pipeline_device=_asked(asked))


def value_host_arithmetic(evaluation: str, asked: Any) -> HostArithmetic:
    """A value program's arithmetic on the deployed host: the record's core dtype throughout (``VALUE_ARITHMETIC``)."""
    return HostArithmetic(evaluation=evaluation, device=deployed_host_device(asked),
                          dtype=_name(VALUE_ARITHMETIC.core_dtype),
                          pipeline_device=_asked(asked))


def announce(arithmetic: HostArithmetic, reporter: Any = None) -> None:
    """Print the evaluation's device and dtype in the step's log and emit it as a ``host_arithmetic`` event."""
    print(arithmetic.line(), flush=True)
    emit_reporter_event(reporter, "host_arithmetic", asdict(arithmetic))


def _modules(owners: Iterable[Any]) -> list:
    """Every module an evaluation borrows from ``owners``: each ``nn.Module`` itself, each program's host-op modules."""
    out: list = []
    for owner in owners:
        if isinstance(owner, nn.Module):
            out.append(owner)
            continue
        for op in program_host_ops(owner):
            module = (getattr(op, "params", None) or {}).get("module")
            if isinstance(module, nn.Module):
                out.append(module)
    return out


def _off_host(module: nn.Module) -> bool:
    return any(t.device.type != DEPLOYED_HOST_DEVICE
               for t in (*module.parameters(), *module.buffers()))


@contextlib.contextmanager
def borrowed_host_placement(*owners: Any, moved: "bool | None" = None) -> Iterator[None]:
    """Hand back, on exit, the placement of every module an evaluation on the deployed host borrows from ``owners``
    (modules, and the host-op modules of programs): its original tensors are restored (``preserved_module_placement``).
    ``moved`` says whether the evaluation runs off the pipeline's device; when it is not given, a module is restored only
    where it lives off the deployed host at entry. Nothing that already lives there is touched, so an evaluation on a
    CPU pipeline is left exactly as it was."""
    modules = _modules(owners)
    if moved is False or (moved is None and not any(_off_host(m) for m in modules)):
        yield
        return
    with contextlib.ExitStack() as stack:
        for module in modules:
            stack.enter_context(preserved_module_placement(module))
        yield


def place_on_host(*owners: Any) -> None:
    """Place every module ``owners`` lend an evaluation (modules, and the host-op modules of programs) on the deployed
    host: an NF twin calls its program's host modules directly, so they live where its model is placed. Used inside
    ``borrowed_host_placement``, which hands them back; a module already there is left as it is."""
    for module in _modules(owners):
        if _off_host(module):
            module.to(DEPLOYED_HOST_DEVICE)


def held_on(module: Any, device: Any) -> Any:
    """``module`` as an executor on ``device`` may hold it: the module itself where every tensor it holds is on
    ``device`` (or it holds none), else a copy on ``device``, so building the executor never moves a module of the
    model it reads from."""
    if not isinstance(module, nn.Module):
        return module
    target = torch.device(device).type
    if all(t.device.type == target for t in (*module.parameters(), *module.buffers())):
        return module
    return copy.deepcopy(module).to(device)


@contextlib.contextmanager
def on_deployed_host(evaluation: str, asked: Any, *, programs: tuple = (), model: Any = None,
                     value: bool = False, reporter: Any = None) -> Iterator[HostArithmetic]:
    """One certificate's evaluation on the deployed host: its arithmetic named and announced, every module it borrows
    from ``model`` and ``programs`` handed back where it was on exit. Yields the arithmetic (``.device`` is where the
    certificate's samples and executors go)."""
    arithmetic = (value_host_arithmetic(evaluation, asked) if value
                  else spiking_host_arithmetic(evaluation, asked, *programs))
    announce(arithmetic, reporter)
    with borrowed_host_placement(model, *programs):
        yield arithmetic
