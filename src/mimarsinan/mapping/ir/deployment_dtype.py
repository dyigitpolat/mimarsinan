"""[U-DAR] The one record of the deployed arithmetic: where, at which dtypes and in which tie-rule plane every
evaluation that stands for the deployed program runs, per domain.

The deployed read (K9, the harvests' executor) fixes the arithmetic; this module writes it down once, and every
evaluation of a deployed program and every analysis that claims "the value the deployed program puts on a line" reads
it: the constant fold and its host-op probe (``mapping/pruning/liveness_transfer``), the value executor and its host
ops, the spiking executors and the chip backends' boundary buffers, the censuses (``pipelining/core/census``), the
deployed host (``chip_simulation/deployed_host.py``), the certificates, the static check's declaration and the
randomized executors. Nothing else in src names a dtype or a device for these evaluations; a remaining float dtype
literal on an executor path is classified by the AST ratchet (``tests/unit/mapping/test_deployed_arithmetic.py``).

Two domains, keyed by the plan (``chip_simulation.deployed_host.deployed_arithmetic``: the value domain is the plan
whose spiking mode is ``INERT_SPIKING_MODE``, exactly when ``core_semantics`` is mvm):

- **value** (``VALUE_ARITHMETIC``): cores ``(x @ W + b) / theta`` at float64 (a law core in integer codes, exact
  below 2^53); host kernels as the module's float64 twin with inputs gathered at float64; the program's input, every
  segment-boundary buffer and the final read at float64; the integer form is the int64 golden (K6).
- **spiking** (``SPIKING_ARITHMETIC``): cores at float64 on the integer membrane lattice under weight quantization;
  host kernels at the stored module's dtype, inputs gathered at float32 and cast to it, a kernel without a floating
  tensor at float32 (the record's value, never the process default dtype); the program's input at float32 (K9's
  spiking read and the randomized executors' default); segment-boundary buffers at float64 (the HCM reference's, as
  SANA-FE, ODIN and TTFS assemble them); the final read at float32; each chip backend declares its own integer form.

Device: the deployed host, the CPU (``deployed_host_device``: a CPU device asked for is kept as named, every other
device is the CPU). Plane (``planes``): every host kernel of the deployed program, and the fold's probe of one, runs
inside the chip-lattice measurement plane, where a lattice-armed node decides an exact tie by the exact value.
"""

from __future__ import annotations

import contextlib
import copy
import itertools
from dataclasses import dataclass, field, replace
from typing import Any, ContextManager, Dict, Mapping

import numpy as np
import torch

from mimarsinan.models.nn.lif_kernels import measurement_plane

#: A host ``ComputeOp`` (``mapping/ir/types.py`` imports this record, so the record reads an op by its ``params``).
_HostOp = Any

__all__ = [
    "DEPLOYED_HOST_DEVICE",
    "DeployedArithmetic",
    "FOLD_PROBE",
    "HOST_KERNELS",
    "HOST_STORED_MODULE",
    "HOST_TWIN",
    "MEASUREMENT_PLANE",
    "SPIKING_ARITHMETIC",
    "SPIKING_DOMAIN",
    "VALUE_ARITHMETIC",
    "VALUE_DOMAIN",
    "arithmetic_of_domain",
    "computeop_deployment_dtype",
    "deployed_host_device",
    "dtype_name",
    "numpy_dtype",
]

#: The device of the deployed read: every evaluation of a deployed program runs here.
DEPLOYED_HOST_DEVICE = "cpu"

VALUE_DOMAIN = "value"
SPIKING_DOMAIN = "spiking"

#: Host-kernel rules: the module's twin at the record's host dtype, or the stored module itself.
HOST_TWIN = "twin"
HOST_STORED_MODULE = "stored module"

#: Evaluations a plane is recorded for, and the plane names.
HOST_KERNELS = "host kernels"
FOLD_PROBE = "fold probe"
MEASUREMENT_PLANE = "measurement"


def deployed_host_device(asked: Any = None) -> str:
    """The device an evaluation of the deployed program runs on, whatever device ``asked`` names (the pipeline's): the
    deployed read's. A CPU device asked for is kept as named (``cpu``); every other device, or none, is the CPU."""
    if asked is not None and torch.device(asked).type == DEPLOYED_HOST_DEVICE:
        return str(torch.device(asked))
    return DEPLOYED_HOST_DEVICE


def dtype_name(dtype: Any) -> str:
    """``float64`` for ``torch.float64`` or ``np.float64``: the name the static check's tables and the logs print."""
    return str(dtype).removeprefix("torch.") if isinstance(dtype, torch.dtype) else np.dtype(dtype).name


def numpy_dtype(dtype: torch.dtype) -> np.dtype:
    """The numpy form of a record dtype (the numpy executors' buffers)."""
    return np.dtype(dtype_name(dtype))


def _stored_floating_dtype(module: Any) -> "torch.dtype | None":
    if module is None or not hasattr(module, "parameters"):
        return None
    buffers = module.buffers() if hasattr(module, "buffers") else ()
    for tensor in itertools.chain(module.parameters(), buffers):
        if tensor.dtype.is_floating_point:
            return tensor.dtype
    return None


@dataclass(frozen=True)
class DeployedArithmetic:
    """One domain's deployed arithmetic (module docstring)."""

    domain: str
    device: str
    core_dtype: torch.dtype
    host_kernels: str
    host_gather_dtype: torch.dtype
    parameterless_host_dtype: torch.dtype
    input_dtype: torch.dtype
    boundary_dtype: torch.dtype
    final_read_dtype: torch.dtype
    integer_form: str
    planes: Mapping[str, str] = field(default_factory=dict)

    @property
    def integer_reach(self) -> int:
        """Every integer below it is exact in the core dtype (2^53 for float64: ``FP64_EXACT_BOUND``, H8's reach)."""
        return 1 << (np.finfo(numpy_dtype(self.core_dtype)).nmant + 1)

    @property
    def core_np(self) -> np.dtype:
        return numpy_dtype(self.core_dtype)

    @property
    def boundary_np(self) -> np.dtype:
        return numpy_dtype(self.boundary_dtype)

    @property
    def final_read_np(self) -> np.dtype:
        return numpy_dtype(self.final_read_dtype)

    @property
    def input_np(self) -> np.dtype:
        return numpy_dtype(self.input_dtype)

    def host_kernel_dtype(self, op: _HostOp) -> torch.dtype:
        """The dtype ``op``'s host kernel computes in when deployed: the twin's (value) or the stored module's floating
        dtype, else the record's dtype for a kernel without a floating tensor (spiking)."""
        if self.host_kernels == HOST_TWIN:
            return self.host_gather_dtype
        stored = _stored_floating_dtype((getattr(op, "params", None) or {}).get("module"))
        return stored if stored is not None else self.parameterless_host_dtype

    def host_kernel_module(self, op: _HostOp, twins: Dict[int, Any]) -> Any:
        """The module ``op`` runs when deployed: its twin at the record's host dtype (value; cached in ``twins`` by op
        id, never mutating the stored module) or the stored module (spiking)."""
        module = (getattr(op, "params", None) or {}).get("module")
        if module is None or self.host_kernels != HOST_TWIN:
            return module
        twin = twins.get(int(op.id))
        if twin is None:
            dtype = self.host_gather_dtype
            # Every floating tensor cast, every other one kept: ``Module.double()``'s rule at the record's dtype.
            twin = copy.deepcopy(module)._apply(lambda t: t.to(dtype) if t.is_floating_point() else t)
            twins[int(op.id)] = twin
        return twin

    def counterfactual(self, dtype: torch.dtype) -> "DeployedArithmetic":
        """A declared counterfactual executor of this domain at ``dtype`` throughout (the H8 arm's float32 value
        executor): the stored module, every buffer at ``dtype``. Never the deployed program's arithmetic."""
        return replace(self, domain=f"{self.domain} counterfactual {dtype_name(dtype)}", core_dtype=dtype,
                       host_kernels=HOST_STORED_MODULE, host_gather_dtype=dtype, parameterless_host_dtype=dtype,
                       input_dtype=dtype, boundary_dtype=dtype, final_read_dtype=dtype)

    def plane(self, evaluation: str) -> ContextManager:
        """The tie-rule plane ``evaluation`` runs in (``planes``): the measurement plane, or none."""
        return measurement_plane() if self.planes.get(evaluation) == MEASUREMENT_PLANE else contextlib.nullcontext()

    def line(self) -> str:
        names = {k: dtype_name(getattr(self, k)) for k in ("core_dtype", "host_gather_dtype", "input_dtype",
                                                         "boundary_dtype", "final_read_dtype")}
        return (f"{self.domain}: device={self.device} cores={names['core_dtype']} host kernels={self.host_kernels} "
                f"(gathered at {names['host_gather_dtype']}) input={names['input_dtype']} "
                f"boundary={names['boundary_dtype']} final read={names['final_read_dtype']} "
                f"integer form={self.integer_form}")


_PLANES = {HOST_KERNELS: MEASUREMENT_PLANE, FOLD_PROBE: MEASUREMENT_PLANE}

VALUE_ARITHMETIC = DeployedArithmetic(
    domain=VALUE_DOMAIN, device=DEPLOYED_HOST_DEVICE, core_dtype=torch.float64, host_kernels=HOST_TWIN,
    host_gather_dtype=torch.float64, parameterless_host_dtype=torch.float64, input_dtype=torch.float64,
    boundary_dtype=torch.float64, final_read_dtype=torch.float64, integer_form="int64", planes=_PLANES)

SPIKING_ARITHMETIC = DeployedArithmetic(
    domain=SPIKING_DOMAIN, device=DEPLOYED_HOST_DEVICE, core_dtype=torch.float64, host_kernels=HOST_STORED_MODULE,
    host_gather_dtype=torch.float32, parameterless_host_dtype=torch.float32, input_dtype=torch.float32,
    boundary_dtype=torch.float64, final_read_dtype=torch.float32, integer_form="backend", planes=_PLANES)


def arithmetic_of_domain(value_domain: bool) -> DeployedArithmetic:
    """The record of the value domain or of the spiking domains."""
    return VALUE_ARITHMETIC if value_domain else SPIKING_ARITHMETIC


def computeop_deployment_dtype(op: _HostOp) -> torch.dtype:
    """The dtype a spiking program's host kernel computes in when deployed (``SPIKING_ARITHMETIC.host_kernel_dtype``):
    the stored module's floating dtype, else float32, the record's value for a kernel without a floating tensor."""
    return SPIKING_ARITHMETIC.host_kernel_dtype(op)
