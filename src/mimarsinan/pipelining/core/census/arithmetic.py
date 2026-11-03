"""[U-CEN] The arithmetic every census of the deployed program runs in: the one place it is decided.

A census is the toolchain's own read of the deployed program over the eval set: Soft Core Mapping's identity census
and Hard Core Mapping's packed census, each as a value program (``value_gates``) or as a spiking program
(``simulation_factory``). Its count is printed as the step metric (``Hard-core Spiking Simulation Test``), carried into
the deployment record, and compared with the deployed read by the harvests' certificates (K9 is the deployed read,
K14 the census against it). A census therefore runs the deployed read's own arithmetic, so its count is the deployed
program's count by construction, not by agreement:

- **value programs** at float64 (``VALUE_CENSUS_DTYPE``): the arithmetic of every value certificate (the R-edge and
  C-edge gates, ``value_gates``), of the deployed read (``ValueHybridCoreFlow(mapping, "cpu", float64)``) and the one
  in which the int64 golden reads an integer program exactly (K6). Integer laws, lattice codes and every weight then
  hold exactly (below 2^53, the static check's H8 reach for the census's declared upload dtype), and host kernels run
  their float64 twins;
- **spiking programs** in the spiking executor's own arithmetic, unchanged: cores at ``COMPUTE_DTYPE`` (float64, on
  the integer membrane lattice under weight quantization) and host kernels at their deployment dtype
  (``computeop_deployment_dtype``, the stored module's);
- **device**: the pipeline's device where it is the CPU, else the CPU (``census_device``). The deployed read runs on
  the CPU. A CUDA device holds neither its float32 host kernels (cuDNN and cuBLAS choose other algorithms and
  reduction orders) nor float64 reductions and transcendental kernels bit for bit, so a census there would agree with
  the deployed read by measurement, never by construction. A census asked for on a CUDA pipeline runs on the CPU, and
  the modules it borrows (the model, the program's host modules) are handed back where they were
  (``borrowed_placement``).

Each census prints its device and dtype in the step's log (``[Census] ...``) and emits a ``census_arithmetic`` event.
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
from typing import Any, Iterator

import torch

from mimarsinan.chip_simulation.deployed_host import (
    DEPLOYED_HOST_DEVICE,
    DEPLOYED_VALUE_DTYPE,
    borrowed_host_placement,
    deployed_host_device,
    program_host_ops,
)
from mimarsinan.common.reporter import emit_reporter_event
from mimarsinan.mapping.ir.deployment_dtype import computeop_deployment_dtype
from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE

#: The dtype a value census evaluates the deployed value program in: the deployed read's ([U-HDA] its SSOT,
#: ``chip_simulation/deployed_host.py``).
VALUE_CENSUS_DTYPE = DEPLOYED_VALUE_DTYPE
#: The device of the deployed read.
DEPLOYED_READ_DEVICE = DEPLOYED_HOST_DEVICE


def census_device(pipeline_device: Any) -> str:
    """The device a census runs on: the pipeline's device where it is the CPU, else the CPU (module docstring; [U-HDA]
    the deployed host's rule, ``deployed_host_device``)."""
    return deployed_host_device(pipeline_device)


@dataclass(frozen=True)
class CensusArithmetic:
    """Where and in which dtype one census runs, beside the device the pipeline asked for."""

    census: str
    device: str
    dtype: str
    pipeline_device: str

    @property
    def moved(self) -> bool:
        """Whether the census runs off the device the pipeline asked for."""
        return torch.device(self.pipeline_device).type != torch.device(self.device).type

    def line(self) -> str:
        moved = (f"; pipeline device {self.pipeline_device}: the deployed read runs on the CPU, where its arithmetic "
                 f"is held exactly" if self.moved else "")
        return f"[Census] {self.census}: device={self.device} dtype={self.dtype}{moved}"


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def _resolved(pipeline: Any, device: Any) -> str:
    return str(device if device is not None else pipeline.config["device"])


def value_census_arithmetic(pipeline: Any, census: str, *, device: Any = None) -> CensusArithmetic:
    """The arithmetic of a value census: ``VALUE_CENSUS_DTYPE`` on ``census_device`` (``device`` defaults to the
    pipeline's)."""
    asked = _resolved(pipeline, device)
    return CensusArithmetic(census=census, device=census_device(asked), dtype=_dtype_name(VALUE_CENSUS_DTYPE),
                            pipeline_device=asked)


def spiking_census_arithmetic(pipeline: Any, census: str, hybrid_mapping: Any, *,
                              device: Any = None) -> CensusArithmetic:
    """The arithmetic of a spiking census: the executor's own (cores at ``COMPUTE_DTYPE``, each host kernel at its
    deployment dtype, named from the program's host ops) on ``census_device``."""
    asked = _resolved(pipeline, device)
    host = sorted({_dtype_name(computeop_deployment_dtype(op)) for op in _host_ops(hybrid_mapping)})
    dtype = f"cores {_dtype_name(COMPUTE_DTYPE)}, host kernels {'/'.join(host) if host else 'none'}"
    return CensusArithmetic(census=census, device=census_device(asked), dtype=dtype, pipeline_device=asked)


def _host_ops(hybrid_mapping: Any) -> list:
    return program_host_ops(hybrid_mapping)


def announce(pipeline: Any, arithmetic: CensusArithmetic) -> None:
    """Print the census's device and dtype in the step's log and emit it as a reporter event."""
    print(arithmetic.line(), flush=True)
    emit_reporter_event(getattr(pipeline, "reporter", None), "census_arithmetic", asdict(arithmetic))


@contextlib.contextmanager
def borrowed_placement(arithmetic: CensusArithmetic, model: Any, hybrid_mapping: Any) -> Iterator[None]:
    """For a census moved off the pipeline's device: restore, on exit, the placement of every module it borrows, the
    model (its preprocessor feeds the spiking executor) and each host op's module (an evaluation moves it to the
    census device). A census on the pipeline's own device moves nothing and is left as it is ([U-HDA] the deployed
    host's one restore, ``borrowed_host_placement``)."""
    with borrowed_host_placement(model, hybrid_mapping, moved=arithmetic.moved):
        yield
