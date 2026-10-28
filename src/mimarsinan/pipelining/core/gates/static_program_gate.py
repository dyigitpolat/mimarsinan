"""[W15] The static per-deployment check at Hard Core Mapping, beside the sampled certificates.

Input-independent and read-only: the built program against its identity program, Theorem 1 and Proposition 2 read
apart, the check's wall against the program's compile wall. It never refuses a run; its two certificate records go
into the deployment record beside the sampled run's, and the full report into the run directory. A check that cannot
run records why by name (its exception), never a pass."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from mimarsinan.common.best_effort import best_effort
from mimarsinan.certification import program_check
from mimarsinan.chip_simulation.backend import BACKEND_REGISTRY
from mimarsinan.chip_simulation.core_semantics import is_mvm_core_semantics
from mimarsinan.chip_simulation.simulation_runner.core import nevresim_weight_type
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.schema import CertificateRecord
from mimarsinan.mapping.ir.core_law import graph_carries_laws
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

REPORT_FILENAME = "static_program_check.json"
#: What a check that cannot read a program raises: a refusal by name, or a lookup, type or value error on an element
#: it reads. Recorded as a failed check with the exception named; anything else (memory, interrupts) propagates.
CHECK_ERRORS = (ToolchainRefusal, ArithmeticError, AssertionError, AttributeError, LookupError, RuntimeError,
                TypeError, ValueError)


def _error_records(exc: BaseException) -> List[Dict[str, Any]]:
    detail = json.dumps({"verdict": f"failed: check error {type(exc).__name__}: {str(exc)[:300]}"}, sort_keys=True)
    return [CertificateRecord(name=f"static_program_check_{theorem}", backend="static", passed=False,
                              neuron_windows_compared=0, exact_match_fraction=0.0, max_abs_delta=0.0,
                              detail=detail).to_dict() for theorem in ("theorem1", "proposition2")]


def backend_accumulation(pipeline: Any) -> Dict[str, str]:
    """The enabled backends whose charges must be exact as well, each with its accumulation dtype: nevresim where its
    step applies to a spiking plan (C++ ``int`` on a weight-quantized chip, ``double`` otherwise)."""
    plan = DeploymentPlan.of(pipeline)
    if is_mvm_core_semantics(plan.core_semantics) or not BACKEND_REGISTRY.get("nevresim").applies(plan):
        return {}
    return {"nevresim": "int32" if nevresim_weight_type(pipeline.config) is int else "float64"}


def run_static_program_check(
    pipeline: Any, ir_graph: Any, hybrid_mapping: Any, *, cores: Any, compile_wall_s: Optional[float] = None,
    compile_source: str = "built", unbounded_ir: Any = None, unbounded_note: Optional[str] = None,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """``(report or None, certificate record dicts)`` for the run's built program; a mapping loaded from the cache
    (``compile_source`` "loaded") is recorded as loaded, its load wall never priced as a compile. [E1 open item 1]
    ``unbounded_ir`` (Soft Core Mapping's second mapping, without core limits) is the reference of ``ir-tiling``, so
    the verdict establishes the IR tiling; without it the condition stays assumed, with ``unbounded_note`` (why the
    second mapping is absent) in its note."""
    try:
        declaration = program_check.declaration_from_config(pipeline.config, cores=cores,
                                                            has_laws=graph_carries_laws(ir_graph),
                                                            backends=backend_accumulation(pipeline))
        report = program_check.check_deployment(hybrid_mapping, ir_graph=ir_graph, declaration=declaration,
                                                compile_wall_s=compile_wall_s, compile_source=compile_source,
                                                unbounded_ir=unbounded_ir)
    except CHECK_ERRORS as exc:  # recorded by name as a failed check, never a pass
        print(f"[StaticProgramCheck] the check did not run: {type(exc).__name__}: {exc}")
        return None, _error_records(exc)
    tiling = report.results.get("ir-tiling")
    if unbounded_ir is None and unbounded_note and tiling is not None:
        tiling.note = f"{tiling.note} (the IR at unbounded geometry: {unbounded_note})"
    print(report.summary())
    with best_effort("write the static program check report"):
        with open(os.path.join(pipeline.working_directory, REPORT_FILENAME), "w") as handle:
            json.dump(report.to_dict(), handle, indent=1, sort_keys=True)
    return report, report.certificate_records()
