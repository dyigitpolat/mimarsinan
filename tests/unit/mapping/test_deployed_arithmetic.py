"""[U-DAR] One record of the deployed arithmetic (``mapping/ir/deployment_dtype.py``), read by every evaluation that
stands for the deployed program: tests T-1 to T-10 (diagnosis.md section 6.5; diagnosis_verify.md M-1, M-2, M-3).

T-1 to T-5 and T-9, T-10 read the fold through the paths Soft Core Mapping runs (``compute_global_pruned_sets``,
``prune_ir_graph``) and the deployed read's value executor, on the constructed vehicles of
``deployed_arithmetic_vehicles``; T-6 checks that each reader's dtype and device is the record's; T-7 is the AST ratchet
over the executor modules (module-level float dtype constants anywhere in src outside the record; float dtype literals
in keyword defaults, casts and ``dtype=`` keywords in the executor modules, each remaining one classified).
"""

from __future__ import annotations

import ast
import inspect
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.certification.value_certificate import VALUE_TWIN_FP64_ATOL
from mimarsinan.chip_simulation.core_semantics import INERT_SPIKING_MODE
from mimarsinan.mapping.ir import ComputeOp, IRGraph, NeuralCore
from mimarsinan.mapping.pruning.graph import compute_global_pruned_sets
from mimarsinan.mapping.pruning.ir_pruning_helpers import _boundary_policy_exemptions
from unit.mapping import deployed_arithmetic_vehicles as V
from unit.mapping.deployed_arithmetic_ratchet import CLASSES, CLASSIFIED

VALUE_MODE = INERT_SPIKING_MODE
SRC = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"


def _gelu(x: float, dtype: torch.dtype) -> float:
    return float(nn.GELU()(torch.tensor([x], dtype=dtype)))


# --------------------------------------------------------------------------------------------- T-1 to T-5: the fold
def test_t1_a_core_column_folds_at_the_deployed_arithmetic():
    """V1: a carrier-only core [[1.0]] at threshold 3 emits 1/3 as the deployed read computes it, in float64
    (0x1.5555555555555p-2); BASE folds float32(1/3) = 0.3333333432674408."""
    reading = V.fold_and_read(V.v1_core_column(), VALUE_MODE)
    assert reading.lattice["0:0"] == float.fromhex("0x1.5555555555555p-2") == V.ONE_THIRD
    assert reading.folded_rows == {2: {0: V.ONE_THIRD}}


def test_t2_a_host_constant_folds_at_the_value_twin():
    """V2: GELU(0.25) folds to the float64 twin's value, the value executor's (BASE: the float32 module's)."""
    want, stored = _gelu(0.25, torch.float64), _gelu(0.25, torch.float32)
    assert want != stored
    reading = V.fold_and_read(V.v2_host_op(), VALUE_MODE)
    assert reading.lattice["1:0"] == want


def test_t2_a_spiking_host_constant_is_the_stored_modules():
    """Guard (the value is BASE's): in a spiking domain the probe evaluates the stored float32 module."""
    from mimarsinan.mapping.ir.deployment_dtype import SPIKING_ARITHMETIC
    from mimarsinan.mapping.pruning.liveness_transfer import derive_constant_outputs
    from mimarsinan.mapping.pruning.liveness_transfer.transfer_registry import derive_liveness_transfer

    op = V.v2_host_op().nodes[1]
    out = derive_constant_outputs(op, derive_liveness_transfer(op), [0.25], arithmetic=SPIKING_ARITHMETIC)
    assert out == {0: _gelu(0.25, torch.float32)}


@pytest.mark.parametrize("build", [V.v1_core_column, V.v2_host_op], ids=["V1", "V2"])
def test_t3_the_folded_program_equals_the_unfolded_one(build):
    """Folded and unpruned programs through ``ValueHybridCoreFlow(..., "cpu", float64)``: at most the float64 twin's
    tolerance apart (BASE: 7.45e-09 on V1, 3.87e-09 on V2)."""
    reading = V.fold_and_read(build(), VALUE_MODE)
    assert reading.max_abs_delta <= VALUE_TWIN_FP64_ATOL


def test_t4_no_class_flips_at_a_margin_below_float32():
    """V3: the head's logits 2^-40 apart in the deployed read's arithmetic (BASE: 16 of 16 classes flip)."""
    assert V.fold_and_read(V.v3_class_flip(), VALUE_MODE).classes_differing == 0


def test_t5_a_line_the_deployment_keeps_is_not_killed_in_the_value_domain():
    """V4: the bias line is 2^-30 exactly (0 in float32): it folds, the head's row survives, outputs equal (BASE: the
    line descends CONST(0), the row dies, max |delta| 2.0, 16 of 16 classes flip)."""
    reading = V.fold_and_read(V.v4_zero_channel(), VALUE_MODE)
    assert reading.lattice["0:0"] == math.ldexp(1.0, -30)
    assert reading.max_abs_delta == 0.0 and reading.classes_differing == 0


def test_t5_a_line_the_deployment_keeps_is_not_killed_in_a_spiking_domain():
    """V4 under the LIF lattice (CONST(0) only): the line is not CONST(0), so the bias core and the head's row reading
    it stay (BASE: the line descends CONST(0), the bias core is deleted, 1 core left)."""
    reading = V.fold_and_read(V.v4_zero_channel(), "lif")
    assert reading.lattice.get("0:0") != 0.0
    assert "bias_core" in reading.core_names and reading.max_abs_delta == 0.0 and reading.classes_differing == 0


# ---------------------------------------------------------------------------------------------- T-9: the probe memo
def _t9_graph() -> IRGraph:
    """A parameterless GELU host op reading the always-on source (known in every domain), then a head."""
    op = ComputeOp(id=0, name="host", input_sources=V.srcs([(-3, 0)]), op_type="GELU",
                   params={"module": nn.GELU().eval(), "input_shape": (1,)}, input_shape=(1,), output_shape=(1,))
    head = NeuralCore(id=1, name="head", input_sources=V.srcs([(0, 0), (-2, 0), (-2, 1), (-3, 0)]),
                      core_matrix=np.array([[1.0, 0.0], [0.5, 0.5], [0.5, 0.5], [0.0, 0.5]]),
                      threshold=1.0, latency=0)
    return IRGraph(nodes=[op, head], output_sources=V.srcs([(1, 0), (1, 1)]))


def _lattice(graph: IRGraph, spiking_mode: str, memo: dict) -> dict:
    exempt_rows, exempt_cols = _boundary_policy_exemptions(graph)
    analysis = compute_global_pruned_sets(graph, exempt_rows_per_node=exempt_rows, exempt_cols_per_node=exempt_cols,
                                          mode="cascade", elimination_constant_folding="full",
                                          spiking_mode=spiking_mode, probe_memo=memo)
    return {f"{k[0]}:{k[1]}": float(v) for k, v in analysis.constant_folds.lattice.snapshot().items()}


def test_t9_a_probe_memo_filled_under_the_spiking_arithmetic_is_not_replayed_under_the_value_one():
    """One memo shared across a LIF analysis and a value one: the value run derives the float64 twin's GELU(1), its
    own battery under its own key (BASE keys by op and input bits only: one entry, the float32 battery replayed)."""
    memo: dict = {}
    _lattice(_t9_graph(), "lif", memo)
    assert len(memo) == 1, "the LIF analysis probed the host op (its input is the always-on source)"
    want = _gelu(1.0, torch.float64)
    assert want != _gelu(1.0, torch.float32)
    assert _lattice(_t9_graph(), VALUE_MODE, memo)["0:0"] == want
    assert len(memo) == 2, "one battery per domain of the deployed arithmetic"


# ------------------------------------------------------------------------------------------------ T-10: the plane
@pytest.mark.parametrize("domain", ["value", "spiking"])
def test_t10_the_probe_derives_what_the_deployed_host_evaluator_computes(domain):
    """V5's host op on CONST(1.0): every deployed host evaluator opens the measurement plane and gives 0.25; the fold's
    probe evaluates through the same seam (BASE: 0.0, outside the plane)."""
    from mimarsinan.mapping.ir.deployment_dtype import SPIKING_ARITHMETIC, VALUE_ARITHMETIC
    from mimarsinan.mapping.pruning.liveness_transfer import derive_constant_outputs
    from mimarsinan.mapping.pruning.liveness_transfer.transfer_registry import derive_liveness_transfer

    arithmetic = VALUE_ARITHMETIC if domain == "value" else SPIKING_ARITHMETIC
    op = V.v5_host_op()
    assert derive_constant_outputs(op, derive_liveness_transfer(op), [1.0], arithmetic=arithmetic) == {0: 0.25}


@pytest.mark.parametrize("spiking_mode", [VALUE_MODE, "lif"])
def test_t10_a_lattice_tie_on_a_host_op_folds_as_deployed(spiking_mode):
    """V5 folded: the host op's line is the deployed 0.25, never CONST(0), and no output moves (BASE: CONST(0), the
    head's row killed, max |delta| 1.0, 16 of 16 classes flip, in both domains)."""
    reading = V.fold_and_read(V.v5_plane(), spiking_mode)
    if spiking_mode == VALUE_MODE:   # the line is CONST(0.25): folded onto the head's carrier, exactly
        assert reading.lattice["0:0"] == 0.25 and reading.folded_rows == {2: {0: 0.25}}
    else:                            # CONST(c != 0) is not admitted: the line stays TOP, the row stays
        assert reading.lattice.get("0:0") != 0.0 and reading.head_rows == 4
    assert reading.classes_differing == 0 and reading.max_abs_delta == 0.0


# ----------------------------------------------------------------------------------------------- T-6: one source
def test_t6_the_record_per_domain():
    from mimarsinan.mapping.ir import deployment_dtype as rec

    v, s = rec.VALUE_ARITHMETIC, rec.SPIKING_ARITHMETIC
    assert (v.device, s.device) == (rec.DEPLOYED_HOST_DEVICE, rec.DEPLOYED_HOST_DEVICE) == ("cpu", "cpu")
    assert v.core_dtype is torch.float64 and s.core_dtype is torch.float64
    assert v.integer_reach == s.integer_reach == 1 << 53
    assert (v.host_gather_dtype, v.input_dtype, v.boundary_dtype, v.final_read_dtype) == (torch.float64,) * 4
    # verify M-3, at today's values: a parameterless spiking kernel at float32, K9's spiking input at float32
    assert s.parameterless_host_dtype is torch.float32 and s.input_dtype is torch.float32
    assert s.host_gather_dtype is torch.float32 and s.boundary_dtype is torch.float64
    assert s.final_read_dtype is torch.float32
    for a in (v, s):
        assert a.planes[rec.HOST_KERNELS] == a.planes[rec.FOLD_PROBE] == rec.MEASUREMENT_PLANE
    assert rec.arithmetic_of_domain(True) is v and rec.arithmetic_of_domain(False) is s


def test_t6_the_host_kernel_rule():
    from mimarsinan.mapping.ir import deployment_dtype as rec

    stored = V.v2_host_op().nodes[1]                      # parameterless GELU
    with_params = ComputeOp(id=0, name="lin", input_sources=V.srcs([(-3, 0)]), op_type="Linear",
                            params={"module": nn.Sequential(nn.Linear(1, 1, bias=False)).eval(), "input_shape": (1,)},
                            input_shape=(1,), output_shape=(1,))   # float32 Linear
    assert rec.SPIKING_ARITHMETIC.host_kernel_dtype(stored) is torch.float32
    assert rec.SPIKING_ARITHMETIC.host_kernel_dtype(with_params) is torch.float32
    assert rec.VALUE_ARITHMETIC.host_kernel_dtype(stored) is torch.float64
    assert rec.computeop_deployment_dtype(with_params) is rec.SPIKING_ARITHMETIC.host_kernel_dtype(with_params)
    twins: dict = {}
    twin = rec.VALUE_ARITHMETIC.host_kernel_module(with_params, twins)
    assert twin is not with_params.params["module"] and twins == {0: twin}
    assert all(p.dtype is torch.float64 for p in twin.parameters())
    assert with_params.params["module"][0].weight.dtype is torch.float32, "the stored module is never mutated"
    assert rec.SPIKING_ARITHMETIC.host_kernel_module(with_params, twins) is with_params.params["module"]


def test_t6_the_fold_reads_the_record():
    from mimarsinan.mapping.ir.deployment_dtype import SPIKING_ARITHMETIC, VALUE_ARITHMETIC
    from mimarsinan.mapping.pruning.graph.pruning_graph_seeding import _build_constant_state

    for mode, want in ((VALUE_MODE, VALUE_ARITHMETIC), ("lif", SPIKING_ARITHMETIC), ("ttfs", SPIKING_ARITHMETIC)):
        state = _build_constant_state(elimination_constant_folding="full", computeop_liveness_transfers="full",
                                      spiking_mode=mode)
        assert state.arithmetic is want, mode


def test_t6_the_plan_selects_the_domain():
    from mimarsinan.chip_simulation import deployed_host
    from mimarsinan.mapping.ir import deployment_dtype as rec

    assert deployed_host.deployed_arithmetic(VALUE_MODE) is rec.VALUE_ARITHMETIC
    for mode in ("lif", "ttfs", "ttfs_cycle_based", "ttfs_quantized"):
        assert deployed_host.deployed_arithmetic(mode) is rec.SPIKING_ARITHMETIC
    # U-HDA's deployed host reads the record: the device rule and the device are the record's own objects
    assert deployed_host.DEPLOYED_HOST_DEVICE is rec.DEPLOYED_HOST_DEVICE
    assert deployed_host.deployed_host_device is rec.deployed_host_device
    assert not hasattr(deployed_host, "DEPLOYED_VALUE_DTYPE"), "the record's value dtype, not a copy"
    assert deployed_host.value_host_arithmetic("k", "cuda").dtype == rec.dtype_name(rec.VALUE_ARITHMETIC.core_dtype)


def test_t6_the_census_reads_the_record():
    from types import SimpleNamespace

    from mimarsinan.mapping.ir import deployment_dtype as rec
    from mimarsinan.pipelining.core.census import arithmetic as census

    assert not hasattr(census, "VALUE_CENSUS_DTYPE") and not hasattr(census, "DEPLOYED_READ_DEVICE")
    pipeline = SimpleNamespace(config={"device": "cuda"})
    assert census.value_census_arithmetic(pipeline, "c").dtype == rec.dtype_name(rec.VALUE_ARITHMETIC.core_dtype)
    assert census.value_census_arithmetic(pipeline, "c").device == rec.DEPLOYED_HOST_DEVICE
    assert census.spiking_census_arithmetic(pipeline, "c", None).dtype.startswith(
        f"cores {rec.dtype_name(rec.SPIKING_ARITHMETIC.core_dtype)}")


def test_t6_the_static_check_declares_the_records_dtypes():
    from mimarsinan.certification.program_check import declaration as d
    from mimarsinan.mapping.ir.deployment_dtype import SPIKING_ARITHMETIC, VALUE_ARITHMETIC, dtype_name

    assert d.DEFAULT_ACCUMULATION == {"hcm": dtype_name(SPIKING_ARITHMETIC.core_dtype)}
    assert d.VALUE_ACCUMULATION == {"value_census": dtype_name(VALUE_ARITHMETIC.core_dtype),
                                    "int64_golden": VALUE_ARITHMETIC.integer_form}
    assert d.VALUE_UPLOAD == {"value_census": dtype_name(VALUE_ARITHMETIC.core_dtype)}
    assert d.EXACT_INTEGER_REACH[dtype_name(VALUE_ARITHMETIC.core_dtype)] == VALUE_ARITHMETIC.integer_reach


def test_t6_the_spiking_core_dtype_and_its_numpy_forms_are_the_records():
    from mimarsinan.chip_simulation.lava_loihi import timing as lava_timing
    from mimarsinan.chip_simulation.odin_fpga import segment as odin_segment
    from mimarsinan.chip_simulation.sanafe.runner import constants as sanafe_constants
    from mimarsinan.chip_simulation.ttfs import ttfs_executor
    from mimarsinan.mapping.ir.deployment_dtype import SPIKING_ARITHMETIC
    from mimarsinan.models.spiking import spiking_config
    from mimarsinan.models.spiking.hybrid import flow

    assert spiking_config.COMPUTE_DTYPE is SPIKING_ARITHMETIC.core_dtype
    assert flow._COMPUTE_DTYPE is SPIKING_ARITHMETIC.core_dtype
    assert sanafe_constants._COMPUTE_DTYPE == SPIKING_ARITHMETIC.boundary_np
    assert odin_segment.COMPUTE_DTYPE == SPIKING_ARITHMETIC.boundary_np
    assert ttfs_executor._CONTRACT_DTYPE == SPIKING_ARITHMETIC.boundary_np
    assert lava_timing._LAVA_DTYPE == SPIKING_ARITHMETIC.core_np


def test_t6_the_randomized_executors_read_the_record():
    """Verify M-3: the randomized spiking input default is the record's float32; the H8 arm keeps its declared float32
    counterfactual and only its paired control reads the record."""
    from mimarsinan.certification.randomized import arms, executors
    from mimarsinan.mapping.ir.deployment_dtype import SPIKING_ARITHMETIC, VALUE_ARITHMETIC, dtype_name

    assert inspect.signature(executors.run_nodes).parameters["value_dtype"].default is VALUE_ARITHMETIC.core_dtype
    assert arms.Arm("x", "x", "x").value_dtype == dtype_name(VALUE_ARITHMETIC.core_dtype)
    h8 = arms.ARMS["H8"]
    assert h8.value_dtype == "float32", "the H8 arm's declared counterfactual"
    assert h8.control().value_dtype == dtype_name(VALUE_ARITHMETIC.core_dtype)
    assert inspect.signature(executors.input_tensor).parameters["dtype"].default is SPIKING_ARITHMETIC.input_dtype


def test_t6_the_value_executor_and_the_host_seam_read_the_record():
    from mimarsinan.chip_simulation.hybrid_run import hybrid_execution as he
    from mimarsinan.mapping.ir.deployment_dtype import SPIKING_ARITHMETIC, VALUE_ARITHMETIC

    final = he.gather_final_output_numpy(V.srcs([(-2, 0), (-3, 0)]), {}, np.array([[0.1]]), 1,
                                         dtype=SPIKING_ARITHMETIC.final_read_np)
    assert final.dtype == SPIKING_ARITHMETIC.final_read_np
    seg = he.assemble_segment_input_numpy([], {}, 2, dtype=VALUE_ARITHMETIC.boundary_np)
    assert seg.dtype == VALUE_ARITHMETIC.boundary_np


def test_t6_the_new_readers_name_the_record():
    """Verify M-2's readers read the record (each line names its field): the spiking final read, TTFS's final read,
    Lava's segment-output rates and input, the randomized spiking input."""
    reads = {
        "models/spiking/hybrid/rate_forward.py": "SPIKING_ARITHMETIC.final_read_dtype",
        "models/spiking/hybrid/ttfs_step.py": "SPIKING_ARITHMETIC.final_read_np",
        "chip_simulation/lava_loihi/segment_runner.py": "SPIKING_ARITHMETIC.boundary_np",
        "chip_simulation/lava_loihi/runner.py": "SPIKING_ARITHMETIC.input_dtype",
        "certification/randomized/executors.py": "SPIKING_ARITHMETIC.input_dtype",
    }
    for rel, name in reads.items():
        assert name in (SRC / rel).read_text(), rel


# --------------------------------------------------------------------------------------------- T-7: the ratchet
#: The executor modules: every module that evaluates a deployed program or a part of one, or claims its value.
EXECUTOR_MODULES = (
    "chip_simulation/hybrid_run", "chip_simulation/value_run", "chip_simulation/simulation_runner",
    "chip_simulation/lava_loihi", "chip_simulation/sanafe/runner", "chip_simulation/odin_fpga",
    "chip_simulation/ttfs", "chip_simulation/neural_segment_executor.py", "chip_simulation/deployed_host.py",
    "models/spiking/hybrid", "spiking/segment_boundary.py", "spiking/count_grid.py", "spiking/boundary_config.py",
    "certification/randomized/executors.py", "mapping/ir/types.py", "mapping/ir/gather_plan.py",
    "mapping/pruning/liveness_transfer", "pipelining/core/census", "pipelining/core/gates/value_gates.py",
    "torch_mapping/value_parity",
)
RECORD = "mapping/ir/deployment_dtype.py"
_FLOAT_ATTRS = {"float32", "float64", "float", "double", "float16", "half", "bfloat16"}



def _literal(node):
    if isinstance(node, ast.Attribute) and node.attr in _FLOAT_ATTRS and isinstance(node.value, ast.Name) \
            and node.value.id in ("torch", "np", "numpy"):
        return f"{node.value.id}.{node.attr}"
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in ("float32", "float64"):
        return repr(node.value)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "dtype" \
            and isinstance(node.func.value, ast.Name) and node.func.value.id in ("np", "numpy") and node.args:
        inner = _literal(node.args[0])     # np.dtype(np.float64): the same literal, wrapped
        return f"np.dtype({inner})" if inner else None
    return None


def _executor_files():
    out = []
    for scope in EXECUTOR_MODULES:
        p = SRC / scope
        out += [p] if p.is_file() else sorted(p.rglob("*.py"))
    return [f for f in out if "__pycache__" not in f.parts and f.relative_to(SRC).as_posix() != RECORD]


def _scan(path: Path):
    """(function, kind) per float dtype literal in a keyword default, a cast or a ``dtype=`` keyword."""
    tree, found = ast.parse(path.read_text()), []
    add = found.extend

    def visit(node, qual):
        for child in ast.iter_child_nodes(node):
            q = qual
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = f"{qual}.{child.name}" if qual else child.name
                if not isinstance(child, ast.ClassDef):
                    a = child.args
                    pos = a.posonlyargs + a.args
                    pairs = list(zip(pos[len(pos) - len(a.defaults):], a.defaults))
                    pairs += [(x, y) for x, y in zip(a.kwonlyargs, a.kw_defaults) if y is not None]
                    add((q, f"default {x.arg}={_literal(y)}") for x, y in pairs if _literal(y))
            if isinstance(child, ast.Call):
                fn = child.func
                if isinstance(fn, ast.Attribute) and fn.attr in ("to", "astype", "type"):
                    add((qual, f"cast .{fn.attr}({_literal(a0)})") for a0 in child.args if _literal(a0))
                if isinstance(fn, ast.Attribute) and fn.attr in ("float", "double") and not child.args \
                        and not child.keywords:
                    add([(qual, f"cast .{fn.attr}()")])
                add((qual, f"dtype= {_literal(k.value)}") for k in child.keywords
                    if k.arg == "dtype" and _literal(k.value))
            visit(child, q)

    visit(tree, "")
    return found


def test_t7_no_module_level_float_dtype_constant_outside_the_record():
    """An ALLCAPS module name bound to one float dtype literal anywhere in src is a second source of the arithmetic
    (BASE: 7 of them, U-HDA's an eighth)."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if "__pycache__" in path.parts or rel == RECORD:
            continue
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None and _literal(node.value) \
                    and _literal(node.value) != "'float32'" and _literal(node.value) != "'float64'":
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                found += [f"{rel}:{node.lineno} {t.id}" for t in targets
                          if isinstance(t, ast.Name) and t.id.lstrip("_").isupper()]
    assert found == [], found


def test_t7_every_float_dtype_literal_on_an_executor_path_is_classified():
    found = Counter()
    for path in _executor_files():
        rel = path.relative_to(SRC).as_posix()
        for qual, kind in _scan(path):
            found[(rel, qual, kind)] += 1
    table = Counter({key: count for key, (count, _cls, _why) in CLASSIFIED.items()})
    unclassified = {k: n - table.get(k, 0) for k, n in found.items() if n > table.get(k, 0)}
    stale = {k: n - found.get(k, 0) for k, n in table.items() if n > found.get(k, 0)}
    assert not unclassified, f"float dtype literals on an executor path, unclassified: {sorted(unclassified)}"
    assert not stale, f"classified entries no longer present (shrink the table): {sorted(stale)}"
    assert all(cls in CLASSES and why for _count, cls, why in CLASSIFIED.values())


def test_t7_the_record_is_the_only_float_dtype_source_of_the_deployed_seams():
    """No keyword default anywhere on an executor path carries a float dtype literal: every seam's dtype is declared
    (T-8) or read from the record."""
    defaults = [(rel, q, k) for (rel, q, k) in CLASSIFIED if k.startswith("default ")]
    assert defaults == [], defaults
