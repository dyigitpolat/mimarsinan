"""Contract-owned semantics are declared, never defaulted, on executor entry points.

The comparator incident: a debug probe constructed ``SpikingHybridCoreFlow``
directly and silently inherited ``thresholding_mode="<="`` — the TTFS family's
comparator — while the run under study deploys ``"<"``, which on integer-lattice
chips is load-bearing (V9). Production was never wrong (every pipeline flow is
built from the deployment contract), but the constructors kept semantic DEFAULTS:
a second, silent source of semantics beside the SSOT — the constructor-shaped
version of the banned ``get(key, default)``. These pins keep that source removed.
"""

import inspect

import pytest


def _param(callable_, name):
    return inspect.signature(callable_).parameters[name]


REQUIRED = [
    # (entry point, axis) — each must exist and carry NO default.
    ("mimarsinan.models.spiking.hybrid.flow.SpikingHybridCoreFlow",
     "thresholding_mode"),
    ("mimarsinan.models.spiking.hybrid.identity_flow.build_identity_spiking_flow",
     "thresholding_mode"),
    ("mimarsinan.chip_simulation.nevresim.nevresim_driver.NevresimDriver",
     "thresholding_mode"),
    ("mimarsinan.chip_simulation.nevresim.nevresim_driver.NevresimDriver",
     "spike_generation_mode"),
    # [U-DAR] T-8: the deployed arithmetic's dtypes and device (the record, mapping/ir/deployment_dtype.py) are
    # declared at every executor seam, never defaulted.
    ("mimarsinan.chip_simulation.value_run.value_flow.ValueHybridCoreFlow", "device"),
    ("mimarsinan.chip_simulation.value_run.value_flow.ValueHybridCoreFlow", "dtype"),
    ("mimarsinan.chip_simulation.hybrid_run.host_compute.execute_compute_op_numpy", "dtype"),
    ("mimarsinan.chip_simulation.hybrid_run.host_compute.execute_compute_op_torch", "arithmetic"),
    ("mimarsinan.chip_simulation.hybrid_run.host_compute.evaluate_gathered_host_op", "arithmetic"),
    ("mimarsinan.chip_simulation.hybrid_run.hybrid_execution.assemble_segment_input_numpy", "dtype"),
    ("mimarsinan.chip_simulation.hybrid_run.hybrid_execution.gather_final_output_numpy", "dtype"),
    ("mimarsinan.mapping.pruning.liveness_transfer.constant_core.derive_core_constants", "core_dtype"),
    ("mimarsinan.mapping.pruning.liveness_transfer.constant_transfer.derive_constant_outputs", "arithmetic"),
    ("mimarsinan.mapping.pruning.liveness_transfer.probe_memo.derive_constant_outputs_memoized", "arithmetic"),
]

#: [U-DAR] T-8 on methods: (owner class, method, axis), each keyword-only and default-free.
REQUIRED_METHODS = [
    ("mimarsinan.mapping.ir.types.IRNode", "gather_inputs", "dtype"),
    ("mimarsinan.mapping.ir.gather_plan.GatherPlan", "gather", "dtype"),
]


def _resolve(path):
    module, name = path.rsplit(".", 1)
    import importlib

    return getattr(importlib.import_module(module), name)


@pytest.mark.parametrize("path,axis", REQUIRED)
def test_the_axis_is_declared_never_defaulted(path, axis):
    target = _resolve(path)
    callable_ = target.__init__ if inspect.isclass(target) else target
    parameter = _param(callable_, axis)
    assert parameter.default is inspect.Parameter.empty, (
        f"{path}({axis}=...) has grown a default again — a bypassing caller "
        f"would silently inherit another family's semantics")
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (
        f"{path}({axis}=...) must be keyword-only, so a positional bypass "
        f"cannot supply it by accident")


@pytest.mark.parametrize("owner,method,axis", REQUIRED_METHODS)
def test_the_gather_dtype_is_declared_never_defaulted(owner, method, axis):
    parameter = _param(getattr(_resolve(owner), method), axis)
    assert parameter.default is inspect.Parameter.empty, f"{owner}.{method}({axis}=...) has grown a default again"
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, f"{owner}.{method}({axis}=...) must be keyword-only"


def test_sanafe_runner_refuses_undeclared_semantics():
    """No loose-scalar fallback: semantics arrive as a contract or a behavior
    config (whose fields are all required), or construction fails loud."""
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner

    signature = inspect.signature(SanafeRunner.__init__)
    for gone in ("thresholding_mode", "spiking_mode", "firing_mode"):
        assert gone not in signature.parameters, (
            f"SanafeRunner regrew the loose {gone!r} scalar; semantics come "
            f"from contract= or behavior=")
    with pytest.raises(TypeError, match="DECLARED"):
        SanafeRunner(mapping=object(), simulation_length=4)


def test_the_behavior_config_itself_has_no_semantic_defaults():
    """The SSOT carrier stays default-free, or the hole just moves into it."""
    from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig

    for axis in ("spiking_mode", "firing_mode", "thresholding_mode",
                 "spike_generation_mode"):
        parameter = _param(NeuralBehaviorConfig.__init__, axis)
        assert parameter.default is inspect.Parameter.empty, axis
