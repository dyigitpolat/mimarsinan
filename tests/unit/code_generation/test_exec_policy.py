"""Unit tests for nevresim execution-policy codegen dispatch."""

import pytest

from mimarsinan.code_generation.generate_main import (
    ExecPolicySpec,
    _build_runtime_exec_decl,
    resolve_exec_policy,
)


@pytest.mark.parametrize(
    "spiking_mode",
    ["lif", "rate", "ttfs", "ttfs_quantized", "ttfs_cycle_based"],
)
def test_resolve_exec_policy_all_modes(spiking_mode: str) -> None:
    spec = resolve_exec_policy(
        spiking_mode=spiking_mode,
        firing_mode="Default",
        thresholding_mode="<=",
        spike_gen_mode="Deterministic",
        weight_type="double",
        simulation_length=8,
        latency=1,
        output_count=4,
    )
    assert isinstance(spec, ExecPolicySpec)
    assert spec.compute_policy
    assert spec.exec_decl.startswith("using exec =")


def test_resolve_exec_policy_lif() -> None:
    spec = resolve_exec_policy(
        spiking_mode="lif",
        firing_mode="Novena",
        thresholding_mode="<=",
        spike_gen_mode="Uniform",
        weight_type="int",
        simulation_length=10,
        latency=2,
        output_count=3,
    )
    assert spec.compute_policy == "SpikingCompute<LIFirePolicy<ZeroReset, InclusiveCompare>>"
    assert "SpikingExecution<10, 2, 3, UniformSpikeGenerator, int," in spec.exec_decl


def test_resolve_exec_policy_ttfs() -> None:
    spec = resolve_exec_policy(
        spiking_mode="ttfs",
        firing_mode="TTFS",
        thresholding_mode="<=",
        spike_gen_mode="TTFS",
        weight_type="double",
        simulation_length=8,
        latency=0,
        output_count=2,
    )
    assert spec.compute_policy == "TTFSAnalyticalCompute"
    assert spec.exec_decl == "using exec = TTFSContinuousExecution;"


def test_resolve_exec_policy_ttfs_quantized() -> None:
    spec = resolve_exec_policy(
        spiking_mode="ttfs_quantized",
        firing_mode="TTFS",
        thresholding_mode="<=",
        spike_gen_mode="TTFS",
        weight_type="double",
        simulation_length=6,
        latency=1,
        output_count=2,
    )
    assert spec.compute_policy == "TTFSQuantizedCompute<6, InclusiveCompare>"
    assert spec.exec_decl == "using exec = TTFSExecution<6, 1, InclusiveCompare>;"


def test_resolve_exec_policy_ttfs_quantized_strict_compare() -> None:
    spec = resolve_exec_policy(
        spiking_mode="ttfs_quantized", firing_mode="TTFS", thresholding_mode="<",
        spike_gen_mode="TTFS", weight_type="double",
        simulation_length=6, latency=1, output_count=2,
    )
    assert spec.compute_policy == "TTFSQuantizedCompute<6, StrictCompare>"
    assert spec.exec_decl == "using exec = TTFSExecution<6, 1, StrictCompare>;"


def test_resolve_exec_policy_ttfs_cycle_based_cascaded() -> None:
    """Genuine cascaded single-spike: fire-once over single-spike TTFS inputs
    through the standard SpikingExecution; config-driven compare policy.
    """
    spec = resolve_exec_policy(
        spiking_mode="ttfs_cycle_based",
        firing_mode="TTFS",
        thresholding_mode="<=",
        spike_gen_mode="TTFS",
        weight_type="int",
        simulation_length=6,
        latency=1,
        output_count=2,
    )
    assert spec.compute_policy == "TTFSCascadeCompute<InclusiveCompare>"
    assert spec.exec_decl == (
        "using exec = TTFSCascadeExecution<6, 1, 2, "
        "TTFSSpikeGenerator, int, InclusiveCompare>;"
    )


def test_resolve_exec_policy_ttfs_cycle_based_strict_compare() -> None:
    spec = resolve_exec_policy(
        spiking_mode="ttfs_cycle_based", firing_mode="TTFS", thresholding_mode="<",
        spike_gen_mode="TTFS", weight_type="int",
        simulation_length=6, latency=1, output_count=2,
    )
    assert spec.compute_policy == "TTFSCascadeCompute<StrictCompare>"
    assert spec.exec_decl == (
        "using exec = TTFSCascadeExecution<6, 1, 2, "
        "TTFSSpikeGenerator, int, StrictCompare>;"
    )


@pytest.mark.parametrize(
    "spiking_mode",
    ["lif", "rate", "ttfs", "ttfs_quantized", "ttfs_cycle_based"],
)
def test_build_runtime_exec_decl_all_modes(spiking_mode: str) -> None:
    compute, exec_decl = _build_runtime_exec_decl(
        spiking_mode=spiking_mode,
        firing_mode="Default" if spiking_mode in ("lif", "rate") else "TTFS",
        thresholding_mode="<=",
        spike_gen_mode="Deterministic" if spiking_mode in ("lif", "rate") else "TTFS",
        weight_type="double",
        simulation_length=4,
        latency=1,
        output_count=2,
    )
    assert compute
    assert exec_decl.startswith("using exec =")
