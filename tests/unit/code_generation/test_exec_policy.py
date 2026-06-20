"""Unit tests for nevresim execution-policy codegen dispatch."""

import pytest

from mimarsinan.code_generation.generate_main import (
    ExecPolicySpec,
    _build_runtime_exec_decl,
    resolve_compare_policy,
    resolve_exec_policy,
    resolve_lif_fire_policy,
)
from mimarsinan.chip_simulation.spiking_mode_policy import (
    NevresimExecParams,
    policy_for_spiking_mode,
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


# ── policy-driven dispatch lock (V2: the firing × sync → codegen SSOT) ─────────
# Golden strings the legacy `resolve_exec_policy` produced before the dispatch was
# moved onto `SpikingModePolicy.nevresim_exec_policy`. Byte-identity guard: a new
# spiking mode must update the policy, not this dispatch, and must not perturb
# these strings.

_LEGACY_CASES = [
    (
        dict(spiking_mode="lif", firing_mode="Novena", thresholding_mode="<=",
             spike_gen_mode="Uniform", weight_type="int",
             simulation_length=10, latency=2, output_count=3),
        "SpikingCompute<LIFirePolicy<ZeroReset, InclusiveCompare>>",
        "using exec = SpikingExecution<10, 2, 3, UniformSpikeGenerator, int, "
        "LIFirePolicy<ZeroReset, InclusiveCompare>>;",
    ),
    (
        dict(spiking_mode="rate", firing_mode="Default", thresholding_mode="<",
             spike_gen_mode="Deterministic", weight_type="double",
             simulation_length=8, latency=1, output_count=4),
        "SpikingCompute<LIFirePolicy<SubtractiveReset, StrictCompare>>",
        "using exec = SpikingExecution<8, 1, 4, DeterministicSpikeGenerator, "
        "double, LIFirePolicy<SubtractiveReset, StrictCompare>>;",
    ),
    (
        dict(spiking_mode="ttfs", firing_mode="TTFS", thresholding_mode="<=",
             spike_gen_mode="TTFS", weight_type="double",
             simulation_length=8, latency=0, output_count=2),
        "TTFSAnalyticalCompute",
        "using exec = TTFSContinuousExecution;",
    ),
    (
        dict(spiking_mode="ttfs_quantized", firing_mode="TTFS", thresholding_mode="<=",
             spike_gen_mode="TTFS", weight_type="double",
             simulation_length=6, latency=1, output_count=2),
        "TTFSQuantizedCompute<6, InclusiveCompare>",
        "using exec = TTFSExecution<6, 1, InclusiveCompare>;",
    ),
    (
        dict(spiking_mode="ttfs_cycle_based", firing_mode="TTFS", thresholding_mode="<=",
             spike_gen_mode="TTFS", weight_type="int",
             simulation_length=6, latency=1, output_count=2),
        "TTFSCascadeCompute<InclusiveCompare>",
        "using exec = TTFSCascadeExecution<6, 1, 2, TTFSSpikeGenerator, int, "
        "InclusiveCompare>;",
    ),
]


@pytest.mark.parametrize(
    "kwargs,compute,exec_decl", _LEGACY_CASES,
    ids=[c[0]["spiking_mode"] for c in _LEGACY_CASES],
)
def test_policy_driven_exec_policy_equals_legacy_string(
    kwargs, compute, exec_decl
) -> None:
    """The migrated, policy-driven dispatch reproduces every legacy string."""
    spec = resolve_exec_policy(**kwargs)
    assert spec.compute_policy == compute
    assert spec.exec_decl == exec_decl


@pytest.mark.parametrize(
    "kwargs,compute,exec_decl", _LEGACY_CASES,
    ids=[c[0]["spiking_mode"] for c in _LEGACY_CASES],
)
def test_resolve_exec_policy_delegates_to_policy(kwargs, compute, exec_decl) -> None:
    """`resolve_exec_policy` == calling the resolved policy's method directly:
    the firing × sync → codegen choice lives in ONE place (the policy)."""
    params = NevresimExecParams(
        compare=resolve_compare_policy(kwargs["thresholding_mode"]),
        lif_fire_policy=resolve_lif_fire_policy(
            kwargs["firing_mode"], kwargs["thresholding_mode"]
        ),
        spike_gen_mode=kwargs["spike_gen_mode"],
        weight_type=kwargs["weight_type"],
        simulation_length=kwargs["simulation_length"],
        latency=kwargs["latency"],
        output_count=kwargs["output_count"],
    )
    direct = policy_for_spiking_mode(kwargs["spiking_mode"]).nevresim_exec_policy(params)
    assert direct == resolve_exec_policy(**kwargs)
    assert direct == ExecPolicySpec(compute_policy=compute, exec_decl=exec_decl)


def test_synchronized_ttfs_cycle_policy_rejects_nevresim() -> None:
    """nevresim never runs the synchronized schedule; the sync policy says so."""
    params = NevresimExecParams(
        compare="InclusiveCompare", lif_fire_policy="LIFirePolicy<SubtractiveReset, "
        "InclusiveCompare>", spike_gen_mode="TTFS", weight_type="int",
        simulation_length=6, latency=1, output_count=2,
    )
    sync_policy = policy_for_spiking_mode("ttfs_cycle_based", "synchronized")
    with pytest.raises(ValueError, match="synchronized"):
        sync_policy.nevresim_exec_policy(params)


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
