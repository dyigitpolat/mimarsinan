"""Integration parity harness for hybrid SCM/HCM and optional backends."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
from mimarsinan.chip_simulation.recording.compare import Diff, compare_records, format_first_diff
from mimarsinan.chip_simulation.recording.records import CoreSpikeCounts, RunRecord, SegmentSpikeRecord
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
    SegmentIOSlice,
)
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow


from mimarsinan.chip_simulation.nevresim.connectivity import (
    ConnectivityMode as NevresimConnectivityMode,
    default_nevresim_connectivity_mode,
)

BackendName = Literal["loihi", "nevresim", "sanafe"]

SUPPORTED_FIRING_MODES: tuple[str, ...] = ("Default", "Novena")
SUPPORTED_THRESHOLDING_MODES: tuple[str, ...] = ("<", "<=")
SUPPORTED_SPIKE_GENERATION_MODES: tuple[str, ...] = (
    "Uniform",
    "FrontLoaded",
    "Deterministic",
    "SpikeTrain",
)

_REBUILD_BACKENDS: frozenset[BackendName] = frozenset({"nevresim", "sanafe"})

_PLUGIN_BUILD_DIR = "build/mimarsinan_sanafe_plugins"
_PLUGIN_SOURCE_DIR = Path("src/mimarsinan/chip_simulation/sanafe/plugins")
_REQUIRED_PLUGIN_LIBS = (
    "libmimarsinan_dendrite.so",
    "libmimarsinan_soma.so",
)
_PLUGIN_CPP_SOURCES = (
    "mimarsinan_dendrite.cpp",
    "mimarsinan_soma.cpp",
    "mimarsinan_ttfs_continuous_soma.cpp",
    "mimarsinan_ttfs_quantized_soma.cpp",
    "mimarsinan_ttfs_cycle_soma.cpp",
    "mimarsinan_ttfs_cascade_soma.cpp",
)


@dataclass
class ParityResult:
    hcm_ran: bool = False
    loihi_ran: bool = False
    sanafe_ran: bool = False
    nevresim_ran: bool = False
    diff_list: List[Diff] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def diffs(self) -> List[str]:
        return [str(d) for d in self.diff_list]

    @property
    def ok(self) -> bool:
        return not self.diff_list

    def first_diff_message(self) -> str:
        return format_first_diff(self.diff_list)


def mimarsinan_root() -> Path:
    return Path(__file__).resolve().parents[2]


def nevresim_connectivity_mode() -> NevresimConnectivityMode:
    """Uses deployment default from ``nevresim.connectivity``."""
    return default_nevresim_connectivity_mode()


def ensure_nevresim_path() -> Path:
    """Configure ``NevresimDriver.nevresim_path``; skip when the tree is absent."""
    root = mimarsinan_root()
    path = root / "nevresim"
    if not path.is_dir():
        pytest.skip(f"nevresim tree missing at {path}")
    from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver

    NevresimDriver.nevresim_path = str(path)
    return path


def have_lava() -> bool:
    try:
        from mimarsinan.chip_simulation.lava_loihi import _subtractive_lif_cls

        _subtractive_lif_cls()
        return True
    except Exception:
        return False


def have_sanafe() -> bool:
    try:
        import sanafe  # noqa: F401

        return True
    except Exception:
        return False


def have_sanafe_plugins() -> bool:
    root = mimarsinan_root()
    return all(
        (root / _PLUGIN_BUILD_DIR / lib).is_file()
        for lib in _REQUIRED_PLUGIN_LIBS
    )


def _plugin_sources_dir() -> Path:
    return mimarsinan_root() / _PLUGIN_SOURCE_DIR


def _plugin_build_dir() -> Path:
    return mimarsinan_root() / _PLUGIN_BUILD_DIR


def sanafe_plugins_stale() -> bool:
    """True when plugin sources are newer than the built ``.so`` artifacts."""
    if not have_sanafe_plugins():
        return True
    build = _plugin_build_dir()
    src_dir = _plugin_sources_dir()
    cmake_lists = src_dir / "CMakeLists.txt"
    source_mtimes = [
        *(path.stat().st_mtime for name in _PLUGIN_CPP_SOURCES if (path := src_dir / name).is_file()),
        cmake_lists.stat().st_mtime,
    ]
    if not source_mtimes:
        return True
    newest_source = max(source_mtimes)
    oldest_plugin = min(
        (build / lib).stat().st_mtime for lib in _REQUIRED_PLUGIN_LIBS
    )
    return newest_source > oldest_plugin


def have_cxx_compiler() -> bool:
    if shutil.which("g++"):
        return True
    for versioned in ("g++-13", "g++-12", "g++-11"):
        if shutil.which(versioned):
            return True
    return False


def build_sanafe_plugins() -> Path:
    """Compile mimarsinan SANA-FE plugins (same output as ``bootstrap_sanafe.sh``)."""
    root = mimarsinan_root()
    plugin_src = _plugin_sources_dir()
    plugin_build = _plugin_build_dir()
    sanafe_src = root / "sana_fe" / "src"
    if not sanafe_src.is_dir():
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive", "sana_fe"],
            cwd=root,
            check=True,
        )
    if not sanafe_src.is_dir():
        raise RuntimeError(
            f"SANA-FE submodule missing at {sanafe_src}; "
            "run scripts/bootstrap_sanafe.sh"
        )
    plugin_build.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(plugin_src),
            "-B",
            str(plugin_build),
            f"-DSANAFE_SRC={sanafe_src}",
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(plugin_build), "--parallel"],
        cwd=root,
        check=True,
    )
    if not have_sanafe_plugins():
        raise RuntimeError(f"plugin build finished but {_PLUGIN_BUILD_DIR} is incomplete")
    return plugin_build


def ensure_sanafe(*, rebuild_if_stale: bool = True) -> None:
    """Install SANA-FE when missing and build plugins when absent or stale."""
    if not have_sanafe():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "sanafe"],
            check=True,
        )
    if not have_sanafe():
        pytest.fail("SANA-FE install failed; run scripts/bootstrap_sanafe.sh")
    if rebuild_if_stale and sanafe_plugins_stale():
        try:
            build_sanafe_plugins()
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            pytest.fail(f"SANA-FE plugin rebuild failed: {exc}")
    if not have_sanafe_plugins():
        pytest.fail(
            "mimarsinan SANA-FE plugins missing after rebuild; "
            "run scripts/bootstrap_sanafe.sh"
        )


def ensure_nevresim_ready() -> Path:
    """Configure nevresim path and verify the C++ toolchain is available."""
    path = ensure_nevresim_path()
    if not have_cxx_compiler():
        pytest.fail(
            "C++ compiler (g++) required for nevresim parity compile step"
        )
    return path


def ensure_backend_ready(backend: BackendName, *, rebuild_if_stale: bool = True) -> None:
    """Prepare a backend for parity: rebuild artifacts when needed."""
    if backend == "loihi":
        if not have_lava():
            pytest.skip("Lava not installed or SubtractiveLIFReset unavailable")
        return
    if backend == "nevresim":
        ensure_nevresim_ready()
        return
    if backend == "sanafe":
        ensure_sanafe(rebuild_if_stale=rebuild_if_stale)
        return
    raise ValueError(f"unknown backend {backend!r}")


def backend_requires_rebuild(backend: BackendName) -> bool:
    return backend in _REBUILD_BACKENDS


def default_behavior(**overrides) -> NeuralBehaviorConfig:
    base = dict(
        spiking_mode="lif",
        firing_mode="Default",
        thresholding_mode="<",
        spike_generation_mode="Uniform",
    )
    base.update(overrides)
    return NeuralBehaviorConfig(**base)


def default_contract(
    *,
    simulation_steps: int = 4,
    ttfs_cycle_schedule: str = "cascaded",
    encoding_layer_placement: str = "subsume",
    bias_mode: str = "on_chip",
    **behavior_overrides,
):
    from mimarsinan.chip_simulation.deployment_contract import (
        SpikingDeploymentContract,
    )

    return SpikingDeploymentContract(
        behavior=default_behavior(**behavior_overrides),
        simulation_steps=simulation_steps,
        ttfs_cycle_schedule=ttfs_cycle_schedule,
        encoding_layer_placement=encoding_layer_placement,
        bias_mode=bias_mode,
    )


def build_toy_hybrid_mapping(*, input_rate: float = 1.0) -> HybridHardCoreMapping:
    """Single-core toy mapping: one input axon -> one output neuron."""
    core = HardCore(4, 4, has_bias_capability=False)
    core.core_matrix = np.zeros((4, 4), dtype=np.float32)
    core.core_matrix[0, 0] = input_rate
    core.axon_sources = [SpikeSource(-2, 0, is_input=True)]
    core.available_axons = 3
    core.available_neurons = 3
    core.threshold = 1.0
    core.latency = 0

    segment = HardCoreMapping([])
    segment.cores = [core]
    segment.output_sources = np.asarray([SpikeSource(0, 0)], dtype=object)

    stage = HybridStage(
        kind="neural",
        name="toy_segment",
        hard_core_mapping=segment,
        input_map=[SegmentIOSlice(node_id=-2, offset=0, size=1)],
        output_map=[SegmentIOSlice(node_id=0, offset=0, size=1)],
    )
    return HybridHardCoreMapping(
        stages=[stage],
        output_sources=np.asarray([IRSource(node_id=0, index=0)], dtype=object),
    )


def build_toy_hcm(behavior: NeuralBehaviorConfig, T: int) -> SpikingHybridCoreFlow:
    hybrid = build_toy_hybrid_mapping()
    return SpikingHybridCoreFlow(
        input_shape=(1,),
        hybrid_mapping=hybrid,
        simulation_length=T,
        preprocessor=nn.Identity(),
        firing_mode=behavior.firing_mode,
        spike_mode=behavior.spike_generation_mode,
        thresholding_mode=behavior.thresholding_mode,
        spiking_mode=behavior.spiking_mode,
    ).eval()


def record_toy_hcm(behavior: NeuralBehaviorConfig, T: int, sample_value: float = 1.0):
    hcm = build_toy_hcm(behavior, T)
    sample = torch.tensor([[sample_value]], dtype=torch.float32)
    with torch.no_grad():
        _, ref = hcm.forward_with_recording(sample, sample_index=0)
    return ref


def _build_nevresim_record_from_ref(
    ref: RunRecord,
    behavior: NeuralBehaviorConfig,
    raw: np.ndarray,
) -> RunRecord:
    """Project nevresim segment output into the HCM ``RunRecord`` shape."""
    out = RunRecord(sample_index=ref.sample_index, T=ref.T)
    for stage_index, ref_seg in ref.segments.items():
        ref_core = ref_seg.cores[0]
        if behavior.spike_generation_mode == "TTFS":
            # nevresim doesn't report input counts; the latched TTFS input is fed
            # identically to both backends (and HCM accumulates it over the full
            # S+latency window), so mirror the HCM-recorded counts and let the
            # output comparison carry the cross-backend parity.
            seg_input_spike_count = ref_seg.seg_input_spike_count.astype(np.int64)
            core_input = ref_core.input_spike_count.astype(np.int64).copy()
        else:
            encoded = behavior.encode_segment_input(ref_seg.seg_input_rates, ref.T)
            seg_input_spike_count = encoded[0].sum(axis=1).astype(np.int64)
            core_input = seg_input_spike_count[: ref_core.n_in_used].copy()
        output_spikes = np.rint(raw[0]).astype(np.int64)[: ref_core.n_out_used]

        out.segments[stage_index] = SegmentSpikeRecord(
            stage_index=ref_seg.stage_index,
            stage_name=ref_seg.stage_name,
            schedule_segment_index=ref_seg.schedule_segment_index,
            schedule_pass_index=ref_seg.schedule_pass_index,
            seg_input_rates=ref_seg.seg_input_rates.copy(),
            seg_input_spike_count=seg_input_spike_count,
            seg_output_spike_count=output_spikes.copy(),
            cores=[
                CoreSpikeCounts(
                    core_index=ref_core.core_index,
                    n_in_used=ref_core.n_in_used,
                    n_out_used=ref_core.n_out_used,
                    core_latency=ref_core.core_latency,
                    has_hardware_bias=ref_core.has_hardware_bias,
                    n_always_on_axons=ref_core.n_always_on_axons,
                    input_spike_count=core_input,
                    output_spike_count=output_spikes.copy(),
                )
            ],
        )
    return out


def run_loihi_parity(behavior: NeuralBehaviorConfig, T: int = 4) -> ParityResult:
    result = ParityResult()
    pytest.importorskip("lava")
    from mimarsinan.chip_simulation.lava_loihi import LavaLoihiRunner

    hybrid = build_toy_hybrid_mapping()
    ref = record_toy_hcm(behavior, T)
    runner = LavaLoihiRunner(
        mapping=hybrid,
        simulation_length=T,
        behavior=behavior,
    )
    actual = runner.run_segments_from_reference(ref)
    result.hcm_ran = True
    result.loihi_ran = True
    result.diff_list = compare_records(ref, actual)
    return result


def _build_nevresim_input_loader(
    behavior: NeuralBehaviorConfig,
    rates: np.ndarray,
    T: int,
) -> list:
    if behavior.nevresim_uses_spike_train_input():
        from mimarsinan.chip_simulation.recording._spike_encoding import (
            flatten_spike_train_sample,
        )

        encoded = behavior.encode_segment_input(rates, T)
        flat = flatten_spike_train_sample(encoded[0])
        return [(flat, np.array([0.0], dtype=np.float64))]
    return [(rates[0], np.array([0.0], dtype=np.float32))]


def run_nevresim_parity(
    behavior: NeuralBehaviorConfig,
    T: int = 4,
    *,
    connectivity_mode: NevresimConnectivityMode | None = None,
    sample_value: float = 1.0,
) -> ParityResult:
    result = ParityResult()
    ensure_nevresim_ready()
    from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
    from mimarsinan.mapping.latency.chip import ChipLatency

    if connectivity_mode is None:
        connectivity_mode = nevresim_connectivity_mode()

    hybrid = build_toy_hybrid_mapping()
    ref = record_toy_hcm(behavior, T, sample_value=sample_value)
    seg = hybrid.stages[0].hard_core_mapping
    assert seg is not None
    rates = ref.segments[0].seg_input_rates
    input_data = _build_nevresim_input_loader(behavior, rates, T)
    latency = ChipLatency(seg).calculate()

    with tempfile.TemporaryDirectory() as tmp:
        driver = NevresimDriver(
            rates.shape[1],
            seg,
            tmp,
            float,
            spike_generation_mode=behavior.spike_generation_mode,
            firing_mode=behavior.firing_mode,
            thresholding_mode=behavior.thresholding_mode,
            spiking_mode=behavior.spiking_mode,
            connectivity_mode=connectivity_mode,
            verbose=False,
        )
        raw = driver.predict_spiking_raw(input_data, T, latency)

    actual = _build_nevresim_record_from_ref(ref, behavior, raw)
    result.hcm_ran = True
    result.nevresim_ran = True
    result.diff_list = compare_records(ref, actual)
    return result


def run_sanafe_parity(
    behavior: NeuralBehaviorConfig,
    T: int = 4,
    *,
    sample_value: float = 1.0,
) -> ParityResult:
    result = ParityResult()
    ensure_sanafe()
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner

    hybrid = build_toy_hybrid_mapping()
    ref = record_toy_hcm(behavior, T, sample_value=sample_value)
    runner = SanafeRunner(
        mapping=hybrid,
        simulation_length=T,
        behavior=behavior,
    )
    sample = np.asarray([[sample_value]], dtype=np.float32)
    sanafe_rec = runner.run(sample, sample_index=0)
    actual = sanafe_rec.to_hcm_subset()
    result.hcm_ran = True
    result.sanafe_ran = True
    result.diff_list = compare_records(ref, actual)
    return result


def run_backend_parity(
    backend: BackendName,
    behavior: NeuralBehaviorConfig,
    T: int = 4,
) -> ParityResult:
    if backend == "loihi":
        return run_loihi_parity(behavior, T)
    if backend == "nevresim":
        return run_nevresim_parity(behavior, T)
    if backend == "sanafe":
        return run_sanafe_parity(behavior, T)
    raise ValueError(f"unknown backend {backend!r}")


def run_mini_hybrid_parity(
    *,
    enable_loihi: bool = False,
    enable_sanafe: bool = False,
) -> ParityResult:
    """Exercise a tiny IR hybrid mapping through HCM; optional backends skipped if unavailable."""
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.mappers.structural import InputMapper
    from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
    from mimarsinan.mapping.model_representation import ModelRepresentation
    from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

    result = ParityResult()
    behavior = default_behavior()

    inp = InputMapper((16,))
    p = Perceptron(8, 16, normalization=nn.Identity(), base_activation_name="ReLU")
    repr_ = ModelRepresentation(PerceptronMapper(inp, p))
    ir = IRMapping(
        q_max=127.0,
        firing_mode=behavior.firing_mode,
        max_axons=64,
        max_neurons=64,
    ).map(repr_)
    cores_config = [{"max_axons": 64, "max_neurons": 64, "count": 8}]
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=cores_config,
    )
    flow = SpikingHybridCoreFlow(
        (16,),
        hybrid,
        simulation_length=4,
        preprocessor=torch.nn.Identity(),
        firing_mode=behavior.firing_mode,
        spike_mode=behavior.spike_generation_mode,
        thresholding_mode=behavior.thresholding_mode,
        spiking_mode=behavior.spiking_mode,
    )
    x = torch.randn(1, 16)
    with torch.no_grad():
        _ = flow(x)
    result.hcm_ran = True

    if enable_loihi:
        pytest.importorskip("lava")
        result.loihi_ran = True
        result.notes.append("loihi: import ok (full replay not run in harness)")

    if enable_sanafe:
        result.sanafe_ran = True
        result.notes.append("sanafe: skipped unless sanafe binary present")

    return result

