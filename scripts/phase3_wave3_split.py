#!/usr/bin/env python3
"""Phase 3 wave 3: split oversized modules into subpackages (no shims)."""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "mimarsinan"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


def _write(rel: str, text: str) -> None:
    path = SRC / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _lines(rel: str) -> list[str]:
    return _read(rel).splitlines(keepends=True)


def _join(lines: list[str]) -> str:
    return "".join(lines)


def _extract(lines: list[str], start: int, end: int) -> str:
    """1-based inclusive line slice."""
    return _join(lines[start - 1 : end])


def split_unified_hybrid() -> None:
    u_lines = _lines("models/spiking/unified_core_flow.py")
    h_lines = _lines("models/spiking/hybrid_core_flow.py")

    unified_stage_io = '''"""Unified IRGraph stage I/O helpers (weights, spans, contracts)."""

from __future__ import annotations

from typing import Dict

import torch

from mimarsinan.mapping.ir import ComputeOp, IRGraph, NeuralCore
from mimarsinan.mapping.support.ir_source_spans import IRSourceSpan
from mimarsinan.models.spiking.signal_spans import fill_signal_from_spans
from mimarsinan.models.spiking.spiking_config import TTFS_SPIKING_MODES


class UnifiedStageIOMixin:
    """Weight/threshold accessors and IR span fill helpers."""

    _TTFS_SPIKING_MODES = TTFS_SPIKING_MODES

'''
    unified_stage_io += _extract(u_lines, 169, 266)
    unified_stage_io += "\n"
    unified_stage_io += _extract(u_lines, 389, 410)
    unified_stage_io += "\n"
    unified_stage_io += _extract(u_lines, 528, 608)

    unified_lif = '''"""Unified IRGraph LIF forward step."""

from __future__ import annotations

from typing import Dict

import torch

from mimarsinan.mapping.ir import ComputeOp, NeuralCore
from mimarsinan.models.spiking.lif_core_step import lif_core_contribute_and_fire


class UnifiedLifStepMixin:
    """Rate-coded LIF simulation over flat IRGraph nodes."""

'''
    unified_lif += _extract(u_lines, 278, 387)

    unified_ttfs = '''"""Unified IRGraph TTFS forward steps."""

from __future__ import annotations

from typing import Dict

import torch

from mimarsinan.mapping.ir import ComputeOp, NeuralCore
from mimarsinan.models.spiking.ttfs_activation import ttfs_activation_from_type
from mimarsinan.models.spiking.ttfs_kernels import ttfs_quantized_activation


class UnifiedTtfsStepMixin:
    """Analytical TTFS paths (continuous and quantized)."""

'''
    unified_ttfs += _extract(u_lines, 412, 526)

    unified_flow = '''"""Spiking simulator for unified IRGraph (NeuralCore + ComputeOp); LIF and TTFS modes."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from mimarsinan.chip_simulation import spike_modes
from mimarsinan.mapping.ir import ComputeOp, IRGraph, NeuralCore
from mimarsinan.mapping.support.ir_source_spans import IRSourceSpan, compress_ir_sources
from mimarsinan.models.spiking.spiking_config import (
    COMPUTE_DTYPE,
    validate_spiking_init,
)
from mimarsinan.models.spiking.ttfs_activation import ttfs_activation_from_type
from mimarsinan.models.spiking.unified.lif_step import UnifiedLifStepMixin
from mimarsinan.models.spiking.unified.stage_io import UnifiedStageIOMixin
from mimarsinan.models.spiking.unified.ttfs_step import UnifiedTtfsStepMixin

# Backward compatibility for tests and external callers.
_ttfs_activation_from_type = ttfs_activation_from_type


class SpikingUnifiedCoreFlow(
    UnifiedStageIOMixin,
    UnifiedLifStepMixin,
    UnifiedTtfsStepMixin,
    nn.Module,
):
    """Flat IRGraph spiking sim: LIF/TTFS cores, ComputeOp sync barriers, shared WeightBank params."""

'''
    unified_flow += _extract(u_lines, 33, 167)
    unified_flow += "\n"
    unified_flow += _extract(u_lines, 161, 167)
    unified_flow += "\n"
    unified_flow += _extract(u_lines, 268, 276)
    unified_flow += "\n"
    unified_flow += _extract(u_lines, 610, 630)

    _write("models/spiking/unified/__init__.py", '''"""Unified IRGraph spiking flow."""
from mimarsinan.models.spiking.unified.flow import (
    SpikingUnifiedCoreFlow,
    _ttfs_activation_from_type,
)

__all__ = ["SpikingUnifiedCoreFlow", "_ttfs_activation_from_type"]
''')
    _write("models/spiking/unified/stage_io.py", unified_stage_io)
    _write("models/spiking/unified/lif_step.py", unified_lif)
    _write("models/spiking/unified/ttfs_step.py", unified_ttfs)
    _write("models/spiking/unified/flow.py", unified_flow)

    hybrid_stage_io = '''"""Hybrid mapping segment I/O and tensor cache."""

from __future__ import annotations

from typing import Dict

import torch

from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
    assemble_segment_input_torch,
    gather_final_output_torch,
    store_segment_output_torch,
)
from mimarsinan.mapping.ir import ComputeOp, IRSource
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridStage, SegmentIOSlice
from mimarsinan.mapping.support.core_geometry import used_axons, used_neurons
from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources
from mimarsinan.models.spiking.signal_spans import fill_signal_from_spans
from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE


class HybridStageIOMixin:
    """Segment tensor cache and state-buffer I/O."""

'''
    hybrid_stage_io += _extract(h_lines, 109, 347)

    hybrid_lif = '''"""Hybrid mapping rate-coded LIF segment execution."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from mimarsinan.chip_simulation.recording.spike_recorder import CoreSpikeCounts, SegmentSpikeRecord
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridStage
from mimarsinan.models.spiking.lif_core_step import lif_core_contribute_and_fire
from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE


class HybridLifStepMixin:
    """Rate-coded neural segments and LIF hybrid forward."""

'''
    hybrid_lif += _extract(h_lines, 349, 493)
    hybrid_lif += "\n"
    hybrid_lif += _extract(h_lines, 629, 651)
    hybrid_lif += "\n"
    hybrid_lif += _extract(h_lines, 653, 784)

    hybrid_ttfs = '''"""Hybrid mapping TTFS segment and forward execution."""

from __future__ import annotations

from typing import Dict

import torch

from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridStage
from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE


class HybridTtfsStepMixin:
    """TTFS neural segments and state-buffer TTFS forward."""

'''
    hybrid_ttfs += _extract(h_lines, 495, 515)
    hybrid_ttfs += "\n"
    hybrid_ttfs += _extract(h_lines, 531, 627)

    hybrid_flow = '''"""Spiking simulation for HybridHardCoreMapping (rate-coded and TTFS)."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.chip_simulation import spike_modes
from mimarsinan.chip_simulation.recording.spike_recorder import RunRecord
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping
from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE, validate_spiking_init
from mimarsinan.models.spiking.hybrid.lif_step import HybridLifStepMixin
from mimarsinan.models.spiking.hybrid.stage_io import HybridStageIOMixin
from mimarsinan.models.spiking.hybrid.ttfs_step import HybridTtfsStepMixin

# Backward compatibility for integration tests.
_COMPUTE_DTYPE = COMPUTE_DTYPE


class SpikingHybridCoreFlow(
    HybridStageIOMixin,
    HybridLifStepMixin,
    HybridTtfsStepMixin,
    nn.Module,
):
    """
    Execute a HybridHardCoreMapping via a global state buffer keyed by IR node_id.
    Neural segments use SegmentIOSlice I/O; ComputeOps gather from the buffer.
    Supports rate-coded (LIF) and TTFS (continuous or quantized analytical) modes.
    """

    _TTFS_FIRING_MODES = HybridTtfsStepMixin._TTFS_FIRING_MODES
    _TTFS_SPIKING_MODES = HybridTtfsStepMixin._TTFS_SPIKING_MODES

'''
    hybrid_flow += _extract(h_lines, 58, 174)
    hybrid_flow += "\n"
    hybrid_flow += _extract(h_lines, 517, 529)
    hybrid_flow += "\n"
    hybrid_flow += _extract(h_lines, 786, 802)

    _write("models/spiking/hybrid/__init__.py", '''"""Hybrid hard-core spiking flow."""
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow, _COMPUTE_DTYPE

__all__ = ["SpikingHybridCoreFlow", "_COMPUTE_DTYPE"]
''')
    _write("models/spiking/hybrid/stage_io.py", hybrid_stage_io)
    _write("models/spiking/hybrid/lif_step.py", hybrid_lif)
    _write("models/spiking/hybrid/ttfs_step.py", hybrid_ttfs)
    _write("models/spiking/hybrid/flow.py", hybrid_flow)

    (SRC / "models/spiking/unified_core_flow.py").unlink(missing_ok=True)
    (SRC / "models/spiking/hybrid_core_flow.py").unlink(missing_ok=True)


def split_nn() -> None:
    d_lines = _lines("models/nn/decorators.py")
    a_lines = _lines("models/nn/activations.py")

    _write(
        "models/nn/decorators/transforms.py",
        _extract(d_lines, 1, 165) + "\n",
    )
    _write(
        "models/nn/decorators/clamp_quantize.py",
        _extract(d_lines, 167, 224) + "\n",
    )
    _write(
        "models/nn/decorators/adjustment.py",
        _extract(d_lines, 227, 327) + "\n",
    )
    _write(
        "models/nn/decorators/__init__.py",
        '''"""Composable decorators and adjustment strategies for activations."""
from mimarsinan.models.nn.decorators.adjustment import (
    ActivationReplacementDecorator,
    DecoratedActivation,
    MixAdjustmentStrategy,
    NestedAdjustmentStrategy,
    NestedDecoration,
    RandomMaskAdjustmentStrategy,
    RateAdjustedDecorator,
)
from mimarsinan.models.nn.decorators.clamp_quantize import ClampDecorator, QuantizeDecorator
from mimarsinan.models.nn.decorators.transforms import (
    NoisyDropout,
    SavedTensorDecorator,
    ScaleDecorator,
    ShiftDecorator,
    StatsDecorator,
)

__all__ = [
    "ActivationReplacementDecorator",
    "ClampDecorator",
    "DecoratedActivation",
    "MixAdjustmentStrategy",
    "NestedAdjustmentStrategy",
    "NestedDecoration",
    "NoisyDropout",
    "QuantizeDecorator",
    "RandomMaskAdjustmentStrategy",
    "RateAdjustedDecorator",
    "SavedTensorDecorator",
    "ScaleDecorator",
    "ShiftDecorator",
    "StatsDecorator",
]
''',
    )

    _write(
        "models/nn/activations/autograd.py",
        _extract(a_lines, 1, 148) + "\n",
    )
    _write(
        "models/nn/activations/lif.py",
        _extract(a_lines, 150, 312) + "\n",
    )
    _write(
        "models/nn/activations/__init__.py",
        '''"""Custom autograd activations and clamp."""
from mimarsinan.models.nn.activations.autograd import (
    ChipInputQuantizer,
    DifferentiableClamp,
    LeakyGradReLU,
    LeakyGradReLUFunction,
    RoundedStaircaseFunction,
    StaircaseFunction,
    StrictATanSurrogate,
)
from mimarsinan.models.nn.activations.lif import LIFActivation, run_cycle_accurate, uniform_encode_to_spike_train

__all__ = [
    "ChipInputQuantizer",
    "DifferentiableClamp",
    "LIFActivation",
    "LeakyGradReLU",
    "LeakyGradReLUFunction",
    "RoundedStaircaseFunction",
    "StaircaseFunction",
    "StrictATanSurrogate",
    "run_cycle_accurate",
    "uniform_encode_to_spike_train",
]
''',
    )

    layers = _read("models/nn/layers.py")
    layers = layers.replace(
        "from mimarsinan.models.nn.activations import",
        "from mimarsinan.models.nn.activations import",
    )
    layers = layers.replace(
        "from mimarsinan.models.nn.decorators import",
        "from mimarsinan.models.nn.decorators import",
    )
    _write("models/nn/layers.py", layers)

    (SRC / "models/nn/decorators.py").unlink(missing_ok=True)
    (SRC / "models/nn/activations.py").unlink(missing_ok=True)


def split_runner_analysis() -> None:
    lines = _lines("chip_simulation/sanafe/runner_analysis.py")
    header = _extract(lines, 1, 18)
    trace = header + _extract(lines, 20, 419)
    energy = header + _extract(lines, 78, 148) + _extract(lines, 527, 597)
    noc = header + _extract(lines, 300, 318) + _extract(lines, 437, 525) + _extract(
        lines, 712, 733
    )
    connectivity = header + _extract(lines, 600, 617) + _extract(lines, 736, 773)

    _write("chip_simulation/sanafe/analysis/trace.py", trace)
    _write("chip_simulation/sanafe/analysis/energy.py", energy)
    _write("chip_simulation/sanafe/analysis/noc.py", noc)
    _write("chip_simulation/sanafe/analysis/connectivity.py", connectivity)
    _write(
        "chip_simulation/sanafe/analysis/__init__.py",
        '''"""SANA-FE trace, energy, and connectivity analysis helpers."""
from mimarsinan.chip_simulation.sanafe.analysis.connectivity import *  # noqa: F403
from mimarsinan.chip_simulation.sanafe.analysis.energy import *  # noqa: F403
from mimarsinan.chip_simulation.sanafe.analysis.noc import *  # noqa: F403
from mimarsinan.chip_simulation.sanafe.analysis.trace import *  # noqa: F403
''',
    )
    (SRC / "chip_simulation/sanafe/runner_analysis.py").unlink(missing_ok=True)


def split_sanafe_runner() -> None:
    lines = _lines("chip_simulation/sanafe/runner.py")
    header = _extract(lines, 1, 57)
    core = header + _extract(lines, 59, 232)
    neural = '''"""SANA-FE neural stage execution."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mimarsinan.chip_simulation.hybrid_run.hybrid_semantics import (
    NeuralSegmentResult,
    is_ttfs_spiking_mode,
    lif_inter_stage_from_spike_counts,
    store_neural_segment_output,
)
from mimarsinan.chip_simulation.recording._spike_encoding import uniform_rate_encode
from mimarsinan.chip_simulation.sanafe.analysis import (
    _build_spike_capture_warning,
    _compute_connectivity_edges,
    _compute_critical_cores,
    _compute_cycle_energy_breakdown,
    _compute_noc_traffic_per_cycle,
    _compute_ttfs_activity_diagnostics,
    _count_cross_tile_connectivity_edges,
    _flatten_message_trace,
    _group_name,
    _group_name_to_size,
    _group_row_offsets,
    _lif_and_input_spike_totals,
    _input_spikes_per_core,
    _pack_potential_trace,
    _pack_spike_trace_matrix,
    _per_core_energy_sanafe,
    _per_core_packet_counts,
    _read_ttfs_core_activations,
    _spike_trace_to_group_counts,
    _summarize_message_trace,
    _aggregate_noc_link_load,
    _aggregate_noc_links,
    _compute_cascade_timeline,
)
from mimarsinan.chip_simulation.sanafe.net_synth import (
    apply_ttfs_preset_membranes,
    build_network_for_segment,
    set_always_on_spike_trains,
    set_input_spike_trains,
    set_ttfs_input_spike_trains,
)
from mimarsinan.chip_simulation.sanafe.records import (
    SanafeCoreRecord,
    SanafeEnergyBreakdown,
    SanafeRunRecord,
    SanafeSegmentRecord,
    SanafeTileRecord,
)
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.support.core_geometry import used_axons as _used_axons
from mimarsinan.mapping.support.core_geometry import used_neurons as _used_neurons

from .core import _COMPUTE_DTYPE


class SanafeNeuralStageMixin:
    """Run one neural hybrid stage through SANA-FE."""

'''
    neural += _extract(lines, 234, 551)

    segment_io = '''"""SANA-FE per-core input/output spike accounting."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mimarsinan.chip_simulation.sanafe.analysis import _group_name
from mimarsinan.chip_simulation.sanafe.records import SanafeCoreRecord, SanafeEnergyBreakdown, SanafeTileRecord
from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources

from .core import _COMPUTE_DTYPE


class SanafeSegmentIOMixin:
    """Derive segment spike counts and tile aggregates."""

'''
    segment_io += _extract(lines, 553, 794)

    core_class = core.replace(
        "class SanafeRunner:",
        "class SanafeRunner(SanafeNeuralStageMixin, SanafeSegmentIOMixin):",
    )
    core_class = core_class.replace(
        "from .arch_synth import _sanafe, build_architecture, derive_arch_spec\n",
        "from mimarsinan.chip_simulation.sanafe.analysis import _group_name\n"
        "from .arch_synth import _sanafe, build_architecture, derive_arch_spec\n",
    )

    _write("chip_simulation/sanafe/runner/core.py", core_class)
    _write("chip_simulation/sanafe/runner/neural_stage.py", neural)
    _write("chip_simulation/sanafe/runner/segment_io.py", segment_io)
    _write(
        "chip_simulation/sanafe/runner/__init__.py",
        '''"""SANA-FE backend driver."""
from mimarsinan.chip_simulation.sanafe.runner.core import SanafeRunner, _COMPUTE_DTYPE, _RAW_INPUT_NODE_ID

__all__ = ["SanafeRunner", "_COMPUTE_DTYPE", "_RAW_INPUT_NODE_ID"]
''',
    )
    (SRC / "chip_simulation/sanafe/runner.py").unlink(missing_ok=True)


def split_records() -> None:
    lines = _lines("chip_simulation/sanafe/records.py")
    _write("chip_simulation/sanafe/records/energy.py", _extract(lines, 1, 80))
    _write(
        "chip_simulation/sanafe/records/hardware.py",
        _extract(lines, 82, 255),
    )
    _write(
        "chip_simulation/sanafe/records/run.py",
        _extract(lines, 1, 30)
        + _extract(lines, 257, 473),
    )
    _write(
        "chip_simulation/sanafe/records/__init__.py",
        '''"""SANA-FE run-record dataclasses."""
from mimarsinan.chip_simulation.sanafe.records.energy import SanafeEnergyBreakdown
from mimarsinan.chip_simulation.sanafe.records.hardware import (
    SanafeArchGeometry,
    SanafeCascadePoint,
    SanafeConnectivityEdge,
    SanafeCoreDiff,
    SanafeCoreRecord,
    SanafeCriticalCore,
    SanafeCycleEnergyPoint,
    SanafeNocLink,
    SanafeNocLinkLoad,
    SanafeTileRecord,
)
from mimarsinan.chip_simulation.sanafe.records.run import SanafeRunRecord, SanafeSegmentRecord

__all__ = [
    "SanafeArchGeometry",
    "SanafeCascadePoint",
    "SanafeConnectivityEdge",
    "SanafeCoreDiff",
    "SanafeCoreRecord",
    "SanafeCriticalCore",
    "SanafeCycleEnergyPoint",
    "SanafeEnergyBreakdown",
    "SanafeNocLink",
    "SanafeNocLinkLoad",
    "SanafeRunRecord",
    "SanafeSegmentRecord",
    "SanafeTileRecord",
]
''',
    )
    run_py = _read("chip_simulation/sanafe/records/run.py")
    run_py = run_py.replace(
        "from mimarsinan.chip_simulation.recording.spike_recorder import (\n",
        "from mimarsinan.chip_simulation.recording.spike_recorder import (\n",
    )
    run_py = (
        _extract(lines, 1, 30)
        + "\nfrom mimarsinan.chip_simulation.sanafe.records.energy import SanafeEnergyBreakdown\n\n"
        + _extract(lines, 257, 473)
    )
    _write("chip_simulation/sanafe/records/run.py", run_py)
    hw = _read("chip_simulation/sanafe/records/hardware.py")
    hw = "from mimarsinan.chip_simulation.sanafe.records.energy import SanafeEnergyBreakdown\n\n" + hw
    _write("chip_simulation/sanafe/records/hardware.py", hw)
    (SRC / "chip_simulation/sanafe/records.py").unlink(missing_ok=True)


def split_net_synth() -> None:
    lines = _lines("chip_simulation/sanafe/net_synth.py")
    _write("chip_simulation/sanafe/net_synth/build.py", _extract(lines, 1, 268))
    _write(
        "chip_simulation/sanafe/net_synth/spike_trains.py",
        _extract(lines, 271, 394),
    )
    _write(
        "chip_simulation/sanafe/net_synth/__init__.py",
        '''"""Build SANA-FE networks from HardCoreMapping segments."""
from mimarsinan.chip_simulation.sanafe.net_synth.build import build_network_for_segment
from mimarsinan.chip_simulation.sanafe.net_synth.spike_trains import (
    apply_ttfs_preset_membranes,
    set_always_on_spike_trains,
    set_input_spike_trains,
    set_ttfs_input_spike_trains,
)

__all__ = [
    "apply_ttfs_preset_membranes",
    "build_network_for_segment",
    "set_always_on_spike_trains",
    "set_input_spike_trains",
    "set_ttfs_input_spike_trains",
]
''',
    )
    spike = _read("chip_simulation/sanafe/net_synth/spike_trains.py")
    spike = (
        "from mimarsinan.chip_simulation.sanafe.neuron_model import (\n"
        "    input_neuron_attributes,\n"
        "    ttfs_continuous_model_attributes,\n"
        "    ttfs_quantized_model_attributes,\n"
        ")\n"
        "from mimarsinan.mapping.support.core_geometry import used_neurons as _used_neurons\n\n"
        + spike
    )
    _write("chip_simulation/sanafe/net_synth/spike_trains.py", spike)
    (SRC / "chip_simulation/sanafe/net_synth.py").unlink(missing_ok=True)


def split_arch_synth() -> None:
    lines = _lines("chip_simulation/sanafe/arch_synth.py")
    _write("chip_simulation/sanafe/arch_synth/spec.py", _extract(lines, 1, 170))
    _write("chip_simulation/sanafe/arch_synth/build.py", _extract(lines, 172, 328))
    build = _read("chip_simulation/sanafe/arch_synth/build.py")
    build = (
        "from mimarsinan.chip_simulation.sanafe.arch_synth.spec import ArchSpec, _sanafe\n\n"
        + build
    )
    _write("chip_simulation/sanafe/arch_synth/build.py", build)
    _write(
        "chip_simulation/sanafe/arch_synth/__init__.py",
        '''"""Synthesize SANA-FE Architecture from hybrid mapping."""
from mimarsinan.chip_simulation.sanafe.arch_synth.build import build_architecture
from mimarsinan.chip_simulation.sanafe.arch_synth.spec import ArchSpec, derive_arch_spec, _sanafe

__all__ = ["ArchSpec", "build_architecture", "derive_arch_spec", "_sanafe"]
''',
    )
    (SRC / "chip_simulation/sanafe/arch_synth.py").unlink(missing_ok=True)


def split_ttfs_segment() -> None:
    lines = _lines("chip_simulation/ttfs/ttfs_segment.py")
    _write("chip_simulation/ttfs/segment_arrays.py", _extract(lines, 1, 107))
    _write(
        "chip_simulation/ttfs/ttfs_segment.py",
        "from mimarsinan.chip_simulation.ttfs.segment_arrays import SegmentTtfsArrays, segment_ttfs_arrays_from_mapping\n\n"
        + _extract(lines, 110, 318),
    )


def split_spike_recorder() -> None:
    lines = _lines("chip_simulation/recording/spike_recorder.py")
    _write("chip_simulation/recording/records.py", _extract(lines, 1, 87))
    _write("chip_simulation/recording/compare.py", _extract(lines, 89, 306))
    compare = _read("chip_simulation/recording/compare.py")
    compare = (
        "from mimarsinan.chip_simulation.recording.records import CoreSpikeCounts, RunRecord, SegmentSpikeRecord\n\n"
        + compare
    )
    _write("chip_simulation/recording/compare.py", compare)
    _write(
        "chip_simulation/recording/spike_recorder.py",
        '''"""Per-segment spike-count records for HCM↔Loihi parity verification."""
from mimarsinan.chip_simulation.recording.compare import Diff, compare_records, format_first_diff
from mimarsinan.chip_simulation.recording.records import CoreSpikeCounts, RunRecord, SegmentSpikeRecord

__all__ = [
    "CoreSpikeCounts",
    "Diff",
    "RunRecord",
    "SegmentSpikeRecord",
    "compare_records",
    "format_first_diff",
]
''',
    )


def split_simulation_runner() -> None:
    lines = _lines("chip_simulation/simulation_runner.py")
    emit = _extract(lines, 1, 88)
    core = _extract(lines, 91, 159) + _extract(lines, 440, 452)
    flat = _extract(lines, 160, 188) + _extract(lines, 190, 198)
    hybrid = _extract(lines, 200, 438)

    _write("chip_simulation/simulation_runner/emit.py", emit)
    _write(
        "chip_simulation/simulation_runner/flat.py",
        "from mimarsinan.mapping.packing.softcore import HardCoreMapping\n\n"
        + flat,
    )
    _write(
        "chip_simulation/simulation_runner/hybrid.py",
        "from __future__ import annotations\n\n"
        + hybrid,
    )
    runner_core = (
        _extract(lines, 1, 11)
        + "from mimarsinan.chip_simulation.simulation_runner.emit import _PreparedSegment, _emit_and_compile_segment\n"
        + "from mimarsinan.chip_simulation.simulation_runner.flat import SimulationFlatMixin\n"
        + "from mimarsinan.chip_simulation.simulation_runner.hybrid import SimulationHybridMixin\n\n"
        + _extract(lines, 91, 139)
        + "\n\nclass SimulationRunner(SimulationFlatMixin, SimulationHybridMixin):\n"
        + _extract(lines, 143, 159)
        + _extract(lines, 440, 452)
    )
    flat_mixin = '''"""Flat nevresim mapping execution."""

from __future__ import annotations

import numpy as np

from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.softcore import HardCoreMapping
from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver


class SimulationFlatMixin:
'''
    flat_mixin += _extract(lines, 160, 198)

    hybrid_mixin = '''"""Hybrid multi-segment nevresim execution."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import torch

from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping, HybridStage, SegmentIOSlice
from mimarsinan.mapping.packing.softcore import HardCoreMapping
from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
    assemble_segment_input_numpy,
    execute_compute_op_numpy,
    gather_final_output_numpy,
    store_segment_output_numpy,
)
from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
from mimarsinan.chip_simulation.simulation_runner.emit import _PreparedSegment, _emit_and_compile_segment


class SimulationHybridMixin:
'''
    hybrid_mixin += _extract(lines, 200, 438)

    _write("chip_simulation/simulation_runner/flat.py", flat_mixin)
    _write("chip_simulation/simulation_runner/hybrid.py", hybrid_mixin)
    _write("chip_simulation/simulation_runner/emit.py", emit)
    _write("chip_simulation/simulation_runner/core.py", runner_core)
    _write(
        "chip_simulation/simulation_runner/__init__.py",
        '''"""Nevresim chip simulation runner."""
from mimarsinan.chip_simulation.simulation_runner.core import SimulationRunner

__all__ = ["SimulationRunner"]
''',
    )
    (SRC / "chip_simulation/simulation_runner.py").unlink(missing_ok=True)


def split_lava_loihi() -> None:
    lines = _lines("chip_simulation/lava_loihi_runner.py")
    helpers = _extract(lines, 1, 151)
    core_lava = _extract(lines, 255, 323)
    segment = _extract(lines, 326, 571)
    runner = (
        _extract(lines, 1, 14)
        + "from mimarsinan.chip_simulation.lava_loihi.core_lava import LavaCoreMixin\n"
        + "from mimarsinan.chip_simulation.lava_loihi.segment_runner import LavaSegmentMixin\n"
        + "from mimarsinan.chip_simulation.lava_loihi.timing import _RunProfile, _SegmentTiming, _StageTrace\n"
        + _extract(lines, 15, 78)
        + "\n\nclass LavaLoihiRunner(LavaCoreMixin, LavaSegmentMixin, SimulationRunner):\n"
        + _extract(lines, 156, 220)
        + _extract(lines, 574, 704)
    )

    _write("chip_simulation/lava_loihi/timing.py", _extract(lines, 80, 151))
    _write(
        "chip_simulation/lava_loihi/core_lava.py",
        helpers.split("class LavaLoihiRunner")[0]
        + "from mimarsinan.chip_simulation.lava_loihi.timing import _LAVA_DTYPE\n\n\n"
        + "class LavaCoreMixin:\n"
        + core_lava,
    )
    _write(
        "chip_simulation/lava_loihi/segment_runner.py",
        "from __future__ import annotations\n\n"
        + "import os\n"
        + "import time as _time\n"
        + "from typing import Dict\n\n"
        + "import numpy as np\n\n"
        + "from mimarsinan.chip_simulation.recording._spike_encoding import uniform_rate_encode as _uniform_rate_encode\n"
        + "from mimarsinan.chip_simulation.recording.spike_recorder import CoreSpikeCounts, SegmentSpikeRecord\n"
        + "from mimarsinan.mapping.packing.softcore import HardCoreMapping\n"
        + "from mimarsinan.chip_simulation.lava_loihi.timing import _LAVA_DTYPE, _SegmentTiming\n\n\n"
        + "class LavaSegmentMixin:\n"
        + segment,
    )
    _write("chip_simulation/lava_loihi/runner.py", runner)
    _write(
        "chip_simulation/lava_loihi/__init__.py",
        '''"""Host-scheduled Lava Loihi simulation."""
from mimarsinan.chip_simulation.lava_loihi.core_lava import _subtractive_lif_cls
from mimarsinan.chip_simulation.lava_loihi.runner import LavaLoihiRunner
from mimarsinan.chip_simulation.recording._spike_encoding import uniform_rate_encode as _uniform_rate_encode

__all__ = ["LavaLoihiRunner", "_subtractive_lif_cls", "_uniform_rate_encode"]
''',
    )
    (SRC / "chip_simulation/lava_loihi_runner.py").unlink(missing_ok=True)


IMPORT_REPLACEMENTS: list[tuple[str, str]] = [
    ("mimarsinan.models.spiking.unified_core_flow", "mimarsinan.models.spiking.unified.flow"),
    ("mimarsinan.models.spiking.hybrid_core_flow", "mimarsinan.models.spiking.hybrid.flow"),
    ("mimarsinan.models.nn.decorators", "mimarsinan.models.nn.decorators"),
    ("mimarsinan.models.nn.activations", "mimarsinan.models.nn.activations"),
    ("mimarsinan.chip_simulation.sanafe.runner_analysis", "mimarsinan.chip_simulation.sanafe.analysis"),
    ("mimarsinan.chip_simulation.sanafe.runner", "mimarsinan.chip_simulation.sanafe.runner"),
    ("mimarsinan.chip_simulation.sanafe.records", "mimarsinan.chip_simulation.sanafe.records"),
    ("mimarsinan.chip_simulation.sanafe.net_synth", "mimarsinan.chip_simulation.sanafe.net_synth"),
    ("mimarsinan.chip_simulation.sanafe.arch_synth", "mimarsinan.chip_simulation.sanafe.arch_synth"),
    ("mimarsinan.chip_simulation.simulation_runner", "mimarsinan.chip_simulation.simulation_runner"),
    ("mimarsinan.chip_simulation.lava_loihi_runner", "mimarsinan.chip_simulation.lava_loihi"),
]


def migrate_imports() -> int:
    total = 0
    scan = [ROOT / "src", ROOT / "tests", ROOT / "scripts"]
    for base in scan:
        for path in sorted(base.rglob("*.py")):
            if "mimarsinan-baseline-test" in str(path):
                continue
            text = path.read_text(encoding="utf-8")
            new = text
            new = new.replace(
                "mimarsinan.models.spiking.unified_core_flow",
                "mimarsinan.models.spiking.unified.flow",
            )
            new = new.replace(
                "mimarsinan.models.spiking.hybrid_core_flow",
                "mimarsinan.models.spiking.hybrid.flow",
            )
            new = new.replace(
                "mimarsinan.chip_simulation.sanafe.runner_analysis",
                "mimarsinan.chip_simulation.sanafe.analysis",
            )
            new = new.replace(
                "mimarsinan.chip_simulation.lava_loihi_runner",
                "mimarsinan.chip_simulation.lava_loihi",
            )
            if new != text:
                path.write_text(new, encoding="utf-8")
                total += 1
    return total


def fix_hybrid_ttfs_constants() -> None:
    ttfs = _read("models/spiking/hybrid/ttfs_step.py")
    if "TTFS_FIRING_MODES" not in ttfs:
        ttfs = ttfs.replace(
            "from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE",
            "from mimarsinan.models.spiking.spiking_config import (\n"
            "    COMPUTE_DTYPE,\n"
            "    TTFS_FIRING_MODES,\n"
            "    TTFS_SPIKING_MODES,\n"
            ")",
        )
        ttfs = ttfs.replace(
            "class HybridTtfsStepMixin:",
            "class HybridTtfsStepMixin:\n\n"
            "    _TTFS_FIRING_MODES = TTFS_FIRING_MODES\n"
            "    _TTFS_SPIKING_MODES = TTFS_SPIKING_MODES",
        )
        _write("models/spiking/hybrid/ttfs_step.py", ttfs)


def fix_runner_core_imports() -> None:
    core = _read("chip_simulation/sanafe/runner/core.py")
    core = core.replace(
        "from .arch_synth import _sanafe, build_architecture, derive_arch_spec\n",
        "from mimarsinan.chip_simulation.sanafe.arch_synth import _sanafe, build_architecture, derive_arch_spec\n",
    )
    core = core.replace(
        "from .net_synth import (\n",
        "from mimarsinan.chip_simulation.sanafe.net_synth import (\n",
    )
    core = core.replace(
        "from .presets import PRESETS\n",
        "from mimarsinan.chip_simulation.sanafe.presets import PRESETS\n",
    )
    core = core.replace(
        "from .records import (\n",
        "from mimarsinan.chip_simulation.sanafe.records import (\n",
    )
    core = (
        "from mimarsinan.chip_simulation.sanafe.runner.neural_stage import SanafeNeuralStageMixin\n"
        "from mimarsinan.chip_simulation.sanafe.runner.segment_io import SanafeSegmentIOMixin\n\n"
        + core
    )
    _write("chip_simulation/sanafe/runner/core.py", core)


def fix_net_synth_build() -> None:
    build = _read("chip_simulation/sanafe/net_synth/build.py")
    build = build.replace("from .arch_synth import _sanafe", "from mimarsinan.chip_simulation.sanafe.arch_synth import _sanafe")
    build = build.replace("from .neuron_model import", "from mimarsinan.chip_simulation.sanafe.neuron_model import")
    build = build.replace("from .presets import", "from mimarsinan.chip_simulation.sanafe.presets import")
    _write("chip_simulation/sanafe/net_synth/build.py", build)


def fix_spike_trains_imports() -> None:
    st = _read("chip_simulation/sanafe/net_synth/spike_trains.py")
    if "import numpy" not in st:
        st = "from __future__ import annotations\n\nfrom typing import Any, Dict, Optional, Tuple\n\nimport numpy as np\n\n" + st
    _write("chip_simulation/sanafe/net_synth/spike_trains.py", st)


def fix_unified_flow_dup() -> None:
    flow = _read("models/spiking/unified/flow.py")
    # Remove duplicate to_spikes block if present twice
    while flow.count("def to_spikes(") > 1:
        first = flow.index("    def to_spikes(")
        second = flow.index("    def to_spikes(", first + 1)
        flow = flow[:second].rstrip() + "\n" + flow[flow.index("    def _get_weight", second):]
    _write("models/spiking/unified/flow.py", flow)


def fix_lava_runner() -> None:
    runner = _read("chip_simulation/lava_loihi/runner.py")
    if "SimulationRunner" not in runner:
        runner = runner.replace(
            "from mimarsinan.chip_simulation.simulation_runner import SimulationRunner\n",
            "from mimarsinan.chip_simulation.simulation_runner import SimulationRunner\n",
        )
    runner = (
        "from __future__ import annotations\n\n"
        + runner.replace("from __future__ import annotations\n\n", "", 1)
    )
    if "_load_test_samples" not in runner:
        pass
    _write("chip_simulation/lava_loihi/runner.py", runner)


def update_chip_init() -> None:
    init = _read("chip_simulation/__init__.py")
    init = init.replace(
        "from mimarsinan.chip_simulation.simulation_runner import SimulationRunner",
        "from mimarsinan.chip_simulation.simulation_runner import SimulationRunner",
    )
    _write("chip_simulation/__init__.py", init)


def update_check_module_budget() -> None:
    path = ROOT / "scripts" / "check_module_budget.py"
    text = path.read_text(encoding="utf-8")
    old_allow = '''ALLOWLIST_FILES: frozenset[str] = frozenset({
    "gui/server.py",
    "models/hybrid_core_flow.py",
    "models/unified_core_flow.py",
    "chip_simulation/sanafe/runner.py",
    "chip_simulation/sanafe/runner_analysis.py",
    "chip_simulation/lava_loihi_runner.py",'''
    new_allow = '''ALLOWLIST_FILES: frozenset[str] = frozenset({
    "gui/server.py",'''
    if old_allow in text:
        text = text.replace(old_allow, new_allow)
        path.write_text(text, encoding="utf-8")


def main() -> int:
    split_unified_hybrid()
    split_nn()
    split_runner_analysis()
    split_sanafe_runner()
    split_records()
    split_net_synth()
    split_arch_synth()
    split_ttfs_segment()
    split_spike_recorder()
    split_simulation_runner()
    split_lava_loihi()
    fix_hybrid_ttfs_constants()
    fix_runner_core_imports()
    fix_net_synth_build()
    fix_spike_trains_imports()
    fix_unified_flow_dup()
    fix_lava_runner()
    update_chip_init()
    migrate_imports()
    update_check_module_budget()
    print("Phase 3 wave 3 split complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
