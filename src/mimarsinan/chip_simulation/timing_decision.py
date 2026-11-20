"""The range of the compiled timing decision, and its named refusal at code generation.

The compiled timing somas (nevresim's quantized timing neuron, SANA-FE's quantized and per-cycle timing somas) decide
through one C++ header, ``nevresim/include/simulator/compute_policy/timing_decision.hpp``: the integer decision of the
Python executor's integral neurons (``chip_simulation/ttfs/membrane_lattice.py``,
``models/spiking/wire_semantics.py:ttfs_quantized_staircase_lattice_np``) in 64-bit integers.

What it holds. The membrane numerator ``m`` over ``2S`` is formed only while ``sum |round(a 2S)| |w| + 2S |b|`` is
below 2^53, the reference's own lattice bound, read per sample by both sides (above it both take their float path), so
``|m| < 2^53``. The comparison divides by ``2 theta``, formed in 64 bits by the reference (an ``int64`` product) and by
the header alike, so it holds for an integral threshold below 2^62. Every other integer of the decision (the window
``S``, the fire step) is far inside 64 bits. A timing program whose cores carry an integral threshold from 2^62 on
therefore has no 64-bit decision on either side, and its code generation refuses it here, by name.

Its version. The header carries the decision's version (``nevresim::timing::abi``) and each SANA-FE soma that decides
through it exports it (``mimarsinan_timing_decision_abi``). A soma plugin is loaded once before the toolchain hands it
the executor's lattice record, and one built without the version, or with another, is refused here by name: a stale
build would read the record's attributes as unknown and decide in its older arithmetic in silence.
"""

from __future__ import annotations

import ctypes
import functools
import os
from typing import Iterable, List, Optional

import numpy as np
import yaml

from mimarsinan.chip_simulation.spiking_semantics import forces_activation_quantization, is_analytical_ttfs
from mimarsinan.common.refusal.toolchain_refusal import ToolchainRefusal

#: The first integral threshold whose double ``2 theta`` passes 64 bits (``timing_decision.hpp: threshold_range``).
THRESHOLD_RANGE = 2 ** 62
#: The reference's lattice bound on ``|m|`` (``membrane_lattice.EXACT_INTEGER_RANGE``), read per sample.
MEMBRANE_RANGE = 2 ** 53
#: Float dust on an integral threshold (``segment_arrays.INTEGRAL_DUST``).
_INTEGRAL_DUST = 1e-6
#: The decision's version (``timing_decision.hpp: nevresim::timing::abi``), exported by each soma under this symbol.
TIMING_DECISION_ABI = 1
VERSION_SYMBOL = "mimarsinan_timing_decision_abi"
#: The SANA-FE soma models that decide through the one timing decision (``sanafe/plugins``), each exporting the version.
TIMING_SOMA_MODELS = ("mimarsinan_ttfs_quantized_soma", "mimarsinan_ttfs_cycle_soma")
_REMEDY = "rebuild with scripts/build_sanafe_plugins.py"


class TimingDecisionRangeError(ToolchainRefusal, ValueError):
    """A timing program whose integer decision 64 bits cannot hold: an integral threshold at or above 2^62."""


class TimingDecisionVersionError(ToolchainRefusal, RuntimeError):
    """A timing soma plugin built without the toolchain's timing decision (a stale build, or other sources)."""


@functools.lru_cache(maxsize=None)
def _exported_version(path: str, _inode: int, _mtime_ns: int, _size: int) -> Optional[int]:
    """The decision version a plugin file exports, None when it exports none (one load per file identity)."""
    exported = getattr(ctypes.CDLL(path), VERSION_SYMBOL, None)
    if exported is None:
        return None
    exported.restype, exported.argtypes = ctypes.c_int, []
    return int(exported())


def require_timing_decision_version(library: str) -> None:
    """Refuse, by name and with the remedy, a timing soma plugin whose decision version is absent or not the
    toolchain's (``TIMING_DECISION_ABI``)."""
    try:
        st = os.stat(library)
    except OSError as e:
        raise TimingDecisionVersionError(f"the SANA-FE timing soma plugin {library} is missing: {_REMEDY}") from e
    found = _exported_version(os.path.realpath(library), st.st_ino, st.st_mtime_ns, st.st_size)
    if found is None:
        raise TimingDecisionVersionError(f"the SANA-FE timing soma plugin {library} was built without the timing "
                                         f"decision {TIMING_DECISION_ABI}: {_REMEDY}")
    if found != TIMING_DECISION_ABI:
        raise TimingDecisionVersionError(f"the SANA-FE timing soma plugin {library} was built with the timing "
                                         f"decision {found}, not the toolchain's {TIMING_DECISION_ABI}: {_REMEDY}")


def timing_somas_of_arch(yaml_text: str) -> List[str]:
    """The plugin of every soma of a SANA-FE architecture whose model decides through the one timing decision."""
    found: List[str] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            attrs = node.get("attributes")
            if isinstance(attrs, dict) and attrs.get("model") in TIMING_SOMA_MODELS and attrs.get("plugin"):
                found.append(str(attrs["plugin"]))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(yaml.safe_load(yaml_text))
    return found


def require_arch_timing_somas(arch_path: str) -> None:
    """A custom architecture names its own somas: each timing soma it names is read for the decision's version."""
    with open(arch_path) as handle:
        for library in timing_somas_of_arch(handle.read()):
            require_timing_decision_version(library)


def _thresholds(core) -> np.ndarray:
    return np.asarray(getattr(core, "threshold"), dtype=np.float64).reshape(-1)


def require_timing_decision_range(cores: Iterable, *, where: str) -> None:
    """Refuse, by name, a timing program a core of which carries an integral threshold the 64-bit decision cannot
    double (``where`` names the code generation that asks)."""
    for index, core in enumerate(cores):
        theta = _thresholds(core)
        integral = np.abs(theta - np.round(theta)) <= _INTEGRAL_DUST
        beyond = integral & (np.round(theta) >= float(THRESHOLD_RANGE))
        if bool(beyond.any()):
            raise TimingDecisionRangeError(
                f"{where}: core {index} carries the integral threshold {float(theta[beyond][0]):.17g}, at or above "
                f"2^62, so the integer timing decision cannot form 2 theta in 64 bits (the Python executor's and the "
                f"compiled somas' alike); scale the program's thresholds below 2^62")


def timing_program_in_range(mapping, spiking_mode: str):
    """``mapping``, after nevresim's code generation has refused it by name when it is a quantized timing program
    (the one nevresim decides through the timing decision) whose decision 64 bits cannot hold."""
    if is_analytical_ttfs(spiking_mode) and forces_activation_quantization(spiking_mode):
        require_timing_decision_range(mapping.cores, where="nevresim's quantized timing code generation")
    return mapping
