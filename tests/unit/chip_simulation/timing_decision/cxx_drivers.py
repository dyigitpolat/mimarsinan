"""The compiled timing somas beside the Python executor, form by form.

The forms: every integral neuron with 1 or 2 inputs, weight codes -3 to 3 (not all zero), bias codes -3 to 3, integer
thresholds 1 to 6, and every producer step of each input (``a = (S - k) / S`` for k in 0..S, k = S a producer that
never fired), at a window ``S``. The reference is the Python executor's decision (``TtfsAnalyticalExecutor``, the
``ttfs_quantized`` closed form on the integer chip's lattice) on a one-core program per threshold.

The compiled sides are built through the toolchain's own build paths:

- nevresim: a driver of ``NeuronCompute<Config, TTFSQuantizedCompute<S, InclusiveCompare>>`` (the neuron every
  quantized timing core runs) compiled by ``compile_simulator`` (the toolchain's compiler, flags and include tree);
  its weight type is the toolchain's ``int`` (Weight Quantization) or ``double``.
- SANA-FE: the plugins built by their own CMake definition (``sanafe/plugins/CMakeLists.txt``, what
  ``scripts/build_sanafe_plugins.py`` runs) and loaded by a driver as SANA-FE loads them (``dlopen`` and the
  ``create_*`` factory), each soma given the attributes the toolchain writes: the quantized soma the executor's preset
  membrane through ``apply_ttfs_preset_membranes``, the per-cycle soma the network build's threshold, bias and window
  and its inputs as single-spike arrivals (``current_in`` at cycle ``k + 1``, the delivery the soma decodes); both take
  whatever lattice record the stage stamps (``stamp_timing_lattice``), when the tree has one.
"""

from __future__ import annotations

import itertools
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fake_cores import FakeCore

REPO_ROOT = Path(__file__).resolve().parents[4]
PLUGIN_SOURCE = REPO_ROOT / "src" / "mimarsinan" / "chip_simulation" / "sanafe" / "plugins"

#: The windows the property reads: apart today wherever S is not a power of two, and the powers of two (exact today).
NON_DYADIC_WINDOWS: Tuple[int, ...] = (3, 5, 6, 7, 10, 11, 12)
DYADIC_WINDOWS: Tuple[int, ...] = (4, 8, 16, 32)
WINDOWS: Tuple[int, ...] = NON_DYADIC_WINDOWS + DYADIC_WINDOWS
CODES: Tuple[int, ...] = tuple(range(-3, 4))
THRESHOLDS: Tuple[int, ...] = tuple(range(1, 7))
WEIGHT_TYPES: Tuple[str, ...] = ("int", "double")

#: The soma attributes a record carries, each with a presence flag (the driver sets only those present).
ATTRIBUTE_KEYS: Tuple[str, ...] = (
    "threshold", "bias", "active_start", "active_length", "preset_membrane", "lattice", "preset_numerator",
)
RECORD_WIDTH = 6 + 2 * len(ATTRIBUTE_KEYS)


def weight_rows(n: int) -> List[Tuple[int, ...]]:
    """``itertools.product`` order over the codes, the all-zero row left out."""
    return [ws for ws in itertools.product(CODES, repeat=n) if any(ws)]


def step_rows(s: int, n: int) -> List[Tuple[int, ...]]:
    return list(itertools.product(range(s + 1), repeat=n))


def form_count(s: int) -> int:
    return sum(len(THRESHOLDS) * len(weight_rows(n)) * len(CODES) * (s + 1) ** n for n in (1, 2))


# ------------------------------------------------------------------------------------------------ the Python executor


@dataclass
class Block:
    """One threshold's forms with ``n`` inputs at window ``s``: neurons ordered (weights, bias), samples by steps."""

    s: int
    n: int
    theta: int
    weights: np.ndarray  # (N, n)
    biases: np.ndarray  # (N,)
    steps_in: np.ndarray  # (B, n) producer steps
    hcm: SimpleNamespace
    fire: np.ndarray  # (B, N) the executor's fire step, -1 when it never fires
    membrane: np.ndarray  # (B, N) the executor's membrane V
    lattice: Optional[Tuple[np.ndarray, np.ndarray]]  # the executor's lattice record of the core, when it keeps one


def _source(core: int, neuron: int, *, is_input: bool):
    from mimarsinan.code_generation.cpp_chip_model import SpikeSource

    return SpikeSource(core, neuron, is_input, False, False)


def one_core_program(weights: np.ndarray, biases: np.ndarray, theta: float) -> SimpleNamespace:
    """A one-core timing program: ``weights`` (N, A) over A segment inputs, one bias per neuron, one threshold."""
    n_neurons, n_axons = weights.shape
    core = FakeCore(
        axons_per_core=n_axons, neurons_per_core=n_neurons, available_axons=0, available_neurons=0,
        threshold=float(theta), core_matrix=np.asarray(weights, dtype=np.float64).T.copy(),
        axon_sources=[_source(-1, i, is_input=True) for i in range(n_axons)],
        hardware_bias=np.asarray(biases, dtype=np.float64), latency=0,
    )
    return SimpleNamespace(
        cores=[core], output_sources=np.asarray([_source(0, j, is_input=False) for j in range(n_neurons)]),
    )


def executor_block(s: int, n: int, theta: int) -> Block:
    from mimarsinan.chip_simulation.ttfs.ttfs_executor import TtfsAnalyticalExecutor

    rows = weight_rows(n)
    weights = np.asarray([ws for ws in rows for _ in CODES], dtype=np.float64)
    biases = np.asarray([b for _ in rows for b in CODES], dtype=np.float64)
    steps_in = np.asarray(step_rows(s, n), dtype=np.int64)
    x = (s - steps_in).astype(np.float64) / s
    hcm = one_core_program(weights, biases, theta)
    result = TtfsAnalyticalExecutor().run_segment(hcm, x, simulation_length=s, spiking_mode="ttfs_quantized")
    act = np.asarray(result.per_core_activations[0], dtype=np.float64)
    fire = np.where(act > 0.0, s - np.rint(act * s), -1).astype(np.int64)
    lattice = getattr(result, "membrane_lattice", None)
    return Block(s=s, n=n, theta=theta, weights=weights, biases=biases, steps_in=steps_in, hcm=hcm, fire=fire,
                 membrane=np.asarray(result.membrane_voltages[0], dtype=np.float64),
                 lattice=None if lattice is None else lattice[0])


def executor_blocks(s: int) -> List[Block]:
    """Every form at window ``s``, in the drivers' order: inputs, threshold, weights, bias, steps."""
    return [executor_block(s, n, theta) for n in (1, 2) for theta in THRESHOLDS]


def flat_fire(blocks: Sequence[Block]) -> np.ndarray:
    return np.concatenate([b.fire.T.reshape(-1) for b in blocks])


def exact_ramp_fire(s: int, *, strict: bool) -> np.ndarray:
    """The integer program's own decision on every form at window ``s``, in the drivers' order, from the ramp's
    definition alone: the membrane is ``m / (2S)`` with ``m = sum w_j 2 (S - k_j) + 2S b`` (integers), and the neuron
    fires at the first step ``t`` in ``[0, S)`` where ``m + 2 t theta`` reaches ``2 S theta`` (strict: passes it)."""
    out = []
    for n in (1, 2):
        rows = np.asarray(weight_rows(n), dtype=np.int64)
        steps = np.asarray(step_rows(s, n), dtype=np.int64)
        q = 2 * (s - steps)  # (B, n)
        for theta in THRESHOLDS:
            m = (rows @ q.T)[:, None, :] + (2 * s * np.asarray(CODES, dtype=np.int64))[None, :, None]  # (W, b, B)
            gap = 2 * s * theta - m
            if strict:
                t = np.where(gap >= 0, gap // (2 * theta) + 1, 0)
            else:
                t = np.maximum(0, -((-gap) // (2 * theta)))
            out.append(np.where(t < s, t, -1).reshape(-1))
    return np.concatenate(out)


# --------------------------------------------------------------------------------------------------------- nevresim

_NEVRESIM_DRIVER = r"""
#include "simulator/compute_policy/neuron/ttfs_quantized_neuron_compute.hpp"
#include "simulator/compute_policy/compare_policy/inclusive_compare.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int k_codes[] = {-3, -2, -1, 0, 1, 2, 3};

template <typename W, int S, std::size_t A>
void run_forms(std::vector<signed char>& out)
{
    using Cfg = nevresim::ChipConfiguration<
        W, double, A, 1, 1, A, 1, 1, nevresim::constant<nevresim::MembraneLeak<W>>{0}>;
    using Neuron = nevresim::NeuronCompute<
        Cfg, nevresim::TTFSQuantizedCompute<S, nevresim::@COMPARE@>>;
    std::vector<std::array<int, A>> rows;
    std::array<int, A> w{};
    for (int i0 = 0; i0 < 7; ++i0)
    {
        if constexpr (A == 1)
        {
            w[0] = k_codes[i0];
            if (w[0] != 0) rows.push_back(w);
        }
        else
        {
            for (int i1 = 0; i1 < 7; ++i1)
            {
                w[0] = k_codes[i0];
                w[1] = k_codes[i1];
                if (w[0] != 0 || w[1] != 0) rows.push_back(w);
            }
        }
    }
    for (int theta = 1; theta <= 6; ++theta)
    {
        for (const auto& row : rows)
        {
            for (int b : k_codes)
            {
                nevresim::NeuronWeights<Cfg{}, W, double> nw{};
                for (std::size_t j = 0; j < A; ++j) nw.weights_[j] = static_cast<W>(row[j]);
                nw.threshold_ = static_cast<double>(theta);
                nw.bias_ = static_cast<W>(b);
                std::array<int, A> k{};
                const long total = (A == 1) ? (S + 1) : (S + 1) * (S + 1);
                for (long idx = 0; idx < total; ++idx)
                {
                    if constexpr (A == 1) { k[0] = static_cast<int>(idx); }
                    else { k[0] = static_cast<int>(idx / (S + 1)); k[1] = static_cast<int>(idx % (S + 1)); }
                    std::array<double, A> signal{};
                    for (std::size_t j = 0; j < A; ++j) signal[j] = static_cast<double>(S - k[j]) / S;
                    Neuron neuron{};
                    neuron.load_weights(nw);
                    neuron.reset();
                    double value = 0.0;
                    for (int step = 0; step < S; ++step) value = neuron(signal);
                    out.push_back(value > 0.0
                        ? static_cast<signed char>(S - std::llround(value * S))
                        : static_cast<signed char>(-1));
                }
            }
        }
    }
}

template <typename W, int S>
void run_window(std::vector<signed char>& out)
{
    run_forms<W, S, 1>(out);
    run_forms<W, S, 2>(out);
}

template <typename W>
void run_type(std::vector<signed char>& out)
{
    @WINDOW_CALLS@
}

} // namespace

int main()
{
    std::vector<signed char> out;
    run_type<int>(out);
    run_type<double>(out);
    std::fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}
"""


def nevresim_driver_source(windows: Sequence[int] = WINDOWS, compare: str = "InclusiveCompare") -> str:
    calls = "\n    ".join(f"run_window<W, {s}>(out);" for s in windows)
    source = _NEVRESIM_DRIVER.replace("@WINDOW_CALLS@", calls).replace("@COMPARE@", compare)
    if compare != "InclusiveCompare":
        source = f'#include "simulator/compute_policy/compare_policy/strict_compare.hpp"\n{source}'
    return source


def build_nevresim_driver(work: Path, windows: Sequence[int] = WINDOWS, compare: str = "InclusiveCompare") -> Path:
    """Compiles the driver through the toolchain's nevresim build (``compile_simulator``) for the config-driven
    comparator ``compare`` (``InclusiveCompare`` for ``<=``, ``StrictCompare`` for ``<``)."""
    from mimarsinan.chip_simulation.nevresim.compile_nevresim import compile_simulator
    from mimarsinan.common.env import nevresim_root

    (work / "main").mkdir(parents=True, exist_ok=True)
    (work / "chip").mkdir(parents=True, exist_ok=True)
    (work / "main" / "main.cpp").write_text(nevresim_driver_source(windows, compare))
    binary = compile_simulator(str(work), nevresim_root(), output_path=str(work / "bin" / "driver"), verbose=False)
    if binary is None:
        raise RuntimeError("the nevresim driver did not compile through compile_simulator")
    return Path(binary)


def run_nevresim_driver(binary: Path, windows: Sequence[int] = WINDOWS) -> Dict[str, Dict[int, np.ndarray]]:
    """``{weight type: {S: fire steps in the drivers' order}}``."""
    raw = np.frombuffer(subprocess.run([str(binary)], check=True, capture_output=True).stdout, dtype=np.int8)
    out: Dict[str, Dict[int, np.ndarray]] = {}
    at = 0
    for wt in WEIGHT_TYPES:
        out[wt] = {}
        for s in windows:
            size = form_count(s)
            out[wt][s] = raw[at:at + size].astype(np.int64)
            at += size
    if at != raw.size:
        raise RuntimeError(f"the nevresim driver wrote {raw.size} steps, the forms are {at}")
    return out


# ---------------------------------------------------------------------------------------------------------- SANA-FE

_SANAFE_DRIVER = r"""
#include <dlfcn.h>

#include <array>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "attribute.hpp"
#include "mapped.hpp"
#include "pipeline.hpp"

namespace {

const char* k_keys[] = {@KEYS@};
constexpr int k_key_count = @KEY_COUNT@;
constexpr int k_width = 6 + 2 * k_key_count;

sanafe::ModelAttribute attribute(double v)
{
    sanafe::ModelAttribute a;
    a.value = v;
    return a;
}

// The toolchain hands SANA-FE window positions and lengths as Python ints (an ``int`` attribute).
sanafe::ModelAttribute integer_attribute(int v)
{
    sanafe::ModelAttribute a;
    a.value = v;
    return a;
}

bool integer_key(const std::string& key)
{
    return key == "active_start" || key == "active_length";
}

// The stage hands the lattice flag as a Python bool (a ``bool`` attribute).
sanafe::ModelAttribute flag_attribute(double v)
{
    sanafe::ModelAttribute a;
    a.value = (v != 0.0);
    return a;
}

bool flag_key(const std::string& key)
{
    return key == "lattice";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 4) { std::fprintf(stderr, "usage: driver <plugin.so> <factory> <arrivals|none>\n"); return 2; }
    void* handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 3; }
    using Factory = sanafe::PipelineUnit* (*)();
    auto make = reinterpret_cast<Factory>(dlsym(handle, argv[2]));
    if (make == nullptr) { std::fprintf(stderr, "dlsym: %s\n", dlerror()); return 4; }
    const bool arrivals = std::string(argv[3]) == "arrivals";
    std::unique_ptr<sanafe::PipelineUnit> soma(make());
    std::array<double, k_width> r{};
    std::vector<signed char> out;
    while (std::fread(r.data(), sizeof(double), k_width, stdin) == static_cast<std::size_t>(k_width))
    {
        const int s = static_cast<int>(r[0]);
        const int n = static_cast<int>(r[1]);
        soma->reset();
        soma->set_attribute_hw("simulation_length", integer_attribute(s));
        long start = 0;
        for (int i = 0; i < k_key_count; ++i)
        {
            if (r[6 + 2 * i] == 0.0) continue;
            const std::string key = k_keys[i];
            soma->set_attribute_neuron(0, key, integer_key(key)
                ? integer_attribute(static_cast<int>(r[7 + 2 * i]))
                : (flag_key(key) ? flag_attribute(r[7 + 2 * i]) : attribute(r[7 + 2 * i])));
            if (std::string(k_keys[i]) == "active_start") start = static_cast<long>(r[7 + 2 * i]);
        }
        signed char fire = -1;
        for (long t = 1; t <= start + 2L * s + 2; ++t)
        {
            const long cycle = t - 1;
            std::optional<double> current;
            if (arrivals)
            {
                double sum = 0.0;
                bool any = false;
                for (int j = 0; j < n; ++j)
                {
                    const int k = static_cast<int>(r[4 + j]);
                    if (k < s && cycle == k + 1) { sum += r[2 + j]; any = true; }
                }
                if (any) current = sum;
            }
            const sanafe::PipelineResult res = soma->update(std::size_t{0}, current, t);
            if (res.status == sanafe::fired) { fire = static_cast<signed char>(cycle - start); break; }
        }
        out.push_back(fire);
    }
    std::fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}
"""


def sanafe_source_tree() -> Optional[Path]:
    """SANA-FE's header tree: ``MIMARSINAN_SANAFE_SRC``, else the tree the plugin build fetched into ``build/``."""
    override = os.environ.get("MIMARSINAN_SANAFE_SRC", "").strip()
    candidates = [Path(override).expanduser()] if override else []
    candidates.append(REPO_ROOT / "build" / "mimarsinan_sanafe_plugins" / "_deps" / "sanafe_src-src" / "src")
    for path in candidates:
        if (path / "pipeline.hpp").is_file():
            return path.resolve()
    return None


def build_plugins(work: Path, sanafe_src: Path) -> Path:
    """The plugins through their own CMake definition (configure and build, as the build script runs them)."""
    if shutil.which("cmake") is None:
        raise RuntimeError("cmake is required to build the SANA-FE plugins")
    build = work / "plugins"
    subprocess.run(["cmake", "-S", str(PLUGIN_SOURCE), "-B", str(build), f"-DSANAFE_SRC={sanafe_src}"],
                   check=True, capture_output=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", "2"], check=True, capture_output=True)
    return build


def build_sanafe_driver(work: Path, sanafe_src: Path) -> Path:
    from mimarsinan.common.build_utils import find_cpp20_compiler

    compiler, _family = find_cpp20_compiler()
    if compiler is None:
        raise RuntimeError("no C++ compiler")
    source = work / "sanafe_driver.cpp"
    keys = ", ".join(f'"{k}"' for k in ATTRIBUTE_KEYS)
    source.write_text(_SANAFE_DRIVER.replace("@KEYS@", keys).replace("@KEY_COUNT@", str(len(ATTRIBUTE_KEYS))))
    binary = work / "sanafe_driver"
    subprocess.run([compiler, "-O2", "-std=c++17", str(source), "-I", str(sanafe_src), "-ldl", "-o", str(binary)],
                   check=True, capture_output=True)
    return binary


class _CapturedNeuron:
    """A SANA-FE neuron stand-in: ``set_attributes`` merges as ``Neuron::set_attributes`` does (insert or assign)."""

    def __init__(self) -> None:
        self.model_attributes: Dict[str, float] = {}

    def set_attributes(self, *, model_attributes=None, **_ignored) -> None:
        for key, value in (model_attributes or {}).items():
            self.model_attributes[key] = value


def _stamp():
    """The stage's lattice stamp on a timing soma, when the tree has one (the set export has none)."""
    try:
        from mimarsinan.chip_simulation.sanafe.net_synth.timing_lattice import stamp_timing_lattice
    except ImportError:
        return None
    return stamp_timing_lattice


def _record_rows(block: Block, attrs: List[List[Dict[str, float]]]) -> np.ndarray:
    """Records in the drivers' order (neuron-major, then samples): window, inputs, weights, steps, attributes."""
    n_samples, n_neurons = block.fire.shape
    rec = np.zeros((n_neurons, n_samples, RECORD_WIDTH), dtype=np.float64)
    rec[:, :, 0] = block.s
    rec[:, :, 1] = block.n
    for j in range(block.n):
        rec[:, :, 2 + j] = block.weights[:, j][:, None]
        rec[:, :, 4 + j] = block.steps_in[:, j][None, :]
    for b in range(n_samples):
        for neuron in range(n_neurons):
            values = attrs[b][neuron]
            for i, key in enumerate(ATTRIBUTE_KEYS):
                if key in values:
                    rec[neuron, b, 6 + 2 * i] = 1.0
                    rec[neuron, b, 7 + 2 * i] = float(values[key])
    return rec.reshape(-1, RECORD_WIDTH)


def quantized_soma_attributes(block: Block) -> List[List[Dict[str, float]]]:
    """What the stage writes on each quantized timing soma, sample by sample: the executor's preset membrane
    (``apply_ttfs_preset_membranes``) and the stage's lattice stamp, when the tree has one."""
    from mimarsinan.chip_simulation.sanafe.net_synth import apply_ttfs_preset_membranes

    stamp = _stamp()
    n_neurons = block.fire.shape[1]
    out = []
    for b in range(block.fire.shape[0]):
        group = [_CapturedNeuron() for _ in range(n_neurons)]
        apply_ttfs_preset_membranes({0: group}, block.hcm, [block.membrane[b:b + 1]],
                                    spiking_mode="ttfs_quantized", simulation_length=block.s)
        if stamp is not None and block.lattice is not None:
            stamp({0: group}, block.hcm, [(block.lattice[0][b:b + 1], block.lattice[1][b:b + 1])],
                  simulation_length=block.s, preset=True)
        out.append([g.model_attributes for g in group])
    return out


def cycle_soma_attributes(block: Block) -> List[List[Dict[str, float]]]:
    """What the network build writes on each per-cycle timing soma of a depth-0 core (threshold, bias, the window
    ``[S, 2S)``) and the stage's lattice stamp, when the tree has one."""
    stamp = _stamp()
    n_neurons = block.fire.shape[1]
    out = []
    for b in range(block.fire.shape[0]):
        group = [_CapturedNeuron() for _ in range(n_neurons)]
        for neuron, g in enumerate(group):
            g.set_attributes(model_attributes={
                "threshold": float(block.theta), "bias": float(block.biases[neuron]),
                "active_start": block.s, "active_length": block.s,
            })
        if stamp is not None and block.lattice is not None:
            stamp({0: group}, block.hcm, [(block.lattice[0][b:b + 1], block.lattice[1][b:b + 1])],
                  simulation_length=block.s, preset=False)
        out.append([g.model_attributes for g in group])
    return out


def run_sanafe_soma(driver: Path, plugin: Path, factory: str, records: np.ndarray, *, arrivals: bool) -> np.ndarray:
    done = subprocess.run([str(driver), str(plugin), factory, "arrivals" if arrivals else "none"],
                          input=np.ascontiguousarray(records).tobytes(), check=True, capture_output=True)
    fire = np.frombuffer(done.stdout, dtype=np.int8).astype(np.int64)
    if fire.size != records.shape[0]:
        raise RuntimeError(f"the SANA-FE driver wrote {fire.size} steps for {records.shape[0]} records")
    return fire


@dataclass
class SanafeBuild:
    driver: Path
    plugins: Path

    def library(self, name: str) -> Path:
        return self.plugins / f"lib{name}.so"


def build_sanafe(work: Path) -> Optional[SanafeBuild]:
    sanafe_src = sanafe_source_tree()
    if sanafe_src is None:
        return None
    work.mkdir(parents=True, exist_ok=True)
    return SanafeBuild(driver=build_sanafe_driver(work, sanafe_src), plugins=build_plugins(work, sanafe_src))


def sanafe_fire(build: SanafeBuild, blocks: Sequence[Block], soma: str) -> np.ndarray:
    """Fire steps of the ``quantized`` or ``cycle`` timing soma on every form of ``blocks``, in the drivers' order."""
    if soma == "quantized":
        records = np.concatenate([_record_rows(b, quantized_soma_attributes(b)) for b in blocks])
        return run_sanafe_soma(build.driver, build.library("mimarsinan_ttfs_quantized_soma"),
                               "create_mimarsinan_ttfs_quantized_soma", records, arrivals=False)
    records = np.concatenate([_record_rows(b, cycle_soma_attributes(b)) for b in blocks])
    return run_sanafe_soma(build.driver, build.library("mimarsinan_ttfs_cycle_soma"),
                           "create_mimarsinan_ttfs_cycle_soma", records, arrivals=True)


def scratch_dir(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))
