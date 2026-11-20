"""The one timing decision's reading of a membrane, checked at compile time in both builds' standards.

``timing_decision.hpp`` is constant-evaluable, so its rules are asserted by ``static_assert`` in a translation unit
compiled by the toolchain's compiler at C++17 (the plugins' standard) and C++20 (nevresim's): the lattice is read
exactly where the Python executor reads it (``membrane_lattice.lattice_membrane``: integral weights, bias and
threshold, every input read with a nonzero weight on the 1/(2S) grid, the bound below 2^53) and the decision is the
executor's ``S - floor(m / 2 theta)`` (inclusive) or its strict twin.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]

_UNIT = r"""
#include "simulator/compute_policy/timing_decision.hpp"
#include <array>

namespace t = nevresim::timing;

constexpr std::array<int, 1> w2{2};
constexpr std::array<double, 1> a23{2.0 / 3.0};
// The review's smallest nevresim form at S 3: m = 2 * 4 - 6 = 2, so the first step is 3 - floor(2 / 2) = 2.
static_assert(t::neuron_membrane(w2, a23, -1.0, 1.0, 3, 0.0).lattice);
static_assert(t::neuron_membrane(w2, a23, -1.0, 1.0, 3, 0.0).m == 2);
static_assert(t::fire_step(t::neuron_membrane(w2, a23, -1.0, 1.0, 3, 0.0), 1.0, 3, false, t::FloatPath::ramp) == 2);
// An input off the grid read with a nonzero weight leaves the lattice; read with a zero weight it does not.
constexpr std::array<double, 2> off{0.3, 0.5};
constexpr std::array<int, 2> reads_off{1, 1};
constexpr std::array<int, 2> skips_off{0, 1};
static_assert(!t::neuron_membrane(reads_off, off, 0.0, 1.0, 3, 0.0).lattice);
static_assert(t::neuron_membrane(skips_off, off, 0.0, 1.0, 3, 0.0).lattice);
// A midpoint (2k+1)/(2S) is on the grid.
constexpr std::array<double, 1> mid{0.5 / 3.0};
static_assert(t::neuron_membrane(std::array<int, 1>{1}, mid, 0.0, 1.0, 3, 0.0).m == 1);
// Integral weights, bias and threshold (to the dust), threshold at least 1 and below 2^62.
static_assert(!t::neuron_membrane(std::array<double, 1>{1.5}, a23, 0.0, 1.0, 3, 0.0).lattice);
static_assert(t::neuron_membrane(std::array<double, 1>{2.0000001}, a23, 0.0, 1.0, 3, 0.0).lattice);
static_assert(!t::neuron_membrane(w2, a23, 0.5, 1.0, 3, 0.0).lattice);
static_assert(!t::neuron_membrane(w2, a23, 0.0, 0.5, 3, 0.0).lattice);
static_assert(!t::neuron_membrane(w2, a23, 0.0, 2.5, 3, 0.0).lattice);
static_assert(!t::neuron_membrane(w2, a23, 0.0, 4611686018427387904.0, 3, 0.0).lattice);
static_assert(t::neuron_membrane(w2, a23, 0.0, 4611686018427387392.0, 3, 0.0).lattice);  // 2^62 - 512
// The reference's bound: sum |q| |w| + 2S |b| below 2^53.
constexpr std::array<double, 1> one{1.0};
static_assert(t::neuron_membrane(std::array<double, 1>{1099511627775.0}, one, 0.0, 1.0, 4096, 0.0).lattice);
static_assert(!t::neuron_membrane(std::array<double, 1>{1099511627776.0}, one, 0.0, 1.0, 4096, 0.0).lattice);
// The decision: inclusive fires on the tie, strict one step later; floor division toward minus infinity.
static_assert(t::lattice_fire_step(4, 1, 3, false) == 1);
static_assert(t::lattice_fire_step(4, 1, 3, true) == 2);
static_assert(t::lattice_fire_step(3, 1, 3, true) == 2);   // V = 1/2: 1 < 1/2 + t/3 first at t = 2
static_assert(t::lattice_fire_step(3, 1, 3, false) == 2);  // and 1 <= 1/2 + t/3 first at t = 2
static_assert(t::lattice_fire_step(-1, 1, 3, false) == -1);
static_assert(t::lattice_fire_step(-2, 1, 8, false) == -1);
static_assert(t::lattice_fire_step(1000, 1, 8, false) == 0);
static_assert(t::lattice_fire_step(1, 1, 8, false) == -1);
// A lattice membrane on a threshold the decision cannot double takes the float path.
static_assert(t::fire_step(t::Membrane{true, 2, 0.0}, 4611686018427387904.0, 3, false, t::FloatPath::closed_form) == -1);
// The SANA-FE readers: the preset numerator, and the arrivals' numerator over S.
static_assert(t::preset_membrane(0.0, true, 12.0).m == 12);
static_assert(t::arrivals_membrane(6, -1.0, 7, true, 0.0).m == 12 - 14);
// The float paths keep the backends' own arithmetic.
static_assert(t::ramp_fire_step(0.5, 1.0, 2, false) == 1);
static_assert(t::closed_form_fire_step(0.5, 1.0, 2) == 1);
static_assert(t::floored_threshold(0.0) == 1e-12);
int main() { return 0; }
"""


@pytest.mark.parametrize("standard", ["c++17", "c++20"])
def test_the_header_reads_and_decides_as_the_executor(tmp_path, standard):
    from mimarsinan.common.build_utils import find_cpp20_compiler
    from mimarsinan.common.env import nevresim_root

    compiler, _family = find_cpp20_compiler()
    if compiler is None:
        pytest.skip("no C++ compiler")
    unit = tmp_path / "header_unit.cpp"
    unit.write_text(_UNIT)
    done = subprocess.run([compiler, f"-std={standard}", "-fsyntax-only", "-Wall", "-Wextra", "-Werror",
                           "-I", str(Path(nevresim_root()) / "include"), str(unit)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-3000:]
