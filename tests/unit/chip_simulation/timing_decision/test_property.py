"""The compiled timing somas decide every integral form as the Python executor does, at every window.

nevresim's quantized timing neuron (both of the toolchain's weight types) and SANA-FE's quantized and per-cycle timing
somas against the executor's integer decision over every form of ``cxx_drivers`` at S 3, 5, 6, 7, 10, 11 and 12 and at
the powers of two 4, 8, 16 and 32; and the two smallest forms that parted by double rounding, by name.

``TIMING_DECISION_COUNTS=<path>`` writes the counts apart per window (a read of record, not a pass condition).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from .cxx_drivers import (
    CODES,
    THRESHOLDS,
    WEIGHT_TYPES,
    WINDOWS,
    build_nevresim_driver,
    build_sanafe,
    exact_ramp_fire,
    executor_blocks,
    flat_fire,
    run_nevresim_driver,
    sanafe_fire,
    step_rows,
    weight_rows,
)

pytestmark = [pytest.mark.slow]


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    return tmp_path_factory.mktemp("timing_decision")


@pytest.fixture(scope="module")
def reference():
    return {s: executor_blocks(s) for s in WINDOWS}


@pytest.fixture(scope="module")
def nevresim_steps(work):
    return run_nevresim_driver(build_nevresim_driver(work / "nevresim"))


@pytest.fixture(scope="module")
def sanafe(work):
    build = build_sanafe(work / "sanafe")
    if build is None:
        pytest.skip("no SANA-FE header tree: set MIMARSINAN_SANAFE_SRC or build the plugins once")
    return build


def _record(key: str, counts) -> None:
    path = os.environ.get("TIMING_DECISION_COUNTS")
    if not path:
        return
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    data[key] = counts
    with open(path, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)


def _apart(got: np.ndarray, want: np.ndarray) -> int:
    assert got.shape == want.shape
    return int(np.count_nonzero(got != want))


def _form_position(s: int, n: int, theta: int, ws, b: int, ks) -> int:
    """Index of one form in the drivers' order (inputs, threshold, weights, bias, steps)."""
    at = 0
    for n_ in (1, 2):
        per_theta = len(weight_rows(n_)) * len(CODES) * (s + 1) ** n_
        if n_ < n:
            at += len(THRESHOLDS) * per_theta
            continue
        at += THRESHOLDS.index(theta) * per_theta
        at += weight_rows(n).index(tuple(ws)) * len(CODES) * (s + 1) ** n
        at += CODES.index(b) * (s + 1) ** n
        return at + step_rows(s, n).index(tuple(ks))
    raise AssertionError("unreachable")


#: The windows of the strict comparator's read (``<``): the Python executor decides the lattice inclusively only, so
#: the reference there is the integer program's own ramp (``exact_ramp_fire``).
STRICT_WINDOWS = (3, 6, 7, 11, 8)


@pytest.fixture(scope="module")
def nevresim_strict_steps(work):
    binary = build_nevresim_driver(work / "nevresim_strict", STRICT_WINDOWS, "StrictCompare")
    return run_nevresim_driver(binary, STRICT_WINDOWS)


@pytest.mark.timeout(3600)
def test_the_executor_decides_as_the_exact_ramp(reference):
    """The reference itself: the Python executor's decision is the integer program's own on every form."""
    apart = {s: _apart(flat_fire(reference[s]), exact_ramp_fire(s, strict=False)) for s in WINDOWS}
    assert apart == {s: 0 for s in WINDOWS}, apart


@pytest.mark.timeout(3600)
def test_nevresim_strict_neuron_decides_as_the_exact_ramp(nevresim_strict_steps):
    apart = {wt: {s: _apart(nevresim_strict_steps[wt][s], exact_ramp_fire(s, strict=True)) for s in STRICT_WINDOWS}
             for wt in WEIGHT_TYPES}
    _record("nevresim_strict", apart)
    assert apart == {wt: {s: 0 for s in STRICT_WINDOWS} for wt in WEIGHT_TYPES}, apart


@pytest.mark.timeout(3600)
def test_nevresim_quantized_neuron_decides_as_the_executor(nevresim_steps, reference):
    apart = {wt: {s: _apart(nevresim_steps[wt][s], flat_fire(reference[s])) for s in WINDOWS}
             for wt in WEIGHT_TYPES}
    _record("nevresim", apart)
    assert apart == {wt: {s: 0 for s in WINDOWS} for wt in WEIGHT_TYPES}, apart


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("soma", ["quantized", "cycle"])
def test_sanafe_timing_soma_decides_as_the_executor(sanafe, reference, soma):
    apart = {s: _apart(sanafe_fire(sanafe, reference[s], soma), flat_fire(reference[s])) for s in WINDOWS}
    _record(f"sanafe_{soma}", apart)
    assert apart == {s: 0 for s in WINDOWS}, apart


@pytest.mark.timeout(3600)
def test_nevresim_smallest_form_at_window_3(nevresim_steps, reference):
    """A producer that fired at step 1 (a = 2/3), weight 2, bias -1, threshold 1: the exact membrane 1/3 reaches the
    threshold at step 2; double sums 0.9999999999999999 at step 2 and never fires."""
    at = _form_position(3, 1, 1, (2,), -1, (1,))
    assert int(flat_fire(reference[3])[at]) == 2
    for wt in WEIGHT_TYPES:
        assert int(nevresim_steps[wt][3][at]) == 2, wt


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("soma", ["quantized", "cycle"])
def test_sanafe_identity_relay_at_window_7(sanafe, reference, soma):
    """An identity relay (weight 1, bias 0, threshold 1) of a producer that fired at step 1 fires at step 1; double
    computes 7 (1 - 6/7) as 1.0000000000000002 and fires at step 2."""
    at = _form_position(7, 1, 1, (1,), 0, (1,))
    assert int(flat_fire(reference[7])[at]) == 1
    assert int(sanafe_fire(sanafe, reference[7], soma)[at]) == 1
