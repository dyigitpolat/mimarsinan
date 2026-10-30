"""[ODIN multi-core, Stage 1] the packaged fixtures on an NC-core WRAPPER, against the twin.

The card runs the frozen fixtures through ``odin_fpga_kernel_top`` built at NC
cores. This gate runs the SAME bytes through the SAME RTL -- the wrapper
testbench ``hw/tb/tb_odin_fpga_kernel_axi.v``, whose behavioural AXI4 memory
serves the program buffer and the stimulus buffer at the board's own split --
and requires the captured counts to equal the toolchain's cycle-accurate twin
at EVERY (sample, cycle, core, neuron), and the windows per sample:

  * the 2-core fixtures (R11a, the E2E segment) at NC 2 and NC 3, and the
    3-core micro-witnesses at NC 3, at every rotation of the fabric
    (``odin_board_driver.rotation_placements``), each placement retargeted by
    the shipped driver's own byte rule and decoded back per core;
  * the 1-core fixtures at NC 1 unchanged: the identity placement sends the
    frozen bytes themselves;
  * every fixture at NC 4, at every rotation: the second geometry built.

The fixtures are built by ``scripts/hacc/make_package.py``'s own recipes, so
the bytes simulated here are the bytes the package freezes. [slow]: each NC is
one Verilator build of the wrapper around NC vendored cores.

``ODIN_MC_SIM_RECORD=<path>`` appends one JSON line per run (fixture, NC,
placement, events, enabled cycles, agreement, wall) for the Stage 1 report.

[targeted pass] THE TRAINED NETWORK'S PASSES. The card got three passes of
``odin_narrowconv_mnist_wb4_s16`` wrong on the NC 4 bitstream (sample 23 and
sample 28 on core 1, sample 42 on core 0; ``stage1_review.md`` M1). The second
test below runs exactly those streams through the wrapper at NC 2 and NC 4: the
bytes the package's own executor sends in its as-run arm (program, then the
stimulus it builds from the frozen answer of the pass before), recorded off the
package's fake pyxrt with the package's frozen replay, at three stall seeds
under each control-port ordering (AW and W together, W first, AW first: the
orders the card's clock-crossing interconnect produces). Every run
must equal the replay's frozen events. ``ODIN_MC_PACKAGE=<dir>`` names the card
package (``make_package.py --deployment`` output); without it these cases skip
by name.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

from integration.odin_rtl_harness import require_simulator

from mimarsinan.chip_simulation.odin_fpga.kernel_sim import (
    CONTROL_ORDERS,
    run_kernel_program_over_axi,
)
from mimarsinan.chip_simulation.odin_rtl.stimulus import decode_ops

pytestmark = [pytest.mark.slow, pytest.mark.integration]

REPO = Path(__file__).resolve().parents[2]
RECORD_ENV = "ODIN_MC_SIM_RECORD"

#: (fixture, NC) pairs of the gate: every rotation of each is simulated.
GRID = (
    [(name, 1) for name in ("nc1_single_core_witness", "nc1_single_core_ceiling")]
    + [(name, nc) for name in ("r11a_three_sample", "e2e_two_core_segment")
       for nc in (2, 3)]
    + [("micro_multiplicity_and_rails", 3)]
    + [(name, 4) for name in ("nc1_single_core_witness", "nc1_single_core_ceiling",
                              "r11a_three_sample", "e2e_two_core_segment",
                              "micro_multiplicity_and_rails")]
)


@pytest.fixture(scope="module")
def packager():
    path = REPO / "scripts" / "hacc" / "make_package.py"
    spec = importlib.util.spec_from_file_location("multicore_sim_make_package", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_BUILT = {}


def _built(packager, name):
    """One fixture as the package freezes it: payloads, plan, twin traces."""
    if name not in _BUILT:
        fixture = next(item for item in packager.FIXTURES if item.name == name)
        mapping, rasters, length, chip_latency = fixture.build()
        export = packager.export_of(mapping)
        reference = packager.traces_for(
            mapping, rasters, simulation_length=length, chip_latency=chip_latency)
        per_cycle_inputs = [list(sample.per_cycle) for sample in reference]
        latencies = [int(v) for v in reference[0].trace.latencies]
        program, stimulus, plan = packager.split_payloads(
            export, per_cycle_inputs, latencies=latencies)
        neurons = [int(core.neurons_per_core) for core in mapping.cores]
        _BUILT[name] = (program, stimulus, plan, reference, latencies, length, neurons)
    return _BUILT[name]


def _words(payload: bytes):
    return [int.from_bytes(payload[i:i + 4], "little") for i in range(0, len(payload), 4)]


def _cases():
    cases = []
    for name, nc in GRID:
        cores = 1 if name.startswith("nc1_") else (3 if name.startswith("micro") else 2)
        for offset in range(nc):
            cases.append(pytest.param(name, nc, offset, cores,
                                      id=f"{name}-nc{nc}-r{offset}"))
    return cases


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("name, nc, offset, cores", _cases())
def test_the_placed_fixture_equals_the_twin_at_every_cycle(packager, name, nc, offset, cores):
    try:
        engine = require_simulator()
    except Exception as exc:  # pragma: no cover - the loud skip path
        pytest.skip(str(exc))
    driver = packager.driver
    program, stimulus, plan, reference, latencies, length, neurons = _built(packager, name)
    assert plan.n_cores == cores
    placement = driver.rotation_placements(cores, nc)[offset]
    placed_program = driver.retarget_payload(program, placement, what=f"{name}/program")
    placed_stimulus = driver.retarget_payload(stimulus, placement, what=f"{name}/stimulus")
    if placement == driver.identity_placement(cores):
        # NC 1 (and every identity) sends the frozen bytes themselves.
        assert placed_program == program and placed_stimulus == stimulus
    program_words = _words(placed_program)
    stream = program_words[:-1] + _words(placed_stimulus)
    ops = decode_ops(stream)
    started = time.monotonic()
    capture, status, build, _run = run_kernel_program_over_axi(
        list(ops), n_cores=nc, split=len(program_words) - 1)
    wall = time.monotonic() - started
    assert not status.err, f"{name}: the fabric raised err"
    assert not status.truncated, f"{name}: the capture reached its capacity"
    assert status.program_words == len(program_words)

    inverse = {fabric: logical for logical, fabric in placement.items()}
    counts = {}
    for event in capture.events:
        assert event.core in inverse, (
            f"{name}: fabric core {event.core} spiked and placement {placement} "
            f"never programmed it")
        sample, cycle = plan.decode_tag(event.tag)
        key = (sample, cycle, inverse[event.core], int(event.neuron))
        counts[key] = counts.get(key, 0) + 1

    frozen = packager._FrozenCosim(counts, plan.samples, plan.cycles_per_sample)
    differences = packager.compare_cycle_counts(frozen, reference)
    windows = [[list(row) for row in per_core] for per_core in packager._window_counts(
        counts, samples=plan.samples, cycles_per_sample=plan.cycles_per_sample,
        latencies=latencies, simulation_length=length, neurons=neurons)]
    expected = [[list(row) for row in sample.trace.window_counts()] for sample in reference]
    per_core_events = {logical: 0 for logical in placement}
    for (_s, _c, core, _n), value in counts.items():
        per_core_events[core] += value
    record = {
        "fixture": name, "nc": nc, "rotation": offset,
        "placement": {str(k): v for k, v in sorted(placement.items())},
        "engine": build.engine, "events": len(capture.events),
        "events_per_logical_core": {str(k): v for k, v in sorted(per_core_events.items())},
        "enabled_cycles": int(capture.cycles),
        "program_words": len(program_words), "stimulus_words": len(_words(placed_stimulus)),
        "program_sha256": hashlib.sha256(placed_program).hexdigest(),
        "stimulus_sha256": hashlib.sha256(placed_stimulus).hexdigest(),
        "cycle_differences": len(differences),
        "windows_equal": windows == expected,
        "wall_s": round(wall, 3), "build_seconds": round(build.build_seconds, 3),
        "build_cached": build.cached,
    }
    path = os.environ.get(RECORD_ENV)
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    assert capture.events, f"{name}: the fabric produced no output events"
    assert not differences, f"{name} at {placement}: {differences[:4]}"
    assert windows == expected


# ---------------------------------------------------------------------------
# [targeted pass] the trained network's passes the card got wrong
# ---------------------------------------------------------------------------

PACKAGE_ENV = "ODIN_MC_PACKAGE"
TRAINED_BUNDLE = "odin_narrowconv_mnist_wb4_s16"
#: (core, sample) of every pass the NC 4 card got wrong in the review's runs.
TRAINED_PASSES = ((1, 23), (1, 28), (0, 42))
#: (stall seed, control-port order) per case: every stall seed (0 is no
#: starvation, 3 and 11 starve the stream) under every control-port order (AW
#: and W together, W first, AW first), so a stall effect and an ordering effect
#: are told apart.
TRAINED_STALL_SEEDS = (0, 3, 11)
TRAINED_RUNS = tuple((seed, order) for order in CONTROL_ORDERS
                     for seed in TRAINED_STALL_SEEDS)
TRAINED_NC = (2, 4)

#: A recording wrapper around the package's fake pyxrt: every kernel call's
#: declared program and stimulus words, in order, exactly as a card gets them.
_RECORDER = '''
import atexit, importlib.util, json, os, sys
_spec = importlib.util.spec_from_file_location("odin_fake_pyxrt_recorded", os.environ["MC_SHIPPED_FAKE"])
_orig = importlib.util.module_from_spec(_spec)
sys.modules["odin_fake_pyxrt_recorded"] = _orig
_spec.loader.exec_module(_orig)
globals().update({k: v for k, v in _orig.__dict__.items() if not k.startswith("__")})
_RUNS = []
_call = _orig.kernel.__call__
def _recording_call(self, *args):
    _RUNS.append([bytes(args[0]._data[:int(args[3]) * 4]).hex(), bytes(args[1]._data[:int(args[4]) * 4]).hex()])
    return _call(self, *args)
_orig.kernel.__call__ = _recording_call
@atexit.register
def _dump():
    with open(os.environ["MC_RECORD_PATH"], "w") as handle:
        json.dump(_RUNS, handle)
'''

_TRAINED = {}


def _canonical(stimulus: bytes) -> bytes:
    """The stimulus with every core argument on core 0: how the replay keys a pass."""
    words = _words(stimulus)
    arity = {0: 0, 1: 3, 2: 3, 3: 2, 4: 1, 5: 1}
    index = 0
    while index < len(words):
        if words[index] in (1, 2, 3):
            words[index + 1] = 0
        index += 1 + arity[words[index]]
    return b"".join(word.to_bytes(4, "little") for word in words)


def _trained_streams(tmp_root: Path, nc: int):
    """{(core, sample): (program, stimulus, frozen events)} as the package's executor sends them."""
    if nc in _TRAINED:
        return _TRAINED[nc]
    package = os.environ.get(PACKAGE_ENV)
    if not package:
        pytest.skip(f"{PACKAGE_ENV} is unset: the trained network's bundle and its frozen "
                    f"replay are the card package's (make_package.py --deployment)")
    package = Path(package)
    bundle = package / "deployment" / f"{TRAINED_BUNDLE}.json"
    replay_path = package / "deployment" / f"{TRAINED_BUNDLE}_capture.json"
    assert bundle.is_file() and replay_path.is_file(), (
        f"{PACKAGE_ENV}={package} holds no {bundle.name} and {replay_path.name}")
    work = tmp_root / f"trained_nc{nc}"
    work.mkdir(parents=True, exist_ok=True)
    recorder = work / "recording_fake.py"
    recorder.write_text(_RECORDER)
    record = work / "runs.json"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", ODIN_KERNEL_CORES=str(nc),
               MC_SHIPPED_FAKE=str(package / "host" / "fake_pyxrt_for_selftest.py"),
               MC_RECORD_PATH=str(record))
    import subprocess
    result = subprocess.run(
        [sys.executable, "-B", str(package / "host" / "odin_deployment_executor.py"),
         "--xclbin", "/selftest/no-such.xclbin", "--fake-pyxrt", str(recorder),
         "--declare-cores", str(nc), "--arm", "as_run", "--bundle", str(bundle),
         "--replay", str(replay_path), "--results", str(work / "report")],
        capture_output=True, text=True, env=env, timeout=1800)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    replay = json.loads(replay_path.read_text())
    by_key = {entry["stimulus_sha256"]: entry for entry in replay["runs"]}
    streams = {}
    for program_hex, stimulus_hex in json.loads(record.read_text()):
        program, stimulus = bytes.fromhex(program_hex), bytes.fromhex(stimulus_hex)
        frozen = by_key.get(hashlib.sha256(_canonical(stimulus)).hexdigest())
        if frozen is not None:
            streams[(int(frozen["core"]), int(frozen["sample"]))] = (program, stimulus, frozen)
    document = json.loads(bundle.read_text())
    _TRAINED[nc] = (streams, document)
    return _TRAINED[nc]


def _trained_cases():
    return [pytest.param(nc, core, sample, seed, order,
                         id=f"{TRAINED_BUNDLE}-nc{nc}-c{core}s{sample}-seed{seed}-order{order}")
            for nc in TRAINED_NC for core, sample in TRAINED_PASSES
            for seed, order in TRAINED_RUNS]


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("nc, core, sample, seed, order", _trained_cases())
def test_the_trained_networks_failed_passes_equal_the_frozen_answers(
        tmp_path_factory, nc, core, sample, seed, order):
    try:
        require_simulator()
    except Exception as exc:  # pragma: no cover - the loud skip path
        pytest.skip(str(exc))
    streams, document = _trained_streams(tmp_path_factory.getbasetemp(), nc)
    program, stimulus, frozen = streams[(core, sample)]
    fabric = document["pass_order"].index(core) % nc
    program_words = _words(program)
    ops = decode_ops(program_words[:-1] + _words(stimulus))
    started = time.monotonic()
    capture, status, build, _run = run_kernel_program_over_axi(
        list(ops), n_cores=nc, split=len(program_words) - 1,
        stall_seed=seed, control_order=order)
    wall = time.monotonic() - started
    got = sorted((int(event.tag), int(event.neuron)) for event in capture.events)
    want = sorted((int(event[0]), int(event[3])) for event in frozen["events"])
    record = {
        "case": "trained_pass", "bundle": TRAINED_BUNDLE, "nc": nc, "core": core,
        "sample": sample, "fabric_core": fabric, "stall_seed": seed,
        "control_order": order, "events": len(capture.events),
        "frozen_events": len(want), "enabled_cycles": int(capture.cycles),
        "equal": got == want, "err": bool(status.err),
        "stall_gaps": status.stall_gaps, "stall_cycles": status.stall_cycles,
        "program_sha256": hashlib.sha256(program).hexdigest(),
        "stimulus_sha256": hashlib.sha256(stimulus).hexdigest(),
        "wall_s": round(wall, 3), "build_cached": build.cached,
    }
    path = os.environ.get(RECORD_ENV)
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    assert not status.err, f"pass {core}/{sample}: the fabric raised err"
    assert not status.truncated
    if seed:
        assert status.stall_gaps > 0, "the stall seed injected no starvation"
    assert {int(event.core) for event in capture.events} <= {fabric}, (
        f"pass {core}/{sample}: a record from a core other than fabric core {fabric}")
    assert got == want, (
        f"pass {core}/{sample} at NC {nc}, seed {seed}, order {order}: "
        f"{len(got)} events against {len(want)} frozen")
