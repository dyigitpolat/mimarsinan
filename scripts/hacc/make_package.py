#!/usr/bin/env python3
"""Assemble the owner-uploadable HACC package: ONE zip, ONE sh, no repo needed.

HACC login is 2FA and this machine cannot reach the cluster, so everything the
board bring-up needs has to travel as a single artifact. That artifact is
``dist/odin_hacc_package.zip`` and this script is its only author.

WHAT GOES IN, AND WHY ONLY THAT
  * the RTL the ``v++`` build compiles — the kernel wrapper, the vendored ODIN
    tree, the BRAM overlays — plus ``build_xclbn.sh``, ``cards.sh`` (the
    per-card SSOT: platform, part, v++ config, preferred Vitis, partitions) and
    EVERY card's ``odin_<card>.cfg`` verbatim from ``scripts/hacc/``, and the
    ``kernel.xml`` this script emits from the host-side register SSOT so the
    build node needs no ``src/`` tree;
  * PRE-EXPORTED FIXTURES: the program word-stream, the stimulus word-stream,
    the run parameters and the EXPECTED per-neuron counts of each fixture,
    frozen HERE by running the committed cosimulation and asserting it against
    the committed golden gates. Packaging FAILS if any of them disagrees;
  * the thin host driver and its fake ``pyxrt``, so the whole driver runs green
    with no hardware before it is ever pointed at a card;
  * the DEPLOYMENT BUNDLE and its replay evidence under ``deployment/``, plus
    the bundle-schema module copied VERBATIM out of ``src/`` and hash-verified
    here, the executor that runs it as host-mediated passes, and the die-map
    renderer the post-build mining calls;
  * ``bootstrap_hacc.sh``, ``run_all.sh``, ``scripts/status.sh``,
    ``scripts/chip_cache.sh``, ``collect_results.sh``, ``README_HACC.md``.

Nothing else from the repository tree.

TWO ARTIFACTS, IN THIS ORDER. ``dist/odin_hacc_package.zip`` is written first,
then hashed, then ``dist/bootstrap_hacc.sh`` is written outside it with that
hash embedded — so the standalone bootstrap the owner uploads alongside the zip
verifies the FINAL bytes. The copy of the bootstrap inside the zip necessarily
keeps its placeholder and says so rather than pretending to verify.

DETERMINISM. Every zip entry is stored with a fixed timestamp and mode, in
sorted order, and every fixture's JSON is canonical (sorted keys, no spaces), so
two runs of this script over the same tree produce byte-identical zips. The
cosimulation results are cached under ``build/hacc_fixture_cache/`` keyed by the
plan and the RTL sources, which is what makes that rebuild cheap as well as
stable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
PACKAGE_SRC = REPO / "scripts" / "hacc" / "package"
DIST = REPO / "dist"
STAGE = DIST / "odin_hacc_package"
ZIP_PATH = DIST / "odin_hacc_package.zip"
#: The DEPLOYMENT artifact: the same bring-up package plus one exported bundle,
#: so a board node gets the bitstream flow and the network in a single upload.
DEPLOYMENT_ZIP_PATH = DIST / "odin_hacc_deployment.zip"
DEPLOYMENT_BOOTSTRAP = DIST / "bootstrap_hacc_deployment.sh"
#: The index phase 8 reads to learn WHICH bundle a deployment package deploys.
DEPLOYMENT_INDEX = "deployment/DEPLOYMENT.json"
CACHE = REPO / "build" / "hacc_fixture_cache"

#: The bundle-schema SSOT ships VERBATIM: the board executor runs THESE bytes,
#: so there is no second copy to drift, only a copy to hash-verify.
BUNDLE_MODULE = (
    REPO / "src" / "mimarsinan" / "chip_simulation" / "odin_deployment_bundle.py")
DEPLOYMENT_SRC = PACKAGE_SRC / "deployment"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

#: The driver is imported, not re-implemented: the fixture SEAL (its canonical
#: byte image and self-hash) has exactly one definition, and it is the one the
#: board node will verify with.
_DRIVER_SPEC = importlib.util.spec_from_file_location(
    "odin_board_driver", PACKAGE_SRC / "host" / "odin_board_driver.py")
assert _DRIVER_SPEC is not None and _DRIVER_SPEC.loader is not None
driver = importlib.util.module_from_spec(_DRIVER_SPEC)
_DRIVER_SPEC.loader.exec_module(driver)

import numpy as np  # noqa: E402

from integration.odin_fpga_harness import (  # noqa: E402
    INPUT_LINES,
    TIMESTEPS,
    two_core_mapping,
)
from integration.odin_rtl_harness import (  # noqa: E402
    MEMBRANE_CEILING,
    ODIN_LAW,
    compare_cycle_counts,
    export_of,
    hard_core,
    mapping_of,
    traces_for,
)

from mimarsinan.chip_simulation import odin_deployment_bundle  # noqa: E402
from mimarsinan.chip_simulation.odin_fpga import kernel_registers  # noqa: E402
from mimarsinan.chip_simulation.odin_fpga.kernel_registers import (  # noqa: E402
    KERNEL_NAME,
    SHIPPED_CAPTURE_EVENTS,
    WORD_BYTES,
)
from mimarsinan.chip_simulation.odin_fpga.payload import split_payloads  # noqa: E402
from mimarsinan.chip_simulation.odin_rtl.cosim import run_cosim  # noqa: E402
from mimarsinan.chip_simulation.odin_rtl.reference import simulate_cycles  # noqa: E402
from mimarsinan.chip_simulation.odin_rtl.toolchain import (  # noqa: E402
    available_engine,
    design_sources,
    kernel_sources,
)
from mimarsinan.code_generation.cpp_chip_model import SpikeSource  # noqa: E402
from mimarsinan.mapping.latency.chip import ChipLatency  # noqa: E402

#: 1980-01-01, the earliest a zip entry may carry: no build clock leaks in.
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


class PackagingRefusal(RuntimeError):
    """A golden gate did not hold; no package is written."""


# ---------------------------------------------------------------------------
# The fixtures, each built from the COMMITTED harness the gates already use
# ---------------------------------------------------------------------------


@dataclass
class Fixture:
    """One fixture's recipe plus the witnesses packaging must see hold."""

    name: str
    title: str
    description: str
    build: Callable[[], Tuple[Any, List[List[List[int]]], int, int | None]]
    witnesses: Callable[[Dict[str, Any], Any], Dict[str, Any]]
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


def _r11a_build():
    """Plan §7 row 15: theta at the ceiling, multiplicity, and a CLEAR-only sample.

    Byte-for-byte the fixture of ``tests/integration/test_odin_rtl_r11a.py``.
    """
    axons = neurons = 16
    length = 4
    rng = np.random.default_rng(7)
    producer_w = rng.integers(-7, 8, size=(axons + 1, neurons)).astype(np.float64)
    consumer_w = rng.integers(3, 8, size=(axons, neurons)).astype(np.float64)
    consumer_w[:, 0] = 7.0
    producer = hard_core(
        producer_w, threshold=15.0,
        sources=[SpikeSource(-2, index, is_input=True) for index in range(axons)]
        + [SpikeSource(-3, 0, is_always_on=True)])
    consumer = hard_core(
        consumer_w, threshold=float(MEMBRANE_CEILING),
        sources=[SpikeSource(0, index) for index in range(axons)])
    mapping = mapping_of(
        [producer, consumer], [SpikeSource(1, index) for index in range(neurons)])
    chip_latency = int(ChipLatency(mapping).calculate())
    ones = [[1] * axons for _ in range(length)]
    alternating = [
        [1 if (cycle + axon) % 2 == 0 else 0 for axon in range(axons)]
        for cycle in range(length)
    ]
    rasters = [ones, [list(row) for row in ones], alternating]
    return mapping, rasters, length, chip_latency


def _micro_build():
    """Plan §7 rows 9/15: the multiplicity-3 slot and the saturation rails.

    Byte-for-byte the fixture of ``tests/integration/test_odin_rtl_micro.py``.
    """
    length, inputs, rail_slots, rail_weight = 5, 14, 9, 7.0
    witness = np.zeros((7, 5), dtype=np.float64)
    witness[0] = [5.0, 1.0, 0.0, 0.0, 0.0]
    witness[2] = [0.0, 0.0, 3.0, -3.0, 0.0]
    witness[3] = [0.0, 0.0, -2.0, 5.0, 0.0]
    witness[4] = witness[5] = witness[6] = [0.0, 0.0, 0.0, 0.0, 5.0]
    core0 = hard_core(
        witness, threshold=4.0,
        sources=[SpikeSource(-2, index, is_input=True)
                 for index in (0, 1, 2, 3, 4, 4, 4)])
    consumer = np.array([[5.0, 0.0], [0.0, 3.0], [0.0, -3.0]], dtype=np.float64)
    core1 = hard_core(consumer, threshold=5.0,
                      sources=[SpikeSource(0, 4)] * 3)
    rail = np.full((rail_slots, 1), rail_weight)
    core2 = hard_core(
        rail, threshold=float(MEMBRANE_CEILING),
        sources=[SpikeSource(-2, 5 + index, is_input=True)
                 for index in range(rail_slots)])
    mapping = mapping_of(
        [core0, core1, core2], [SpikeSource(1, 0), SpikeSource(2, 0)])
    ChipLatency(mapping).calculate()
    row = [1] * inputs
    row[1] = 0
    return mapping, [[list(row) for _ in range(length)]], length, None


def _e2e_build():
    """The P7a end-to-end deployment fixture, at the transport's own boundary.

    The two-core hybrid segment of ``tests/integration/test_odin_fpga_e2e.py``,
    driven by the saturated entry raster its ``_twin_windows`` uses.
    """
    mapping = two_core_mapping()
    chip_latency = int(ChipLatency(mapping).calculate())
    raster = [[1] * INPUT_LINES for _ in range(TIMESTEPS)]
    return mapping, [raster], TIMESTEPS, chip_latency


def _single_core_witness_build():
    """The NC=1 board-runnable fixture: multiplicity, the floor, and CLEAR.

    Same soma law and same harness as the micro witnesses, on ONE core, because
    the v1 packaging flow builds NC=1 only and a two-core program cannot run
    there at all. Three samples with the second repeating the first, so the
    per-sample CLEAR is the only thing that can make them equal.
    """
    length, inputs = 4, 5
    witness = np.zeros((7, 5), dtype=np.float64)
    witness[0] = [5.0, 1.0, 0.0, 0.0, 0.0]
    witness[2] = [0.0, 0.0, 3.0, -3.0, 0.0]
    witness[3] = [0.0, 0.0, -2.0, 5.0, 0.0]
    witness[4] = witness[5] = witness[6] = [0.0, 0.0, 0.0, 0.0, 5.0]
    core = hard_core(
        witness, threshold=4.0,
        sources=[SpikeSource(-2, index, is_input=True)
                 for index in (0, 1, 2, 3, 4, 4, 4)])
    mapping = mapping_of([core], [SpikeSource(0, index) for index in range(5)])
    chip_latency = int(ChipLatency(mapping).calculate())
    ones = [[1] * inputs for _ in range(length)]
    alternating = [
        [1 if (cycle + axon) % 2 == 0 else 0 for axon in range(inputs)]
        for cycle in range(length)
    ]
    rasters = [ones, [list(row) for row in ones], alternating]
    return mapping, rasters, length, chip_latency


def _single_core_ceiling_build():
    """The NC=1 saturation rail: theta AT the 8-bit membrane ceiling.

    Nine slots of weight 7 charge 63 per cycle; the fifth cycle's charge crosses
    255 only because the membrane CLAMPS at 8'hFF instead of wrapping — a wrap
    would sit at 3 and the spike would be missed.
    """
    slots, weight, length = 9, 7.0, 5
    core = hard_core(
        np.full((slots, 1), weight), threshold=float(MEMBRANE_CEILING),
        sources=[SpikeSource(-2, index, is_input=True) for index in range(slots)])
    mapping = mapping_of([core], [SpikeSource(0, 0)])
    chip_latency = int(ChipLatency(mapping).calculate())
    return mapping, [[[1] * slots for _ in range(length)]], length, chip_latency


# --- the witnesses each fixture must still carry when it is frozen ----------


def _multiplicity(reference) -> int:
    return max(
        (max(counts) for sample in reference
         for per_core in sample.trace.outputs for counts in per_core),
        default=0)


def _clear_witness(document, context) -> Dict[str, Any]:
    """Sample 1 repeats sample 0's input, so equality can only come from CLEAR."""
    windows = document["expected"]["window"]
    if len(windows) < 3 or windows[0] != windows[1]:
        raise PackagingRefusal(
            f"{document['name']}: the repeated sample did not reproduce the "
            f"first, so the per-sample CLEAR is not witnessed")
    if windows[0] == windows[2]:
        raise PackagingRefusal(
            f"{document['name']}: the third sample equals the first, so the "
            f"fixture cannot tell a working CLEAR from a dead injector")
    mapping, rasters, reference = context[0], context[1], context[4]
    # The teeth test of tests/integration/test_odin_rtl_r11a.py, verbatim: run
    # the fixture's first raster TWICE with no CLEAR between and compare the two
    # spans. They must differ, or carried membranes would reproduce sample 0
    # anyway and the equality above would prove nothing.
    axons = len(rasters[0][0])
    span = len(reference[0].trace.outputs)
    continuous = simulate_cycles(
        mapping, soma_law=ODIN_LAW,
        input_counts=[row for _ in range(2) for row in rasters[0]]
        + [[0] * axons] * (2 * span),
        simulation_length=2 * span, chip_latency=0)
    if continuous.outputs[:span] == continuous.outputs[span:2 * span]:
        raise PackagingRefusal(
            f"{document['name']}: the membranes come back to their starting "
            f"state on their own, so an equal second sample proves nothing "
            f"about CLEAR")
    return {"clear_only_second_sample": True, "clear_has_teeth": True}


def _r11a_witnesses(document, context) -> Dict[str, Any]:
    mapping, _rasters, _length, _latency = context[:4]
    found = dict(_clear_witness(document, context))
    if float(mapping.cores[1].threshold) != float(MEMBRANE_CEILING):
        raise PackagingRefusal("R11a: the consumer theta left the membrane ceiling")
    found.update({"theta_at_membrane_ceiling": MEMBRANE_CEILING,
                  "consecutive_samples": len(document["expected"]["window"])})
    return found


def _micro_witnesses(document, context) -> Dict[str, Any]:
    per_cycle = driver.per_cycle_table(document["expected"]["per_cycle"])
    producer = [per_cycle.get((0, cycle, 0, 4), 0) for cycle in range(5)]
    if producer != [3] * 5:
        raise PackagingRefusal(
            f"micro: the multiplicity producer emitted {producer}, not three "
            f"spikes in every cycle — the multiplicity-3 witness is gone")
    rail = [per_cycle.get((0, cycle, 2, 0), 0) for cycle in range(5)]
    if rail != [0, 0, 0, 0, 1]:
        raise PackagingRefusal(
            f"micro: the saturation rail fired {rail}, not on the fifth cycle "
            f"alone — the 8'hFF clamp witness is gone")
    return {"multiplicity_per_cycle": 3, "ceiling_rail_cycle": 4,
            "theta_at_membrane_ceiling": MEMBRANE_CEILING}


def _e2e_witnesses(document, context) -> Dict[str, Any]:
    reference = context[4]
    multiplicity = _multiplicity(reference)
    if multiplicity < 2:
        raise PackagingRefusal(
            "e2e: no multiplicity on the wire — a per-cycle law would "
            "reproduce this fixture and the fixture would prove nothing")
    return {"multiplicity_per_cycle": multiplicity}


def _single_core_witnesses(document, context) -> Dict[str, Any]:
    found = dict(_clear_witness(document, context))
    per_cycle = driver.per_cycle_table(document["expected"]["per_cycle"])
    producer = [per_cycle.get((0, cycle, 0, 4), 0) for cycle in range(1, 4)]
    if producer != [3, 3, 3]:
        raise PackagingRefusal(
            f"single-core: the multiplicity producer emitted {producer} rather "
            f"than three spikes per cycle")
    found["multiplicity_per_cycle"] = 3
    return found


def _ceiling_witnesses(document, context) -> Dict[str, Any]:
    per_cycle = driver.per_cycle_table(document["expected"]["per_cycle"])
    rail = [per_cycle.get((0, cycle, 0, 0), 0) for cycle in range(6)]
    if sum(rail) != 1 or rail[:4] != [0, 0, 0, 0]:
        raise PackagingRefusal(
            f"ceiling: the rail fired {rail} — a saturating membrane fires "
            f"exactly once, on the cycle whose charge overflows 8'hFF")
    return {"theta_at_membrane_ceiling": MEMBRANE_CEILING,
            "ceiling_rail_fired_once": True}


FIXTURES: Tuple[Fixture, ...] = (
    Fixture(
        name="r11a_three_sample",
        title="R11a: three samples, a ceiling theta and a CLEAR-only repeat",
        description=(
            "The plan §7 row 15 fixture: a 16x16 producer feeding a consumer "
            "whose theta sits at the 8-bit membrane ceiling, three consecutive "
            "samples of which the second repeats the first's input, and no "
            "weight or parameter byte rewritten between them."),
        build=_r11a_build, witnesses=_r11a_witnesses,
        notes="tests/integration/test_odin_rtl_r11a.py"),
    Fixture(
        name="micro_multiplicity_and_rails",
        title="Micro-witnesses: a multiplicity-3 slot and the saturation rails",
        description=(
            "Three cores carrying the single-row witness, the inhibitory "
            "row-pair, a producer that emits THREE spikes in one cycle, and a "
            "theta-255 rail that fires only because the membrane clamps at "
            "8'hFF instead of wrapping."),
        build=_micro_build, witnesses=_micro_witnesses,
        notes="tests/integration/test_odin_rtl_micro.py"),
    Fixture(
        name="e2e_two_core_segment",
        title="E2E: the deployed two-core segment at the transport boundary",
        description=(
            "The hybrid program the P7a end-to-end gate deploys, frozen at the "
            "device seam: the same export, the same per-cycle injection plan "
            "and the same latencies the segment driver hands a transport."),
        build=_e2e_build, witnesses=_e2e_witnesses,
        notes="tests/integration/test_odin_fpga_e2e.py"),
    Fixture(
        name="nc1_single_core_witness",
        title="NC=1: one core, multiplicity 3, the inhibitory floor, CLEAR",
        description=(
            "The board-runnable fixture. The v1 packaging flow builds NC=1 "
            "only, so this is the one that actually programs the shipped "
            "bitstream: one core, three samples with the second repeating the "
            "first, a producer emitting three spikes per cycle, and an "
            "inhibitory row pair that floors at zero instead of wrapping."),
        build=_single_core_witness_build, witnesses=_single_core_witnesses,
        notes="derived from tests/integration/test_odin_rtl_micro.py core 0"),
    Fixture(
        name="nc1_single_core_ceiling",
        title="NC=1: theta at the 8-bit membrane ceiling",
        description=(
            "The board-runnable saturation rail: nine slots of weight 7 and a "
            "theta of 255, which fires on the cycle whose charge overflows the "
            "membrane register — and would MISS if that register wrapped."),
        build=_single_core_ceiling_build, witnesses=_ceiling_witnesses,
        notes="derived from tests/integration/test_odin_rtl_micro.py core 2"),
)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def git_head() -> Tuple[str, bool]:
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        capture_output=True, text=True, check=True).stdout.strip())
    return head, dirty


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rtl_digest() -> str:
    """One hash over every RTL source the cosimulation and the build compile."""
    digest = hashlib.sha256()
    for path in sorted(design_sources(overlay=True) + kernel_sources()):
        digest.update(str(path.relative_to(REPO)).encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


#: ONE device-protocol table. The driver is the file the board node runs, so the
#: driver's copy is the one that ships; packaging REFUSES if the repository-side
#: SSOT (`kernel_registers`) has drifted from it, because two tables that
#: disagree would drive a kernel nobody built.
KERNEL_TABLE = dict(driver.KERNEL_TABLE)

#: (driver name, kernel_registers name) for every constant both halves declare.
_PROTOCOL_MIRRORS = (
    ("KERNEL_NAME", "KERNEL_NAME"),
    ("ARG_PROGRAM", "ARG_PROGRAM"),
    ("ARG_STIMULUS", "ARG_STIMULUS"),
    ("ARG_CAPTURE", "ARG_CAPTURE"),
    ("ARG_PROGRAM_WORDS", "ARG_PROGRAM_WORDS"),
    ("ARG_STIMULUS_WORDS", "ARG_STIMULUS_WORDS"),
    ("ARG_CAPTURE_WORDS", "ARG_CAPTURE_WORDS"),
    ("KERNEL_ARGS", "KERNEL_ARGS"),
    ("WORD_BYTES", "WORD_BYTES"),
    ("CAPTURE_HEADER_WORDS", "CAPTURE_HEADER_WORDS"),
    ("HEADER_EVENTS_SEEN", "HEADER_EVENTS_SEEN"),
    ("HEADER_DEVICE_CYCLES", "HEADER_DEVICE_CYCLES"),
    ("CAPTURE_RECORD_WORDS", "CAPTURE_RECORD_WORDS"),
    ("RECORD_TAG", "RECORD_TAG"),
    ("RECORD_CYCLE", "RECORD_CYCLE"),
    ("RECORD_CORE", "RECORD_CORE"),
    ("RECORD_NEURON", "RECORD_NEURON"),
    ("CAPTURE_NO_VERDICT", "CAPTURE_NO_VERDICT"),
    ("SHIPPED_PROGRAM_WORDS", "SHIPPED_PROGRAM_WORDS"),
    ("SHIPPED_CAPTURE_WORDS", "SHIPPED_CAPTURE_WORDS"),
    ("SHIPPED_CAPTURE_EVENTS", "SHIPPED_CAPTURE_EVENTS"),
)


#: The seal is the fixtures' discipline and the bundles': two implementations
#: that disagreed would let a board node verify a document nobody sealed.
_SEAL_PROBE = {"schema": "probe", "a": [1, 2, {"b": "c"}], "d": True}


def require_seal_agrees() -> None:
    """The driver's fixture seal and the bundle module's are the SAME function."""
    if driver.self_hash(_SEAL_PROBE) != odin_deployment_bundle.self_hash(_SEAL_PROBE):
        raise PackagingRefusal(
            "the shipped driver and the bundle-schema module compute different "
            "self-hashes over the same document; one of the two was edited "
            "alone, and a board node would verify evidence nobody sealed")
    payload = bytes(range(256)) * 3
    if (driver.encode_payload(payload)
            != odin_deployment_bundle.encode_payload(payload)):
        raise PackagingRefusal(
            "the shipped driver and the bundle-schema module encode a payload "
            "differently; the two halves of the package would not read each "
            "other's bytes")


def require_bundles_load() -> List[Path]:
    """Every shipped deployment bundle must seal, and must carry OUR kernel table."""
    bundles = sorted(DEPLOYMENT_SRC.glob("*.json")) if DEPLOYMENT_SRC.is_dir() else []
    if not bundles:
        raise PackagingRefusal(
            f"no deployment bundle under {DEPLOYMENT_SRC}; regenerate it with "
            f"scripts/hacc/make_deployment_bundle.py (it needs an RTL simulator, "
            f"which is why the bundle is committed rather than built here)")
    for path in bundles:
        document = require_sealed(path)
        if document.get("schema") == odin_deployment_bundle.SCHEMA \
                and document.get("kernel") != KERNEL_TABLE:
            raise PackagingRefusal(
                f"{path.name}: the bundle's device-protocol table is not this "
                f"package's; regenerate it against this kernel")
    return bundles


def require_sealed(path: Path) -> Dict[str, Any]:
    """One committed document, refusing unless it hashes to its own bytes."""
    if not path.is_file():
        raise PackagingRefusal(f"{path}: no such file")
    document = json.loads(path.read_text(encoding="utf-8"))
    stamped = {k: v for k, v in document.items() if k != "self_hash"}
    if odin_deployment_bundle.self_hash(stamped) != document.get("self_hash"):
        raise PackagingRefusal(
            f"{path.name}: its self-hash does not match its content; "
            f"regenerate it rather than editing it")
    return document


def require_bundle_document(path: Path) -> Dict[str, Any]:
    """One BUNDLE, refusing unless it seals AND speaks this package's protocol."""
    document = require_sealed(path)
    if document.get("schema") != odin_deployment_bundle.SCHEMA:
        raise PackagingRefusal(
            f"{path.name}: schema {document.get('schema')!r} is not "
            f"{odin_deployment_bundle.SCHEMA!r}; --deployment takes a sealed "
            f"deployment bundle, not its replay or some other document")
    if document.get("kernel") != KERNEL_TABLE:
        raise PackagingRefusal(
            f"{path.name}: the bundle's device-protocol table is not this "
            f"package's; regenerate it against this kernel")
    return document


def require_protocol_agrees() -> None:
    """The shipped driver and the repository SSOT declare the SAME protocol."""
    drifted = [
        (theirs, getattr(driver, mine), getattr(kernel_registers, theirs))
        for mine, theirs in _PROTOCOL_MIRRORS
        if getattr(driver, mine) != getattr(kernel_registers, theirs)
    ]
    if drifted:
        raise PackagingRefusal(
            f"the shipped driver and kernel_registers disagree about the device "
            f"protocol: {drifted} (name, driver, repo). One of the two was "
            f"edited alone; a package built from halves would drive a kernel "
            f"nobody built")


# ---------------------------------------------------------------------------
# Freezing one fixture
# ---------------------------------------------------------------------------


def _cache_key(name: str, plan_ops_digest: str, rtl: str) -> Path:
    key = hashlib.sha256(
        f"{name}|{plan_ops_digest}|{rtl}".encode("utf-8")).hexdigest()[:32]
    return CACHE / f"{name}_{key}.json"


def _cosim_counts(name, export, per_cycle_inputs, latencies, program, rtl, engine):
    """The RTL's per-cycle counts, cached by (fixture, program bytes, RTL)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = _cache_key(name, hashlib.sha256(program).hexdigest(), rtl)
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        print(f"[package] {name}: cosim cache hit ({path.name})")
        return (
            {tuple(int(v) for v in key.split(",")): int(value)
             for key, value in cached["counts"].items()},
            int(cached["device_cycles"]), int(cached["events"]),
            str(cached["engine"]))
    print(f"[package] {name}: running the committed cosimulation on {engine} ...")
    started = time.monotonic()
    result = run_cosim(export, per_cycle_inputs, latencies=latencies)
    print(f"[package] {name}: cosim done in {time.monotonic() - started:.1f}s "
          f"({len(result.capture.events)} events, "
          f"{result.capture.cycles} device cycles)")
    payload = {
        "counts": {",".join(str(v) for v in key): int(value)
                   for key, value in sorted(result.counts.items())},
        "device_cycles": int(result.capture.cycles),
        "events": len(result.capture.events),
        "engine": str(result.build.engine),
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return (dict(result.counts), int(result.capture.cycles),
            len(result.capture.events), str(result.build.engine))


class _FrozenCosim:
    """The cached cosimulation, wearing the shape the golden gates compare."""

    def __init__(self, counts, samples, cycles_per_sample):
        self.counts = counts
        self.samples = samples
        self.cycles_per_sample = cycles_per_sample

    def cycle_counts(self, sample, cycle, core, n_neurons):
        return tuple(self.counts.get((sample, cycle, core, neuron), 0)
                     for neuron in range(n_neurons))


def freeze(fixture: Fixture, *, head: str, dirty: bool, rtl: str,
           engine: str) -> Dict[str, Any]:
    """Build, cosimulate, gate, and seal one fixture."""
    mapping, rasters, length, chip_latency = fixture.build()
    export = export_of(mapping)
    reference = traces_for(
        mapping, rasters, simulation_length=length, chip_latency=chip_latency)
    per_cycle_inputs = [list(sample.per_cycle) for sample in reference]
    latencies = [int(v) for v in reference[0].trace.latencies]
    neurons = [int(core.neurons_per_core) for core in mapping.cores]

    program, stimulus, plan = split_payloads(
        export, per_cycle_inputs, latencies=latencies)
    counts, device_cycles, events, engine_used = _cosim_counts(
        fixture.name, export, per_cycle_inputs, latencies, program, rtl, engine)

    # --- GOLDEN GATE 1: the RTL equals the cycle-accurate twin, every cycle.
    frozen = _FrozenCosim(counts, plan.samples, plan.cycles_per_sample)
    differences = compare_cycle_counts(frozen, reference)
    if differences:
        raise PackagingRefusal(
            f"{fixture.name}: the cosimulation disagrees with the reference "
            f"twin at {len(differences)} point(s); the first is "
            f"{differences[0]}. No package is written from counts that do not "
            f"pass the gate they came from")

    # --- GOLDEN GATE 2: the windowed counts equal the twin's, per sample.
    measured_windows = [
        [list(row) for row in per_core]
        for per_core in _window_counts(
            counts, samples=plan.samples,
            cycles_per_sample=plan.cycles_per_sample, latencies=latencies,
            simulation_length=length, neurons=neurons)
    ]
    for index, sample in enumerate(reference):
        expected = [list(row) for row in sample.trace.window_counts()]
        if measured_windows[index] != expected:
            raise PackagingRefusal(
                f"{fixture.name}: sample {index} window counts "
                f"{measured_windows[index]} against the twin's {expected}")

    # --- GOLDEN GATE 3: the capture fits the shipped fabric.
    if events >= SHIPPED_CAPTURE_EVENTS:
        raise PackagingRefusal(
            f"{fixture.name}: {events} capture events against the shipped "
            f"fabric's {SHIPPED_CAPTURE_EVENTS}-record RAM; a board run would "
            f"refuse as truncated and the fixture would never certify")

    document: Dict[str, Any] = {
        "schema": driver.SCHEMA,
        "name": fixture.name,
        "title": fixture.title,
        "description": fixture.description,
        "source": fixture.notes,
        "provenance": {
            "generating_commit": head,
            "worktree_dirty": dirty,
            "rtl_sha256": rtl,
            "cosim_engine": engine_used,
            "generator": "scripts/hacc/make_package.py",
        },
        "kernel": dict(KERNEL_TABLE),
        "program": driver.encode_payload(program),
        "stimulus": driver.encode_payload(stimulus),
        "run": {
            "cores": int(plan.n_cores),
            "samples": int(plan.samples),
            "cycles_per_sample": int(plan.cycles_per_sample),
            "first_tag": int(plan.tag_of(0, 0)),
            "barrier_cycles": int(plan.barrier_cycles),
            "latencies": latencies,
            "neurons": neurons,
            "simulation_length": int(length),
            "capture_events_needed": int(events),
            "program_words_needed": (
                driver.stimulus_base_word(len(program) // WORD_BYTES)
                + len(stimulus) // WORD_BYTES),
        },
        "expected": {
            "per_cycle": [
                [int(sample), int(cycle), int(core), int(neuron), int(value)]
                for (sample, cycle, core, neuron), value in sorted(counts.items())
                if value
            ],
            "window": measured_windows,
            "device_cycles_cosim": int(device_cycles),
        },
    }
    # --- GOLDEN GATE 4: the token stream fits the program RAM of a kernel
    # built with this fixture's core count (PROG_WORDS = NC * 262144).
    capacity = plan.n_cores * driver.PROG_WORDS_PER_CORE
    if document["run"]["program_words_needed"] > capacity:
        raise PackagingRefusal(
            f"{fixture.name}: the run needs "
            f"{document['run']['program_words_needed']} program words but an "
            f"NC={plan.n_cores} kernel's program RAM holds {capacity}; the "
            f"board would refuse it and the fixture could never certify")

    document["witnesses"] = fixture.witnesses(
        document, (mapping, rasters, length, chip_latency, reference))

    # --- GOLDEN GATE 5: the frozen per-cycle table refolds to the frozen
    # windows under the DRIVER's own window rule, not this script's.
    refolded = driver.window_counts(
        driver.per_cycle_table(document["expected"]["per_cycle"]),
        document["run"])
    if refolded != measured_windows:
        raise PackagingRefusal(
            f"{fixture.name}: the shipped per-cycle table does not refold to "
            f"the shipped windows under the driver's rule; the two halves of "
            f"the evidence disagree")
    return driver.seal(document)


def _window_counts(counts, *, samples, cycles_per_sample, latencies,
                   simulation_length, neurons):
    from mimarsinan.chip_simulation.odin_rtl.cosim import window_counts_of

    return window_counts_of(
        counts, samples=samples, cycles_per_sample=cycles_per_sample,
        latencies=latencies, simulation_length=simulation_length,
        neurons=neurons)


# ---------------------------------------------------------------------------
# The tree
# ---------------------------------------------------------------------------


def hw_payload() -> List[Tuple[Path, str]]:
    """Exactly the RTL ``build_xclbn.sh`` compiles, and its vendoring paperwork."""
    entries: List[Tuple[Path, str]] = []
    for pattern in ("hw/fpga/kernel/*.v", "hw/fpga/mem/*.v"):
        for path in sorted(REPO.glob(pattern)):
            entries.append((path, str(path.relative_to(REPO))))
    vendor = REPO / "hw" / "vendor" / "odin"
    for path in sorted(vendor.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in (".v", ".md", ".sha256") and path.name != "LICENSE":
            continue
        entries.append((path, str(path.relative_to(REPO))))
    return entries


def kernel_xml_text() -> str:
    """``kernel.xml`` emitted from the register SSOT, so the build needs no src/."""
    generator = REPO / "scripts" / "hacc" / "gen_kernel_xml.py"
    output = DIST / "_kernel.xml"
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(generator), "--output", str(output),
         "--kernel", KERNEL_NAME],
        check=True, capture_output=True, text=True, cwd=str(REPO))
    text = output.read_text(encoding="utf-8")
    output.unlink()
    return text


def stage_deployment(bundles: Sequence[Path], copy, write_text) -> None:
    """Stage exported bundles + the index that names the DEFAULT one.

    The executor's own default stays the committed witness bundle, so a
    bring-up package is unchanged; a DEPLOYMENT package additionally declares
    which network phase 8 should run, and run_all.sh reads exactly this file.
    """
    staged: List[Dict[str, Any]] = []
    for path in bundles:
        document = require_bundle_document(path)
        copy(path, f"deployment/{path.name}")
        capture = path.with_name(f"{path.stem}_capture.json")
        if not capture.is_file():
            raise PackagingRefusal(
                f"{path.name}: no {capture.name} beside it. The replay is what "
                f"lets the node's own selftest run the bundle with no card; a "
                f"deployment package without it can only be verified on silicon")
        copy(capture, f"deployment/{capture.name}")
        staged.append({
            "bundle": f"deployment/{path.name}",
            "replay": f"deployment/{capture.name}",
            "name": document["name"],
            "self_hash": document["self_hash"],
            "samples": len(document["samples"]),
            "cores": len(document["cores"]),
            "pass_order": list(document["pass_order"]),
            "cycles_per_sample": int(document["cycles_per_sample"]),
            "certification_samples": len(document["certification"]["samples"]),
            "provenance": document["provenance"],
            "model": document["model"],
        })
    write_text(DEPLOYMENT_INDEX, json.dumps({
        "schema": "odin_hacc_deployment_index/1",
        "default": staged[0]["bundle"],
        "default_replay": staged[0]["replay"],
        "bundles": staged,
        "note": (
            "run_all.sh phase 8 runs 'default' unless ODIN_BUNDLE names another "
            "file. ODIN_DEPLOY_SAMPLES bounds the campaign on the node; the "
            "bundle can only execute the samples it ships, so raising it past "
            "'samples' runs every shipped sample and no more."),
    }, indent=2, sort_keys=True) + "\n")


def build_tree(documents: Sequence[Dict[str, Any]], *, head: str, dirty: bool,
               rtl: str, deployments: Sequence[Path] = ()) -> Dict[str, Any]:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    written: List[str] = []

    def write_text(relative: str, text: str) -> None:
        target = STAGE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        written.append(relative)

    def copy(source: Path, relative: str) -> None:
        target = STAGE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        written.append(relative)

    for source, relative in hw_payload():
        copy(source, relative)
    for name in ("build_xclbn.sh", "cards.sh", "odin_u55c.cfg", "odin_u250.cfg",
                 "toolchain.sh", "mine_checkpoint.sh"):
        copy(REPO / "scripts" / "hacc" / name, f"scripts/hacc/{name}")
    frozen_xml = kernel_xml_text()
    write_text("scripts/hacc/kernel.xml", frozen_xml)
    for source in sorted((PACKAGE_SRC / "sbatch").glob("*.sbatch")):
        copy(source, f"scripts/hacc/{source.name}")
    # The stand-in verifies the frozen kernel.xml before handing it to a build;
    # its expected digest is injected here, at the only moment both exist.
    stand_in = (PACKAGE_SRC / "gen_kernel_xml.py").read_text()
    xml_digest = hashlib.sha256(frozen_xml.encode()).hexdigest()
    assert stand_in.count("__ODIN_KERNEL_XML_SHA256__") == 1
    write_text("scripts/hacc/gen_kernel_xml.py",
               stand_in.replace("__ODIN_KERNEL_XML_SHA256__", xml_digest))
    copy(PACKAGE_SRC / "scripts" / "status.sh", "scripts/status.sh")
    copy(PACKAGE_SRC / "scripts" / "chip_cache.sh", "scripts/chip_cache.sh")
    # The in-zip bootstrap keeps its placeholder hash: no file can carry the
    # digest of an archive that contains that same file. The STANDALONE copy
    # written beside the zip afterwards is the one that verifies.
    copy(PACKAGE_SRC / "bootstrap_hacc.sh", "bootstrap_hacc.sh")
    for name in ("run_all.sh", "collect_results.sh", "README_HACC.md"):
        copy(PACKAGE_SRC / name, name)
    copy(PACKAGE_SRC / "host" / "odin_board_driver.py",
         "host/odin_board_driver.py")
    copy(PACKAGE_SRC / "host" / "fake_pyxrt_for_selftest.py",
         "host/fake_pyxrt_for_selftest.py")
    copy(PACKAGE_SRC / "host" / "odin_deployment_executor.py",
         "host/odin_deployment_executor.py")
    copy(PACKAGE_SRC / "host" / "render_die_map.py", "host/render_die_map.py")
    # VERBATIM, and hash-verified below: the board executor imports the very
    # module the repository's cycle-accurate twin executes.
    copy(BUNDLE_MODULE, "host/odin_deployment_bundle.py")
    if sha256_file(STAGE / "host/odin_deployment_bundle.py") != sha256_file(BUNDLE_MODULE):
        raise PackagingRefusal(
            "the staged bundle-schema module is not byte-identical to "
            f"{BUNDLE_MODULE.relative_to(REPO)}")
    for path in require_bundles_load():
        copy(path, f"deployment/{path.name}")
    if deployments:
        stage_deployment(deployments, copy, write_text)

    for document in documents:
        write_text(
            f"fixtures/{document['name']}.json",
            json.dumps(document, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True) + "\n")
    write_text("fixtures/INDEX.json", json.dumps({
        "fixtures": [
            {"name": d["name"], "title": d["title"], "cores": d["run"]["cores"],
             "samples": d["run"]["samples"], "self_hash": d["self_hash"],
             "witnesses": d["witnesses"]}
            for d in documents
        ],
    }, indent=2, sort_keys=True) + "\n")

    manifest = {
        "package": "odin_hacc_package",
        "generating_commit": head,
        "worktree_dirty": dirty,
        "rtl_sha256": rtl,
        "kernel": dict(KERNEL_TABLE),
        "files": {
            relative: sha256_file(STAGE / relative)
            for relative in sorted(written)
        },
    }
    write_text("MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


EXECUTABLE = (".sh", ".py")

#: The line the standalone bootstrap's digest is injected into. Exactly one, and
#: packaging refuses if the file ever grows a second.
BOOTSTRAP_TOKEN = 'ZIP_SHA256="__ODIN_ZIP_SHA256__"'
BOOTSTRAP_STANDALONE = DIST / "bootstrap_hacc.sh"


def write_zip(zip_path: Path = ZIP_PATH) -> None:
    DIST.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    names = sorted(
        str(path.relative_to(STAGE))
        for path in STAGE.rglob("*") if path.is_file())
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in names:
            info = zipfile.ZipInfo(f"odin_hacc_package/{name}", date_time=ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            mode = 0o755 if name.endswith(EXECUTABLE) else 0o644
            info.external_attr = (mode << 16) | 0o600
            archive.writestr(info, (STAGE / name).read_bytes())


def write_standalone_bootstrap(zip_sha256: str, *, zip_path: Path = ZIP_PATH,
                              target: Path = BOOTSTRAP_STANDALONE) -> Path:
    """The bootstrap that travels BESIDE the zip, carrying the zip's digest.

    Order is the whole trick: the archive is written first, hashed second, and
    this file third and OUTSIDE it, so the embedded value always describes the
    final bytes of the shipped zip.
    """
    source = PACKAGE_SRC / "bootstrap_hacc.sh"
    text = source.read_text(encoding="utf-8")
    if text.count(BOOTSTRAP_TOKEN) != 1:
        raise PackagingRefusal(
            f"{source} carries {text.count(BOOTSTRAP_TOKEN)} copies of "
            f"{BOOTSTRAP_TOKEN!r}; the injection needs exactly one, or the "
            f"shipped bootstrap would verify against the wrong value")
    injected = text.replace(BOOTSTRAP_TOKEN, f'ZIP_SHA256="{zip_sha256}"')
    target.write_text(injected, encoding="utf-8")
    target.chmod(0o755)
    if zip_sha256 not in target.read_text(encoding="utf-8"):
        raise PackagingRefusal("the standalone bootstrap lost its digest")
    if sha256_file(zip_path) != zip_sha256:
        raise PackagingRefusal(
            "the zip changed after it was hashed; the embedded digest would "
            "refuse the very package it ships with")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-zip", action="store_true",
                        help="build the tree and the fixtures, skip the archive")
    parser.add_argument("--only", action="append", default=[],
                        help="freeze only this fixture (repeatable, for triage)")
    parser.add_argument(
        "--deployment", action="append", default=[], metavar="BUNDLE",
        help="DEPLOYMENT MODE: also ship this exported bundle (and its "
             "_capture.json sibling) and write dist/odin_hacc_deployment.zip. "
             "Repeatable; the FIRST is what phase 8 runs by default.")
    options = parser.parse_args()
    deployments = [Path(p).resolve() for p in options.deployment]

    require_protocol_agrees()
    require_seal_agrees()
    require_bundles_load()
    engine = available_engine()
    head, dirty = git_head()
    rtl = rtl_digest()
    print(f"[package] commit {head}{' (dirty)' if dirty else ''}")
    print(f"[package] rtl    {rtl}")
    print(f"[package] engine {engine}")

    chosen = [f for f in FIXTURES if not options.only or f.name in options.only]
    if not chosen:
        raise PackagingRefusal(f"--only matched no fixture; have "
                               f"{[f.name for f in FIXTURES]}")
    documents = [
        freeze(fixture, head=head, dirty=dirty, rtl=rtl, engine=engine)
        for fixture in chosen
    ]
    for document in documents:
        run = document["run"]
        print(f"[package] froze {document['name']}: cores={run['cores']} "
              f"samples={run['samples']} program={document['program']['words']}w "
              f"stimulus={document['stimulus']['words']}w "
              f"events={run['capture_events_needed']} "
              f"witnesses={sorted(document['witnesses'])}")

    manifest = build_tree(
        documents, head=head, dirty=dirty, rtl=rtl, deployments=deployments)
    print(f"[package] staged {len(manifest['files'])} files under {STAGE}")
    for path in deployments:
        print(f"[package] deploying {path.name} "
              f"({sha256_file(path)[:16]}\u2026)")
    if options.no_zip:
        return 0
    zip_path = DEPLOYMENT_ZIP_PATH if deployments else ZIP_PATH
    bootstrap = DEPLOYMENT_BOOTSTRAP if deployments else BOOTSTRAP_STANDALONE
    write_zip(zip_path)
    size = zip_path.stat().st_size
    zip_sha256 = sha256_file(zip_path)
    print(f"[package] wrote {zip_path} ({size / 1e6:.2f} MB, "
          f"sha256 {zip_sha256})")
    (DIST / f"{zip_path.name}.sha256").write_text(
        f"{zip_sha256}  {zip_path.name}\n", encoding="utf-8")
    written = write_standalone_bootstrap(
        zip_sha256, zip_path=zip_path, target=bootstrap)
    print(f"[package] wrote {written} "
          f"(verifies that digest before it unpacks anything)")
    # The DEPLOYMENT package carries a whole network's per-core programs, so it
    # is allowed to be larger than the bring-up upload the owner was promised;
    # the ceiling still exists, because an upload nobody can complete is not a
    # deliverable either.
    ceiling = 200 if deployments else 20
    if size > ceiling * 1000 * 1000:
        raise PackagingRefusal(
            f"the package is {size / 1e6:.1f} MB, over the {ceiling} MB a "
            f"single upload was promised")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
