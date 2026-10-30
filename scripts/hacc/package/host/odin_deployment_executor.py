#!/usr/bin/env python3
"""HACC NUS - ODIN Deployment: one sealed bundle, one card, one measured verdict.

THE SHAPE OF THE PROBLEM. A bitstream instantiates NC ODIN cores behind one
sequencer (NC is a build parameter, ODIN_KERNEL_CORES; the archived bitstreams
are NC = 1), and the chip routes nothing between cores (SPI_OPEN_LOOP). A
multi-core network therefore runs as one host-mediated PASS per core: program
the core, run a sample, read its per-cycle counts back, TRANSCODE them through
the axon source table into the next core's slot counts, build that core's
stimulus on the host, run again. Fabric memories persist across sequencer runs,
so a core is programmed ONCE and stimulated per sample; the per-sample membrane
CLEAR is an op inside each stimulus, not a reprogram.

CORE SELECTION. Every pass of a bundle addresses ``bundle.PASS_CORE`` (its own
one-core mapping). The executor PLACES pass k on fabric core (position of k in
the pass order) mod NC, NC being the session's declared cores, and retargets
the pass's program and stimuli onto it (``driver.retarget_payload``); each
capture is decoded back onto the pass, and a record from any other fabric core
refuses the pass. At NC >= passes every core of the network is resident on its
own fabric core and spikes between them cross at the segment boundaries through
the host; at NC = 1 every pass runs on core 0, byte for byte the archived run.

WHAT IT PROVES, AND WHAT IT DOES NOT. It re-derives nothing: the expected counts
were frozen from the committed RTL cosimulation by
scripts/hacc/make_deployment_bundle.py, one pass at a time, gated cycle by cycle
against the cycle-accurate twin. This file compares, accumulates and TIMES. A
red certificate here is the card or the transcode, never a tolerance.

THE TRANSCODE IS NOT REIMPLEMENTED HERE. It lives in odin_deployment_bundle.py,
the module the repository's own cycle-accurate twin executes
(chip_simulation/odin_rtl/reference.py), shipped verbatim beside this file. Nor
is the XRT session: this drives odin_board_driver.py's BoardSession, the same
one B0/B1 drive, with the buffer objects allocated once per core and rewritten.

TWO EXECUTION ARMS (``--arm``, ``$ODIN_ARM``). ``as_run`` is the archived
behaviour: the core's buffers are written once, but every pass declares the
whole program, so the wrapper re-streams it (and the stock core re-shifts it
over SPI) before each sample. ``resident`` programs each core ONCE per campaign
with one run of the whole program and an END-only stimulus, then runs every
sample with ``program_words = 1``, which the wrapper streams no program word
for (``odin_fpga_kernel_top.v``, the ``prog_stream_w`` split). The core keeps
its configuration across runs: ``ap_start`` re-arms the capture, the FIFO and
the cycle counter, never the core's own reset. Resident counts are certified
pass by pass exactly like as-run ones, and a card that forgot its program
between runs fails those certificates rather than a tolerance.

A TIMED LOOP (``--loop-s``, and ``timed_loop`` for the power protocol) repeats
whole certified campaigns back to back until the time is spent, so a power
window holds only complete inferences.

N COMPUTE UNITS AT ONCE (``--compute-units N``, route D). An xclbin linked
with N units of the kernel (``scripts/hacc/gen_link_cfg.py``) runs the SAME
certified campaign on every unit concurrently, one process per unit, each
holding its unit exclusively (``odin_fpga_kernel_top:{odin_<k>}``) and its own
buffers on its own bank. The parent loads the xclbin once, starts the units one
at a time (no two loads race), releases them together, and writes one report
that keeps every unit's certificates and readouts apart, each row stamped with
its unit; every unit's own report sits under ``cu<k>/``. N = 1 (the default) is
the archived single-session run, unchanged.

A TRANSFER CONTROL (``TransferControl`` and ``transfer_loop``, the power
protocol's ``transfer_<arm>`` condition) replays the host-card transfers of a
certified campaign of an arm: the same buffer allocations and program DMA per
core, then per sample the same stimulus write, capture poison and capture
readback, in the same order and sizes, each pass paced by the wall that pass
took when it was certified. It never calls the kernel, so ``ap_start`` is never
raised. At the card boundary it isolates the PCIe and HBM activity of the
arm's transfers from the kernel's own run.

    python3 host/odin_deployment_executor.py --xclbin <path> \\
        --bundle deployment/nc1_two_core_passes.json --results results/deploy
    ODIN_DEPLOY_SAMPLES=8 ...   # bound the campaign; the default is every
                                # shipped sample, capped at 64
    ODIN_ARM=resident ...       # program once per core, then stimulate
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import odin_board_driver as driver
    import odin_deployment_bundle as bundle
    import odin_deployment_encoding as encoding
except ImportError as _exc:  # pragma: no cover - a package missing its own halves
    raise SystemExit(
        f"REFUSING: {_exc}. odin_deployment_executor.py runs beside "
        f"odin_board_driver.py, odin_deployment_bundle.py and "
        f"odin_deployment_encoding.py in the package's host/ directory; "
        f"re-unzip the package rather than moving one half")

#: The default campaign size, and the env var that bounds it. Every shipped
#: sample is certified against its frozen readout, but a bundle carrying
#: thousands would spend a board hour before anyone saw a number.
SAMPLES_ENV = "ODIN_DEPLOY_SAMPLES"
DEFAULT_SAMPLE_CAP = 64

#: How many per-sample rows the TSV carries. The aggregates are over everything;
#: the rows are for reading, and an unbounded TSV is not read.
TSV_ROW_CAP = 512

CAPTURE_SCHEMA = "odin_hacc_deployment_capture/1"

REPORT_NAME = "deployment_report.json"
TSV_NAME = "deployment_samples.tsv"

#: The execution arms, and the env var that picks one. ``as_run`` is the default
#: and is the archived behaviour byte for byte.
ARM_AS_RUN = "as_run"
ARM_RESIDENT = "resident"
ARMS = (ARM_AS_RUN, ARM_RESIDENT)
ARM_ENV = "ODIN_ARM"

#: What a resident pass declares as its program: one word, which the wrapper
#: treats as "no program to stream" (its END would be overwritten by the
#: stimulus anyway, ``stimulus_base_word``).
RESIDENT_PROGRAM_WORDS = 1

#: N compute units (route D): the env var a board job may set, and how long a
#: unit may take to open its session before the parent refuses the campaign.
UNITS_ENV = "ODIN_COMPUTE_UNITS"
UNIT_READY_TIMEOUT_S = 300.0
UNIT_START_TIMEOUT_S = 600.0

#: What a window of N units sums: every count and wall the loop and the
#: transfer control report. Duty is averaged (each unit's own share of its own
#: wall) and also summed; the window spans the earliest start to the latest end.
UNIT_SUM_KEYS = (
    "campaigns", "inferences", "passes", "run_s", "device_cycles",
    "programming_runs", "programming_run_s", "programming_device_cycles",
    "certificates", "certificates_failed", "readouts_drifted",
    "programming_replays", "ap_starts", "paced_s", "bytes_to_device",
    "bytes_from_device",
)


class OdinDeploymentRefusal(driver.OdinDriverError):
    """The deployment cannot proceed as configured; the message says why."""


class OdinBundleCorrupt(driver.OdinFixtureCorrupt):
    """A bundle's self-hash or payload hash does not match its own bytes."""


class OdinTranscodeDiverged(driver.OdinDriverError):
    """A live-built stimulus is not the one the frozen evidence was built from."""


def load_bundle(path: str) -> Dict[str, Any]:
    """The sealed bundle, refusing in the fixture family's own corrupt class."""
    try:
        return bundle.load_bundle(path)
    except bundle.OdinBundleCorrupt as exc:
        raise OdinBundleCorrupt(str(exc)) from exc


def decode(entry: Dict[str, Any], *, what: str) -> bytes:
    """ONE decode point, so a damaged payload refuses in the driver's own family."""
    try:
        return bundle.decode_payload(entry, what=what)
    except bundle.OdinBundleCorrupt as exc:
        raise OdinBundleCorrupt(str(exc)) from exc


def require_protocol_agrees(document: Dict[str, Any], *, source: str) -> None:
    """The bundle's device-protocol table must be this driver's."""
    if document.get("kernel") != driver.KERNEL_TABLE:
        raise OdinBundleCorrupt(
            f"{source}: the bundle's device-protocol table disagrees with this "
            f"driver's. Bundle={document.get('kernel')}, "
            f"driver={driver.KERNEL_TABLE}. The bundle was assembled against a "
            f"different kernel — rebuild it with "
            f"scripts/hacc/make_deployment_bundle.py rather than mixing halves")


def require_fabric_matches(document: Dict[str, Any], capacity: Any, *,
                           source: str) -> Any:
    """The FABRIC this session declares must be the one the bundle was mapped on.

    A bundle carries no chip name — it carries the claims that identify one, and
    the axon-slot count those claims imply is what its stimulus addresses. A
    stock bundle on a wide bitstream (or the reverse) would word every event for
    the wrong crossbar and answer with a number nobody could tell was wrong.
    """
    # A bundle sealed before one of these claims existed is refused by name here (OdinBundleStale), not by a KeyError.
    bundle.require_fabric_claims(document["chip_config"], source=source)
    aer = encoding.encoding_of_bundle(document)
    fabric = driver.chip_config_named(capacity.chip)
    mapped = str(document["chip_config"]["weight_sign_granularity"])
    if mapped != str(fabric["weight_sign_granularity"]):
        raise OdinDeploymentRefusal(
            f"{source}: this bundle was mapped against a "
            f"{mapped!r}-signed crossbar (AER wording {aer.name}), and chip "
            f"{capacity.chip!r} signs {fabric['weight_sign_granularity']!r}. "
            f"The two word an axon event differently and neither refuses the "
            f"other's stream — pass --chip for the fabric this bitstream holds")
    slots = int(document["chip_config"]["effective_max_axons"]) + 1
    declared = int(capacity.axon_slots_per_core)
    if slots > declared:
        raise OdinDeploymentRefusal(
            f"{source}: the bundle's claims need {slots} axon slot(s) per core "
            f"and chip {capacity.chip!r} declares {declared}; every slot above "
            f"the crossbar's width would be delivered to a row that is not "
            f"there")
    # [U-H, survey gap B6.1] the neurons per core the bundle's fabric declares.
    neurons = int(document["chip_config"]["max_neurons"])
    if neurons > int(capacity.neurons_per_core):
        raise OdinDeploymentRefusal(
            f"{source}: the bundle was mapped on a fabric of {neurons} neuron(s) "
            f"per core and chip {capacity.chip!r} declares "
            f"{capacity.neurons_per_core}; its passes would address neurons the "
            f"crossbar does not have")
    return aer


def verify_replay(replay: Dict[str, Any], document: Dict[str, Any], *,
                  source: str) -> None:
    """The replay must be sealed like a fixture, and must be THIS bundle's."""
    if replay.get("schema") != CAPTURE_SCHEMA:
        raise OdinBundleCorrupt(
            f"{source}: schema {replay.get('schema')!r} is not "
            f"{CAPTURE_SCHEMA!r}; this is not a replay document")
    stamped = {key: value for key, value in replay.items() if key != "self_hash"}
    if bundle.self_hash(stamped) != replay.get("self_hash"):
        raise OdinBundleCorrupt(
            f"{source}: self-hash {replay.get('self_hash')} but the content "
            f"hashes to {bundle.self_hash(stamped)}; a replay is frozen "
            f"evidence and is not a file to edit")
    if replay.get("bundle_self_hash") != document["self_hash"]:
        raise OdinBundleCorrupt(
            f"{source}: this replay was frozen for bundle "
            f"{replay.get('bundle_self_hash')}, not {document['self_hash']}. "
            f"Replaying one bundle's answers against another's stimuli would "
            f"certify a run nobody made")


def require_pass_order_is_causal(document: Dict[str, Any]) -> None:
    """Every route must name a core that has ALREADY run in the pass order."""
    seen: List[int] = []
    for plan in bundle.pass_plans(document):
        for slot, (kind, _index) in enumerate(plan["routes"]):
            if int(kind) < 0:
                continue
            if int(kind) not in seen:
                raise OdinDeploymentRefusal(
                    f"pass order {document['pass_order']} runs core "
                    f"{plan['core']} before core {kind}, which its slot {slot} "
                    f"reads. A consumer whose producer has not run would be "
                    f"stimulated with counts nobody measured")
        seen.append(int(plan["core"]))


def sample_indices(document: Dict[str, Any], requested: int | None) -> List[int]:
    """Which samples this campaign runs, and the honest reason for the bound."""
    shipped = [int(entry["index"]) for entry in document["samples"]]
    bound = requested if requested is not None else DEFAULT_SAMPLE_CAP
    if bound <= 0:
        raise OdinDeploymentRefusal(
            f"{SAMPLES_ENV}={bound} asks for no samples at all; a campaign that "
            f"runs nothing is not a result")
    return shipped[:bound]


# ---------------------------------------------------------------------------
# One core, resident on the fabric
# ---------------------------------------------------------------------------


class ResidentPass:
    """One core programmed ONCE, then stimulated per sample on reused buffers."""

    def __init__(self, session: Any, plan: Dict[str, Any], *,
                 capture_events: int, arm: str = ARM_AS_RUN,
                 fabric_core: int = bundle.PASS_CORE) -> None:
        if arm not in ARMS:
            raise OdinDeploymentRefusal(
                f"arm {arm!r} is not one of {ARMS}")
        self.session = session
        self.plan = plan
        self.arm = arm
        self.core = int(plan["core"])
        #: The fabric core this pass is placed on; its streams are retargeted
        #: from the bundle's PASS_CORE onto it, and its captures decoded back.
        self.fabric_core = int(fabric_core)
        self.placement = {bundle.PASS_CORE: self.fabric_core}
        self.neurons = int(plan["neurons"])
        self.latency = int(plan["latency"])
        self.capture_events = int(capture_events)
        self.capture_bytes = driver.capture_buffer_bytes(self.capture_events)
        self.program = self.placed(
            decode(plan["program"], what=f"core {self.core}/program"),
            what=f"core {self.core}/program")
        self.program_words = len(self.program) // driver.WORD_BYTES
        self.stimulus_capacity = int(plan["max_stimulus_words"]) * driver.WORD_BYTES
        self._program_bo: Any = None
        self._stimulus_bo: Any = None
        self._capture_bo: Any = None

    def placed(self, payload: bytes, *, what: str) -> bytes:
        """``payload`` (addressing PASS_CORE) moved onto this pass's fabric core."""
        return driver.retarget_payload(payload, self.placement, what=what)

    def decode(self, words: Sequence[int]) -> Tuple[List[Tuple[int, int, int, int]], int]:
        """This pass's capture, back on PASS_CORE; another core's record refuses."""
        return decode_pass_capture(
            words, capture_events=self.capture_events,
            fabric_core=self.fabric_core, transport=self.session.name)

    def program_once(self) -> Dict[str, Any]:
        """Allocate this core's three buffers and stream its program in.

        This IS the segment boundary: everything between the previous core's
        last sample and this core's first one. The fabric holds no copy of the
        op stream, so the wall below is a HOST-LINK cost that recurs at every
        boundary of a multi-pass deployment — measured, never budgeted.
        """
        started = time.perf_counter()
        self._program_bo = self.session.allocate(
            len(self.program), driver.ARG_PROGRAM)
        self._stimulus_bo = self.session.allocate(
            self.stimulus_capacity, driver.ARG_STIMULUS)
        self._capture_bo = self.session.allocate(
            self.capture_bytes, driver.ARG_CAPTURE)
        allocated = time.perf_counter()
        walls = self.session.dma_in(self._program_bo, self.program)
        boundary_s = time.perf_counter() - started
        row: Dict[str, Any] = {
            "core": self.core,
            "fabric_core": self.fabric_core,
            "program_bytes": len(self.program),
            "program_words": self.program_words,
            "allocate_s": allocated - started,
            "program_bo_write_s": walls["bo_write_s"],
            "program_sync_s": walls["sync_s"],
            "core_program_s": boundary_s,
            "segment_boundary_init_s": boundary_s,
        }
        # The stream's own wall — the DMA alone, without the allocation the
        # boundary also pays — is what a BANDWIDTH may honestly be taken over.
        row.update(driver.stream_metrics(
            len(self.program), walls["bo_write_s"] + walls["sync_s"]))
        if self.arm == ARM_RESIDENT:
            programming = self.programming_run()
            row["programming_run"] = programming
            row["core_program_s"] = time.perf_counter() - started
            row["segment_boundary_init_s"] = row["core_program_s"]
        return row

    def programming_run(self) -> Dict[str, Any]:
        """Resident arm: run the whole program once, with an END-only stimulus.

        The wrapper streams the program without its END, then the one-word
        stimulus, which IS an END: the sequencer configures the core and
        stops. Nothing is stimulated, so the capture should come back empty;
        its event count is recorded rather than refused, because the per-pass
        certificates that follow are what decide whether the core still holds
        the network.
        """
        terminator = struct.pack("<I", driver.OP_END)
        write = self.session.dma_in(self._stimulus_bo, terminator)
        poison = self.session.poison_capture(self._capture_bo)
        run_s = self.session.start_and_wait(
            self._program_bo, self._stimulus_bo, self._capture_bo,
            program_words=self.program_words, stimulus_words=1,
            capture_events=self.capture_events)
        words, back = self.session.read_capture(
            self._capture_bo, self.capture_bytes)
        events, device_cycles = self.decode(words)
        return {
            "program_words": self.program_words, "stimulus_words": 1,
            "run_s": run_s, "device_cycles": int(device_cycles),
            "capture_events": len(events),
            "bo_write_s": write["bo_write_s"] + poison["bo_write_s"],
            "sync_s": write["sync_s"] + poison["sync_s"] + back["sync_s"],
            "readback_s": back["readback_s"],
        }

    @property
    def declared_program_words(self) -> int:
        """What each sample's run declares: the whole program, or one word."""
        return (self.program_words if self.arm == ARM_AS_RUN
                else RESIDENT_PROGRAM_WORDS)

    def run_stimulus(self, stimulus: bytes) -> Tuple[List[int], Dict[str, float]]:
        """One sample on the resident core: rewrite, poison, start, read back.

        ``stimulus`` addresses PASS_CORE, as the bundle froze it; it is placed
        on this pass's fabric core here, so the frozen-stimulus check upstream
        compares the bundle's own bytes.
        """
        if self._program_bo is None:
            raise OdinDeploymentRefusal(
                f"core {self.core}: a stimulus before program_once() — the "
                f"crossbar holds no weights, so the counts would be another "
                f"network's")
        if len(stimulus) > self.stimulus_capacity:
            raise OdinTranscodeDiverged(
                f"core {self.core}: the live stimulus is {len(stimulus)} bytes "
                f"against the bundle's declared ceiling of "
                f"{self.stimulus_capacity}. The transcode produced more events "
                f"than the frozen evidence ever did, so this run is not the "
                f"deployment the bundle describes")
        stimulus = self.placed(stimulus, what=f"core {self.core}/stimulus")
        stimulus_words = len(stimulus) // driver.WORD_BYTES
        write = self.session.dma_in(self._stimulus_bo, stimulus)
        poison = self.session.poison_capture(self._capture_bo)
        run_s = self.session.start_and_wait(
            self._program_bo, self._stimulus_bo, self._capture_bo,
            program_words=self.declared_program_words,
            stimulus_words=stimulus_words,
            capture_events=self.capture_events)
        words, back = self.session.read_capture(
            self._capture_bo, self.capture_bytes)
        return words, {
            "bo_write_s": write["bo_write_s"] + poison["bo_write_s"],
            "sync_s": write["sync_s"] + poison["sync_s"] + back["sync_s"],
            "run_s": run_s,
            "readback_s": back["readback_s"],
            "stimulus_words": stimulus_words,
        }

    def transfers_only(self, stimulus: bytes, pause_s: float) -> Dict[str, Any]:
        """``run_stimulus`` without ``ap_start``: the transfer control's pass.

        The same three transfers as a sample (stimulus write and sync, capture
        poison, capture sync and read back), with a pause standing for the
        run, and no kernel call at all.
        """
        if self._program_bo is None:
            raise OdinDeploymentRefusal(
                f"core {self.core}: a control pass before program_once(); the "
                f"control replays the transfers of a programmed core")
        stimulus = self.placed(stimulus, what=f"core {self.core}/stimulus")
        write = self.session.dma_in(self._stimulus_bo, stimulus)
        poison = self.session.poison_capture(self._capture_bo)
        if pause_s > 0.0:
            time.sleep(float(pause_s))
        _words, back = self.session.read_capture(
            self._capture_bo, self.capture_bytes)
        return {
            "bo_write_s": write["bo_write_s"] + poison["bo_write_s"],
            "sync_s": write["sync_s"] + poison["sync_s"] + back["sync_s"],
            "readback_s": back["readback_s"], "paused_s": max(0.0, float(pause_s)),
            "bytes_to_device": len(stimulus) + len(driver.no_verdict_header()),
            "bytes_from_device": self.capture_bytes,
        }

    def release(self) -> None:
        self._program_bo = self._stimulus_bo = self._capture_bo = None


# ---------------------------------------------------------------------------
# The campaign
# ---------------------------------------------------------------------------


def pass_placement(document: Dict[str, Any], fabric_cores: int) -> Dict[int, int]:
    """Which fabric core each pass runs on: its position in the pass order, mod NC.

    At NC >= passes every pass has a fabric core of its own (all resident); at
    NC = 1 every pass runs on core 0, where the archived runs ran.
    """
    cores = int(fabric_cores)
    if cores < 1:
        raise OdinDeploymentRefusal(
            f"a session declaring {cores} core(s) has no fabric core to place "
            f"a pass on")
    return {int(core): position % cores
            for position, core in enumerate(document["pass_order"])}


def decode_pass_capture(words: Sequence[int], *, capture_events: int, fabric_core: int,
                        transport: str) -> Tuple[List[Tuple[int, int, int, int]], int]:
    """One pass's capture decoded, its records moved back onto PASS_CORE.

    Only the pass's own fabric core may have spiked: every other core is idle
    (unstimulated) during the pass, so a record from one is refused rather than
    folded into, or dropped from, the pass being certified.
    """
    events, device_cycles = driver.decode_capture(
        words, int(capture_events), transport=transport)
    placed, _per_core = driver.logical_events(
        events, {bundle.PASS_CORE: int(fabric_core)}, transport=transport)
    return placed, device_cycles


def _zero_rows(document: Dict[str, Any]) -> List[Tuple[int, ...]]:
    """An empty per-core output row, so a read of a core that has not run refuses."""
    return [() for _ in document["cores"]]


def transcode_slots(
    document: Dict[str, Any], plan: Dict[str, Any],
    outputs: Dict[int, List[Sequence[int]]], raster: Sequence[Sequence[int]],
) -> List[Tuple[int, ...]]:
    """This core's per-cycle slot counts, gathered through the routing plan.

    A producer's counts arrive from the cycle BEFORE (the twin gathers the
    previous cycle's outputs), while the entry raster is read at THIS cycle.
    """
    cycles = int(document["cycles_per_sample"])
    empty = _zero_rows(document)
    slots: List[Tuple[int, ...]] = []
    for cycle in range(cycles):
        previous = list(empty)
        for core, rows in outputs.items():
            previous[int(core)] = (
                rows[cycle - 1] if cycle >= 1 else tuple([0] * len(rows[0])))
        entry = raster[cycle] if cycle < len(raster) else [0] * len(raster[0])
        slots.append(bundle.gather_axon_slots(
            plan["routes"], previous, entry, where=f"core {plan['core']}"))
    return slots


def _run_dict(plan: Dict[str, Any], cycles: int, length: int) -> Dict[str, Any]:
    """The shape driver.fold_events/window_counts read a single-sample run in."""
    return {
        "samples": 1, "cycles_per_sample": int(cycles),
        "first_tag": bundle.FIRST_TAG, "latencies": [int(plan["latency"])],
        "neurons": [int(plan["neurons"])], "simulation_length": int(length),
    }


def percentiles(values: Sequence[float]) -> Dict[str, float]:
    """p50/p90/p99 by nearest rank — no numpy on a board node."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {}
    def at(fraction: float) -> float:
        rank = max(1, min(len(ordered), int(-(-fraction * len(ordered) // 1))))
        return ordered[rank - 1]
    return {"p50": at(0.50), "p90": at(0.90), "p99": at(0.99),
            "min": ordered[0], "max": ordered[-1],
            "mean": sum(ordered) / len(ordered), "total": sum(ordered)}


class Campaign:
    """Every pass of every sample, certified and timed."""

    #: A timed loop repeats the campaign hundreds of times; it prints a summary
    #: per window instead of a line per pass.
    quiet = False

    #: Keep every pass's stimulus and pacing, for a ``TransferControl`` to
    #: replay. Off by default; it changes nothing the campaign sends or checks.
    record_transfers = False

    def __init__(self, document: Dict[str, Any], session: Any, *,
                 samples: Sequence[int], arm: str = ARM_AS_RUN) -> None:
        if arm not in ARMS:
            raise OdinDeploymentRefusal(f"arm {arm!r} is not one of {ARMS}")
        self.document = document
        self.session = session
        self.samples = list(samples)
        self.arm = arm
        self.cycles = int(document["cycles_per_sample"])
        self.length = int(document["chip_config"]["simulation_length"])
        # THE AER WORDING, dispatched once on this bundle's own chip claims: a
        # stock stream read on a generated core would deliver the all-neuron
        # TREF 0x7F as a spike on axon slot 127.
        self.encoding = encoding.encoding_of_bundle(document)
        self.certified = set(bundle.certification_samples(document))
        self.rasters = {int(entry["index"]): entry["entry_raster"]
                        for entry in document["samples"]}
        self.frozen_stimulus = {
            int(entry["index"]): decode(
                entry["stimulus"], what=f"sample {entry['index']}/stimulus")
            for entry in document["samples"]}
        self.first_core = int(document["pass_order"][0])
        #: Pass core -> fabric core, from the NC this session declares.
        self.fabric_cores = int(getattr(session, "cores_implied", 1) or 1)
        self.placement = pass_placement(document, self.fabric_cores)
        self.reset()

    def reset(self) -> None:
        """Forget every pass's answer, so the same campaign can run again."""
        self.outputs: Dict[int, Dict[int, List[Tuple[int, ...]]]] = {
            index: {} for index in self.samples}
        self.windows: Dict[int, Dict[int, List[int]]] = {
            index: {} for index in self.samples}
        self.pass_rows: List[Dict[str, Any]] = []
        self.core_rows: List[Dict[str, Any]] = []
        self.certificates: List[Dict[str, Any]] = []
        #: (core, sample) -> {"stimulus", "pause_s"}, and core -> {"pause_s"} for
        #: the resident programming run; filled only with ``record_transfers``.
        self.transfers: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.programming_transfers: Dict[int, Dict[str, Any]] = {}

    def run(self) -> None:
        for plan in bundle.pass_plans(self.document):
            resident = ResidentPass(
                self.session, plan,
                capture_events=self.session.capture_capacity, arm=self.arm,
                fabric_core=self.placement[int(plan["core"])])
            row = resident.program_once()
            if not self.quiet:
                print(f"[deploy] core {row['core']}: programmed once on fabric "
                      f"core {row['fabric_core']}, "
                      f"{row['program_words']} words in "
                      f"{row['core_program_s'] * 1e3:.2f} ms")
                print(f"[deploy] core {row['core']}: segment boundary init "
                      f"{row['segment_boundary_init_s'] * 1e3:.2f} ms, "
                      f"{driver.stream_line(row)}")
                if "programming_run" in row:
                    run = row["programming_run"]
                    print(f"[deploy] core {row['core']}: resident programming "
                          f"run {run['device_cycles']} enabled cycles, "
                          f"{run['capture_events']} events, "
                          f"{run['run_s'] * 1e3:.2f} ms")
            self.core_rows.append(row)
            if self.record_transfers and "programming_run" in row:
                self.programming_transfers[int(row["core"])] = {
                    "pause_s": float(row["programming_run"]["run_s"])}
            try:
                for sample in self.samples:
                    self._one_pass(resident, plan, sample)
            finally:
                resident.release()

    def _one_pass(self, resident: ResidentPass, plan: Dict[str, Any],
                  sample: int) -> None:
        core = int(plan["core"])
        started = time.perf_counter()
        transcode_started = time.perf_counter()
        slots = transcode_slots(
            self.document, plan, self.outputs[sample], self.rasters[sample])
        tokens = bundle.pass_stimulus_tokens(
            plan, slots, encoding=self.encoding, sample=0,
            cycles_per_sample=self.cycles)
        stimulus = bundle.tokens_to_bytes(tokens)
        transcode_s = time.perf_counter() - transcode_started
        if core == self.first_core:
            self._require_frozen_stimulus(sample, stimulus)

        words, walls = resident.run_stimulus(stimulus)
        decode_started = time.perf_counter()
        events, device_cycles = resident.decode(words)
        run = _run_dict(plan, self.cycles, self.length)
        counts = driver.fold_events(events, run)
        rows = [
            tuple(counts.get((0, cycle, bundle.PASS_CORE, neuron), 0)
                  for neuron in range(resident.neurons))
            for cycle in range(self.cycles)
        ]
        window = driver.window_counts(counts, run)[0][0]
        decode_s = time.perf_counter() - decode_started

        self.outputs[sample][core] = rows
        self.windows[sample][core] = list(window)
        row = {
            "sample": int(sample), "core": core,
            "fabric_core": resident.fabric_core,
            "capture_events": len(events), "device_cycles": int(device_cycles),
            "transcode_s": transcode_s, "decode_s": decode_s,
            "pass_total_s": time.perf_counter() - started,
        }
        row.update({key: value for key, value in walls.items()})
        self.pass_rows.append(row)
        if self.record_transfers:
            # Everything the pass spent that was not a transfer (the transcode,
            # the run, the decode) is what a control pass waits instead.
            moved = walls["bo_write_s"] + walls["sync_s"] + walls["readback_s"]
            self.transfers[(core, int(sample))] = {
                "stimulus": stimulus,
                "pause_s": max(0.0, row["pass_total_s"] - moved)}
        if sample in self.certified:
            certificate = self._certify(sample, core, window)
            certificate["fabric_core"] = resident.fabric_core
            self.certificates.append(certificate)

    def _require_frozen_stimulus(self, sample: int, stimulus: bytes) -> None:
        """The first pass's live stimulus must BE the frozen one, byte for byte.

        Nothing else in this file re-derives the sequencer's token arithmetic;
        this is the check that the reader which builds every LATER pass's
        stimulus is the same one the repository built the evidence with.
        """
        frozen = self.frozen_stimulus[int(sample)]
        if stimulus != frozen:
            raise OdinTranscodeDiverged(
                f"sample {sample}: the stimulus this host built for the first "
                f"pass is {len(stimulus)} bytes and the bundle's frozen one is "
                f"{len(frozen)}; they differ. The shipped bundle reader and the "
                f"encoder that froze the evidence disagree, so every later "
                f"pass's stimulus would be built by a rule nobody verified")

    def _certify(self, sample: int, core: int, window: Sequence[int]
                 ) -> Dict[str, Any]:
        # One sample, one core: the certificate's shape is [sample][core][neuron].
        expected = bundle.expected_pass_windows(self.document, sample)[core]
        certificate = driver.Certificate([expected], [[list(window)]], samples=1)
        if not self.quiet:
            print(f"[deploy] sample {sample} core {core}: {certificate.line()}")
        return {"sample": int(sample), "core": int(core),
                "certificate": certificate.as_dict(),
                "certificate_line": certificate.line(),
                "passed": certificate.passed}

    def readout(self) -> Tuple[List[Dict[str, Any]], float]:
        """Every sample's final readout against its frozen expectation."""
        frozen = {int(row["sample"]): row
                  for row in self.document["expected"]["final"]}
        rows: List[Dict[str, Any]] = []
        correct = 0
        for sample in self.samples:
            scores = bundle.readout_scores(self.document, self.windows[sample])
            predicted = bundle.predicted_label(self.document, scores)
            want = frozen[sample]
            agrees = (scores == [int(v) for v in want["scores"]]
                      and predicted == int(want["predicted"]))
            correct += int(predicted == int(want["label"]))
            rows.append({
                "sample": int(sample), "label": int(want["label"]),
                "scores": [int(value) for value in scores],
                "expected_scores": [int(value) for value in want["scores"]],
                "predicted": int(predicted),
                "expected_predicted": int(want["predicted"]),
                "matches_frozen": bool(agrees),
                "correct": bool(predicted == int(want["label"])),
            })
        return rows, (correct / len(self.samples) if self.samples else 0.0)


def _kernel_starts(session: Any) -> Optional[int]:
    """The kernel starts a session has issued so far (None if it counts none)."""
    count = getattr(session, "kernel_starts", None)
    return None if count is None else int(count)


def timed_loop(campaign: Campaign, seconds: float, *,
               clock: Any = time.monotonic) -> Dict[str, Any]:
    """Whole campaigns back to back until ``seconds`` have passed.

    The loop never stops inside a campaign, so the window it spans holds only
    complete inferences: every sample's every pass, certified as it ran. It
    returns what the power estimator divides by (inferences, wall) and what it
    checks before trusting a window (certificates, readouts), with the
    kernel's own share of the wall (run walls over the loop wall), and the
    kernel starts the session counted (``ap_starts``).
    """
    starts_before = _kernel_starts(campaign.session)
    started = clock()
    totals = {
        "campaigns": 0, "inferences": 0, "passes": 0, "run_s": 0.0,
        "device_cycles": 0, "programming_runs": 0, "programming_run_s": 0.0,
        "programming_device_cycles": 0, "certificates": 0,
        "certificates_failed": 0, "readouts_drifted": 0,
    }
    while True:
        campaign.reset()
        campaign.run()
        readout_rows, _accuracy = campaign.readout()
        totals["campaigns"] += 1
        totals["inferences"] += len(campaign.samples)
        totals["passes"] += len(campaign.pass_rows)
        totals["run_s"] += sum(float(row["run_s"]) for row in campaign.pass_rows)
        totals["device_cycles"] += sum(
            int(row["device_cycles"]) for row in campaign.pass_rows)
        for row in campaign.core_rows:
            run = row.get("programming_run")
            if run:
                totals["programming_runs"] += 1
                totals["programming_run_s"] += float(run["run_s"])
                totals["programming_device_cycles"] += int(run["device_cycles"])
        totals["certificates"] += len(campaign.certificates)
        totals["certificates_failed"] += sum(
            1 for row in campaign.certificates if not row["passed"])
        totals["readouts_drifted"] += sum(
            1 for row in readout_rows if not row["matches_frozen"])
        if clock() - started >= float(seconds):
            break
    ended = clock()
    wall = ended - started
    starts_after = _kernel_starts(campaign.session)
    if starts_before is not None and starts_after is not None:
        totals["ap_starts"] = starts_after - starts_before
    totals.update({
        "arm": campaign.arm, "t_start": started, "t_end": ended, "wall_s": wall,
        "kernel_duty": (totals["run_s"] + totals["programming_run_s"]) / wall
        if wall > 0 else None,
        "passed": (totals["certificates_failed"] == 0
                   and totals["readouts_drifted"] == 0),
    })
    return totals


class TransferControl:
    """An arm's host-card transfers without ``ap_start``: the control condition.

    Built from a campaign that ran with ``record_transfers``. Each control
    campaign walks the same cores in the same order: the three buffers
    allocated and the program streamed in (``program_once`` of the as-run arm,
    which runs nothing), the resident arm's programming-run transfers when the
    source campaign had them, then per sample the stimulus write, the capture
    poison and the capture readback, each pass paced by the wall that pass took
    when it was certified. The kernel is never called.
    """

    def __init__(self, campaign: Campaign) -> None:
        if not campaign.transfers:
            raise OdinDeploymentRefusal(
                "the source campaign recorded no transfers: run it once with "
                "record_transfers = True, so the control replays what a "
                "certified campaign actually sent")
        self.document = campaign.document
        self.session = campaign.session
        self.arm = campaign.arm
        self.samples = list(campaign.samples)
        self.transfers = dict(campaign.transfers)
        self.programming_transfers = dict(campaign.programming_transfers)
        self.plans = list(bundle.pass_plans(campaign.document))
        self.placement = dict(campaign.placement)
        self.passes_per_inference = max(1, len(self.plans))
        readout_rows, _accuracy = campaign.readout()
        self.source_certified = bool(
            all(row["passed"] for row in campaign.certificates)
            and all(row["matches_frozen"] for row in readout_rows))
        missing = [(int(plan["core"]), sample) for plan in self.plans
                   for sample in self.samples
                   if (int(plan["core"]), sample) not in self.transfers]
        if missing:
            raise OdinDeploymentRefusal(
                f"the source campaign recorded no transfers for (core, sample) "
                f"{missing[:4]}: the control would not be the campaign it mirrors")

    def run(self, totals: Dict[str, Any], *, deadline: float,
            clock: Any = time.monotonic) -> bool:
        """One control campaign; False when ``deadline`` cut it at a pass boundary."""
        terminator = struct.pack("<I", driver.OP_END)
        for plan in self.plans:
            core = int(plan["core"])
            resident = ResidentPass(
                self.session, plan, capture_events=self.session.capture_capacity,
                arm=ARM_AS_RUN, fabric_core=self.placement[core])
            resident.program_once()
            totals["bytes_to_device"] += len(resident.program)
            try:
                programming = self.programming_transfers.get(core)
                if programming is not None:
                    moved = resident.transfers_only(terminator, programming["pause_s"])
                    totals["programming_replays"] += 1
                    totals["paced_s"] += moved["paused_s"]
                    totals["bytes_to_device"] += moved["bytes_to_device"]
                    totals["bytes_from_device"] += moved["bytes_from_device"]
                for sample in self.samples:
                    recorded = self.transfers[(core, sample)]
                    moved = resident.transfers_only(recorded["stimulus"],
                                                    recorded["pause_s"])
                    totals["passes"] += 1
                    totals["paced_s"] += moved["paused_s"]
                    totals["bytes_to_device"] += moved["bytes_to_device"]
                    totals["bytes_from_device"] += moved["bytes_from_device"]
                    if clock() >= deadline:
                        return False
            finally:
                resident.release()
        return True


def transfer_loop(control: TransferControl, seconds: float, *,
                  clock: Any = time.monotonic) -> Dict[str, Any]:
    """Control campaigns back to back until ``seconds`` have passed.

    Unlike ``timed_loop`` it may stop inside a campaign, at the first pass
    boundary after the time is spent: nothing here is certified, so nothing
    needs a whole campaign. ``seconds <= 0`` runs exactly one whole control
    campaign, as ``timed_loop(campaign, 0.0)`` runs one campaign. ``inferences``
    counts replayed passes over the passes one inference takes (a fraction when
    the window ends mid-sample). ``ap_starts`` is the kernel starts the
    session counted over the window, so a control that started the kernel
    would say so.
    """
    starts_before = _kernel_starts(control.session)
    if starts_before is None:
        raise OdinDeploymentRefusal(
            "the control's session counts no kernel starts (kernel_starts): the "
            "control could not show that it never started the kernel")
    started = clock()
    totals: Dict[str, Any] = {
        "campaigns": 0, "passes": 0, "programming_replays": 0, "ap_starts": 0,
        "run_s": 0.0, "paced_s": 0.0, "bytes_to_device": 0, "bytes_from_device": 0,
    }
    if float(seconds) <= 0.0:
        if control.run(totals, deadline=float("inf"), clock=clock):
            totals["campaigns"] += 1
    else:
        deadline = started + float(seconds)
        while True:
            if control.run(totals, deadline=deadline, clock=clock):
                totals["campaigns"] += 1
            if clock() >= deadline:
                break
    ended = clock()
    totals["ap_starts"] = int(_kernel_starts(control.session) or 0) - starts_before
    totals.update({
        "arm": control.arm, "control": "transfer", "t_start": started,
        "t_end": ended, "wall_s": ended - started,
        "inferences": totals["passes"] / float(control.passes_per_inference),
        "kernel_duty": 0.0, "source_certified": control.source_certified,
        "passed": control.source_certified,
    })
    return totals


def aggregate_units(rows: Sequence[Dict[str, Any]], names: Sequence[str]) -> Dict[str, Any]:
    """One window of N units: the sums, the mean and summed duty, and every unit.

    ``rows`` are the units' own ``timed_loop`` or ``transfer_loop`` results in
    unit order, ``names`` their unit names. Nothing is dropped: each unit's row
    is kept under ``per_cu`` with its name, and the window passes only when
    every unit's did.
    """
    rows = list(rows)
    if len(rows) != len(names):
        raise OdinDeploymentRefusal(
            f"{len(rows)} unit result(s) for {len(names)} unit name(s)")
    out: Dict[str, Any] = {
        "compute_units": len(rows),
        "per_cu": [dict(row, compute_unit=str(name)) for row, name in zip(rows, names)],
    }
    for key in UNIT_SUM_KEYS:
        if any(key in row for row in rows):
            out[key] = sum(row.get(key, 0) for row in rows)
    starts = [float(row["t_start"]) for row in rows if row.get("t_start") is not None]
    ends = [float(row["t_end"]) for row in rows if row.get("t_end") is not None]
    if starts and ends:
        out.update({"t_start": min(starts), "t_end": max(ends),
                    "wall_s": max(ends) - min(starts)})
    duties = [float(row["kernel_duty"]) for row in rows if row.get("kernel_duty") is not None]
    if duties:
        out["kernel_duty"] = sum(duties) / len(duties)
        out["kernel_duty_sum"] = sum(duties)
    for key in ("arm", "control"):
        values = {row.get(key) for row in rows if key in row}
        if len(values) == 1:
            out[key] = values.pop()
    if any("source_certified" in row for row in rows):
        out["source_certified"] = all(bool(row.get("source_certified")) for row in rows)
    out["passed"] = bool(rows) and all(bool(row.get("passed")) for row in rows)
    return out


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

TIMING_FIELDS = (
    "bo_write_s", "sync_s", "run_s", "readback_s", "decode_s", "transcode_s",
    "pass_total_s",
)


def programming(core_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The campaign's whole programming stream: bytes, seconds, bandwidth.

    Every segment boundary of a multi-pass deployment pays one of these, so the
    campaign total is what "what did it cost to feed the chip" means here.
    """
    total_bytes = sum(int(row[driver.STREAM_BYTES]) for row in core_rows)
    total_seconds = sum(float(row[driver.STREAM_SECONDS]) for row in core_rows)
    summary = dict(driver.stream_metrics(total_bytes, total_seconds))
    summary["segment_boundaries"] = len(core_rows)
    summary["segment_boundary_init_s"] = percentiles(
        [row["segment_boundary_init_s"] for row in core_rows])
    return summary


def aggregates(campaign: Campaign) -> Dict[str, Any]:
    per_field = {
        field: percentiles([row[field] for row in campaign.pass_rows])
        for field in TIMING_FIELDS
    }
    per_sample = {}
    for sample in campaign.samples:
        per_sample[sample] = sum(
            row["pass_total_s"] for row in campaign.pass_rows
            if row["sample"] == sample)
    return {
        "per_stage": per_field,
        "sample_total_s": percentiles(list(per_sample.values())),
        "core_program_s": percentiles(
            [row["core_program_s"] for row in campaign.core_rows]),
        "segment_boundary_init_s": percentiles(
            [row["segment_boundary_init_s"] for row in campaign.core_rows]),
        "passes": len(campaign.pass_rows),
        "samples": len(campaign.samples),
    }


def write_tsv(path: str, campaign: Campaign) -> int:
    header = ["sample", "core"] + list(TIMING_FIELDS) + [
        "capture_events", "device_cycles", "stimulus_words"]
    rows = campaign.pass_rows[:TSV_ROW_CAP]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(
                f"{row[key]:.9f}" if key.endswith("_s") else str(row[key])
                for key in header) + "\n")
    return len(rows)


def report(campaign: Campaign, *, document: Dict[str, Any], options: Any,
           readout_rows: Sequence[Dict[str, Any]], accuracy: float,
           wall_s: float, tsv_rows: int,
           device: Dict[str, Any]) -> Dict[str, Any]:
    failed = [row for row in campaign.certificates if not row["passed"]]
    drifted = [row for row in readout_rows if not row["matches_frozen"]]
    per_core = []
    for row in campaign.core_rows:
        core = int(row["core"])
        certs = [cert for cert in campaign.certificates if int(cert["core"]) == core]
        per_core.append(dict(
            row, passes=sum(1 for p in campaign.pass_rows if int(p["core"]) == core),
            certificates=len(certs),
            certificates_passed=sum(1 for cert in certs if cert["passed"])))
    return {
        "schema": "odin_hacc_deployment_report/1",
        "bundle": {
            "name": document["name"], "self_hash": document["self_hash"],
            "provenance": document["provenance"], "model": document["model"],
            "chip_config": document["chip_config"],
            "aer_encoding": campaign.encoding.as_dict(),
            "pass_order": document["pass_order"],
            "cycles_per_sample": document["cycles_per_sample"],
        },
        "host": driver.host_stamp(),
        "device": device,
        "campaign": {
            "samples_run": campaign.samples,
            "samples_shipped": len(document["samples"]),
            "sample_bound": options.samples,
            "sample_bound_source": options.samples_source,
            "certification_samples": sorted(campaign.certified),
            "arm": campaign.arm,
            "arm_source": getattr(options, "arm_source", None),
        },
        "accuracy": accuracy,
        "correct": sum(1 for row in readout_rows if row["correct"]),
        "readout": list(readout_rows),
        "certificates": campaign.certificates,
        "placement": {
            "fabric_cores": campaign.fabric_cores,
            "pass_to_fabric": {str(core): int(fabric)
                               for core, fabric in sorted(campaign.placement.items())},
        },
        "per_core": per_core,
        "programming": programming(campaign.core_rows),
        "walls": aggregates(campaign),
        "wall_s": wall_s,
        "tsv_rows": tsv_rows,
        "passed": not failed and not drifted and bool(campaign.samples),
    }


def _wait_for_start(barrier: str, index: int) -> None:
    """A unit that has opened its session says so, then waits for the release."""
    with open(f"{barrier}.ready.{int(index)}", "w", encoding="utf-8") as handle:
        handle.write(f"{time.monotonic()}\n")
    deadline = time.monotonic() + UNIT_START_TIMEOUT_S
    while not os.path.exists(f"{barrier}.go"):
        if time.monotonic() > deadline:
            raise OdinDeploymentRefusal(
                f"unit {index}: no start signal within {UNIT_START_TIMEOUT_S:.0f} s; "
                f"the parent of an N-unit campaign releases every unit together")
        time.sleep(0.01)


def mode_deploy(options: Any) -> int:
    if int(options.compute_units) > 1 and options.cu_index is None:
        return mode_units(options)
    if options.cu_index is not None:
        options.cu = driver.compute_unit_name(options.cu_index)
    document = load_bundle(options.bundle)
    require_protocol_agrees(document, source=options.bundle)
    require_pass_order_is_causal(document)
    samples = sample_indices(document, options.samples)
    print(f"[deploy] bundle   : {document['name']} ({document['self_hash']})")
    print(f"[deploy] provenance: {document['provenance']['derivation']}")
    print(f"[deploy] passes   : {document['pass_order']} over "
          f"{document['cycles_per_sample']} cycles/sample")
    print(f"[deploy] samples  : {len(samples)} of "
          f"{len(document['samples'])} shipped ({options.samples_source})")
    session = driver.open_session(options)
    if getattr(options, "start_barrier", None):
        _wait_for_start(options.start_barrier, int(options.cu_index or 0))
    aer = require_fabric_matches(document, session.capacity, source=options.bundle)
    print(f"[deploy] fabric   : {session.capacity.chip} "
          f"({session.capacity.axon_slots_per_core} axon slots/core), "
          f"AER {aer.name}")
    if options.replay:
        # A FAKE pyxrt is handed the frozen per-pass answers to replay, keyed by
        # the stimulus that earned each one; the real binding has no arm().
        with open(options.replay, "r", encoding="utf-8") as handle:
            replay = json.load(handle)
        verify_replay(replay, document, source=options.replay)
        if session.arm_fake(deployment=replay):
            print(f"[deploy] REPLAY   : {len(replay['runs'])} frozen answers "
                  f"from {options.replay} — structure, not silicon")
    print(f"[deploy] arm      : {options.arm} ({options.arm_source})")
    print(f"[deploy] placement: pass -> fabric core "
          f"{pass_placement(document, session.cores_implied)} on NC "
          f"{session.cores_implied} ({session.capacity.provenance})")
    started = time.perf_counter()
    window_start = time.monotonic()
    loop = None
    try:
        campaign = Campaign(document, session, samples=samples, arm=options.arm)
        campaign.run()
        readout_rows, accuracy = campaign.readout()
        campaign_ended = time.perf_counter()
        if options.loop_s:
            # The certified campaign above is the report; the loop repeats it,
            # quietly, for a power window an external sampler can bracket.
            # ``campaign`` is reset by the loop, so it runs on a twin.
            twin = Campaign(document, session, samples=samples, arm=options.arm)
            twin.quiet = True
            loop = timed_loop(twin, options.loop_s)
            print(f"[deploy] loop     : {loop['campaigns']} campaign(s), "
                  f"{loop['inferences']} inference(s) in {loop['wall_s']:.3f} s, "
                  f"kernel duty {loop['kernel_duty']:.3f}, "
                  f"{loop['certificates_failed']} failed certificate(s)")
        # Stamp the device while the session is still open: close() clears the
        # kernel/bank/capacity views, and a report stamped afterwards would
        # ship empty provenance for exactly the fields board triage reads.
        device_stamp = session.device_stamp()
        window_end = time.monotonic()
    finally:
        session.close()
    # The campaign's wall as the archived reports measured it (through the
    # stamp and the close); a loop, when asked for, is reported apart.
    wall_s = (campaign_ended if loop is not None else time.perf_counter()) - started
    os.makedirs(options.results, exist_ok=True)
    tsv_rows = write_tsv(os.path.join(options.results, TSV_NAME), campaign)
    payload = report(
        campaign, document=document, options=options,
        readout_rows=readout_rows, accuracy=accuracy, wall_s=wall_s,
        tsv_rows=tsv_rows, device=device_stamp)
    if loop is not None:
        payload["loop"] = loop
        payload["passed"] = bool(payload["passed"] and loop["passed"])
    if options.cu_index is not None:
        # One unit of an N-unit campaign: which unit, and when it ran (the host's
        # CLOCK_MONOTONIC, the clock every unit and the parent share).
        payload["compute_unit"] = {"name": options.cu, "index": int(options.cu_index),
                                   "of": int(options.compute_units)}
        payload["campaign_window"] = {"t_start": window_start, "t_end": window_end}
    path = driver.write_result(options.results, REPORT_NAME, payload)
    walls = payload["walls"]["per_stage"]
    print(f"[deploy] ACCURACY : {accuracy:.6f} "
          f"({payload['correct']}/{len(samples)} samples)")
    for field in TIMING_FIELDS:
        stage = walls[field]
        print(f"[deploy] {field:<14}: total {stage['total'] * 1e3:9.3f} ms  "
              f"p50 {stage['p50'] * 1e3:8.3f}  p90 {stage['p90'] * 1e3:8.3f}  "
              f"max {stage['max'] * 1e3:8.3f}")
    boundary = payload["walls"]["segment_boundary_init_s"]
    print(f"[deploy] {'segment_boundary_init_s':<14}: total "
          f"{boundary['total'] * 1e3:9.3f} ms  p50 {boundary['p50'] * 1e3:8.3f}  "
          f"p90 {boundary['p90'] * 1e3:8.3f}  max {boundary['max'] * 1e3:8.3f}")
    print(f"[deploy] programming: {driver.stream_line(payload['programming'])} "
          f"over {payload['programming']['segment_boundaries']} segment "
          f"boundary/boundaries")
    print(f"[deploy] wall     : {wall_s:.3f}s over "
          f"{payload['walls']['passes']} pass(es)")
    print(f"[deploy] {'PASS' if payload['passed'] else 'FAIL'}: wrote {path}")
    return 0 if payload["passed"] else 1


def _unit_argv(options: Any, index: int, results: str, barrier: str) -> List[str]:
    """The command line of unit ``index``: the parent's own options, one unit."""
    argv = [sys.executable, os.path.abspath(__file__),
            "--bundle", options.bundle, "--results", results, "--xclbin", options.xclbin,
            "--device-index", str(options.device_index),
            "--capture-events", str(options.capture_events),
            "--declare-cores", str(options.declare_cores),
            "--capture-ram-events", str(options.capture_ram_events),
            "--run-timeout-ms", str(options.run_timeout_ms),
            "--samples", str(options.samples), "--arm", str(options.arm),
            "--loop-s", str(options.loop_s),
            "--compute-units", str(options.compute_units),
            "--cu-index", str(int(index)), "--start-barrier", barrier]
    if options.chip:
        argv += ["--chip", options.chip]
    if options.fake_pyxrt:
        argv += ["--fake-pyxrt", options.fake_pyxrt]
    if options.replay:
        argv += ["--replay", options.replay]
    return argv


def mode_units(options: Any) -> int:
    """N units at once: one load, N unit processes released together, one report."""
    units = int(options.compute_units)
    names = driver.compute_unit_names(units)
    document = load_bundle(options.bundle)
    require_protocol_agrees(document, source=options.bundle)
    require_pass_order_is_causal(document)
    samples = sample_indices(document, options.samples)
    print(f"[deploy] bundle   : {document['name']} ({document['self_hash']})")
    print(f"[deploy] units    : {units} ({', '.join(names)}), {len(samples)} sample(s) "
          f"each, arm {options.arm} ({options.compute_units_source})")
    started = time.perf_counter()
    # The xclbin is loaded ONCE, here, before any unit opens: N processes
    # loading one file at the same moment would race for the same card.
    session = driver.open_session(options, open_kernel=False)
    try:
        require_fabric_matches(document, session.capacity, source=options.bundle)
        device = session.device_stamp()
    finally:
        session.close()
    device["compute_units"] = names
    os.makedirs(options.results, exist_ok=True)
    barrier = os.path.join(options.results, ".unit_start")
    for stale in [f"{barrier}.go"] + [f"{barrier}.ready.{k}" for k in range(units)]:
        if os.path.exists(stale):
            os.remove(stale)
    procs: List[Tuple[int, Any, Any]] = []
    codes: List[int] = []
    try:
        for index in range(units):
            unit_results = os.path.join(options.results, f"cu{index}")
            os.makedirs(unit_results, exist_ok=True)
            log = open(os.path.join(unit_results, "executor.log"), "w", encoding="utf-8")
            proc = subprocess.Popen(_unit_argv(options, index, unit_results, barrier),
                                    stdout=log, stderr=subprocess.STDOUT)
            procs.append((index, proc, log))
            # One unit at a time: the next one starts once this one holds its unit.
            deadline = time.monotonic() + UNIT_READY_TIMEOUT_S
            while not os.path.exists(f"{barrier}.ready.{index}"):
                if proc.poll() is not None or time.monotonic() > deadline:
                    raise OdinDeploymentRefusal(
                        f"unit {names[index]} did not open its session (exit "
                        f"{proc.poll()}); see {unit_results}/executor.log")
                time.sleep(0.01)
        with open(f"{barrier}.go", "w", encoding="utf-8") as handle:
            handle.write(f"{time.monotonic()}\n")
        codes = [proc.wait() for _index, proc, _log in procs]
    except BaseException:
        for _index, proc, _log in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        raise
    finally:
        for _index, _proc, log in procs:
            log.close()
    wall_s = time.perf_counter() - started
    reports: List[Any] = []
    for index in range(units):
        path = os.path.join(options.results, f"cu{index}", REPORT_NAME)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                reports.append(json.load(handle))
        else:
            reports.append(None)
    per_cu: List[Dict[str, Any]] = []
    certificates: List[Dict[str, Any]] = []
    readout: List[Dict[str, Any]] = []
    for index, (name, report_k) in enumerate(zip(names, reports)):
        if report_k is None:
            per_cu.append({"compute_unit": name, "index": index, "exit": codes[index],
                           "passed": False, "report": None})
            continue
        certs, rows = report_k["certificates"], report_k["readout"]
        per_cu.append({
            "compute_unit": name, "index": index, "exit": codes[index],
            "passed": bool(report_k["passed"]) and codes[index] == 0,
            "accuracy": report_k["accuracy"], "correct": report_k["correct"],
            "samples": len(rows), "certificates": len(certs),
            "certificates_passed": sum(1 for row in certs if row["passed"]),
            "readouts_matching_frozen": sum(1 for row in rows if row["matches_frozen"]),
            "identity": (report_k.get("device") or {}).get("identity"),
            "campaign_window": report_k.get("campaign_window"),
            "wall_s": report_k.get("wall_s"), "loop": report_k.get("loop"),
            "report": f"cu{index}/{REPORT_NAME}",
        })
        certificates += [dict(row, compute_unit=name) for row in certs]
        readout += [dict(row, compute_unit=name) for row in rows]
        print(f"[deploy] {name}: {'PASS' if per_cu[-1]['passed'] else 'FAIL'} "
              f"{per_cu[-1]['certificates_passed']}/{len(certs)} certificate(s), "
              f"{per_cu[-1]['readouts_matching_frozen']}/{len(rows)} readout(s) frozen")
    first = next((report_k for report_k in reports if report_k is not None), None)
    payload: Dict[str, Any] = {
        "schema": "odin_hacc_deployment_report/1",
        "compute_units": units,
        "bundle": first["bundle"] if first else {"name": document["name"],
                                                  "self_hash": document["self_hash"]},
        "host": driver.host_stamp(),
        "device": device,
        "campaign": {
            "samples_run": samples, "samples_shipped": len(document["samples"]),
            "sample_bound": options.samples, "sample_bound_source": options.samples_source,
            "certification_samples": sorted(bundle.certification_samples(document)),
            "arm": options.arm, "arm_source": getattr(options, "arm_source", None),
            "compute_units": units, "compute_units_source": options.compute_units_source,
        },
        "accuracy": (sum(row["correct"] for row in readout) / len(readout)) if readout else 0.0,
        "correct": sum(1 for row in readout if row["correct"]),
        "readout": readout,
        "certificates": certificates,
        "per_cu": per_cu,
        "wall_s": wall_s,
        "passed": len(per_cu) == units and all(row["passed"] for row in per_cu),
    }
    loops: List[Dict[str, Any]] = [row["loop"] for row in per_cu if row.get("loop")]
    if options.loop_s and len(loops) == units:
        payload["loop"] = aggregate_units(loops, names)
        payload["passed"] = bool(payload["passed"] and payload["loop"]["passed"])
    path = driver.write_result(options.results, REPORT_NAME, payload)
    print(f"[deploy] {'PASS' if payload['passed'] else 'FAIL'}: {units} unit(s), wrote {path}")
    return 0 if payload["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    package = os.path.dirname(_HERE)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--bundle",
        default=os.path.join(package, "deployment", "nc1_two_core_passes.json"),
        help="the sealed deployment bundle to execute")
    parser.add_argument("--results",
                        default=os.path.join(package, "results", "deployment"))
    parser.add_argument("--xclbin", default=os.environ.get("ODIN_XCLBIN", ""))
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--capture-events", type=int,
                        default=driver.DEFAULT_CAPTURE_EVENTS)
    parser.add_argument("--declare-cores", type=int, default=0)
    parser.add_argument("--capture-ram-events", type=int, default=0)
    parser.add_argument(
        "--chip", default=os.environ.get("ODIN_CHIP", ""),
        help=f"WHICH FABRIC this bitstream holds ($ODIN_CHIP; default "
             f"{driver.SHIPPED_CHIP_CONFIG}). The bundle names the fabric it "
             f"was mapped against through its own claims, and a mismatch "
             f"refuses rather than words events for the wrong crossbar")
    parser.add_argument("--run-timeout-ms", type=int,
                        default=driver.BLOCK_UNTIL_DONE_MS)
    parser.add_argument("--fake-pyxrt", default=None,
                        help="import this module instead of pyxrt (no hardware)")
    parser.add_argument("--replay", default=None,
                        help="frozen per-pass answers for a FAKE pyxrt to serve "
                             "(no hardware); ignored against a real card")
    parser.add_argument("--samples", type=int, default=None,
                        help=f"run at most this many shipped samples "
                             f"(${SAMPLES_ENV}; default {DEFAULT_SAMPLE_CAP})")
    parser.add_argument("--arm", default=None, choices=ARMS,
                        help=f"{ARM_AS_RUN}: every pass declares the whole "
                             f"program (the archived behaviour); "
                             f"{ARM_RESIDENT}: program each core once, then "
                             f"run with program_words = 1 (${ARM_ENV}; "
                             f"default {ARM_AS_RUN})")
    parser.add_argument("--loop-s", type=float, default=0.0,
                        help="after the certified campaign, repeat whole "
                             "campaigns for this many seconds (a power window)")
    parser.add_argument("--compute-units", type=int, default=None,
                        help=f"run the campaign on this many compute units at once, "
                             f"one process per unit (${UNITS_ENV}; default 1, the "
                             f"archived single session)")
    parser.add_argument("--cu-index", type=int, default=None,
                        help="(set by an N-unit parent) the one unit this process holds")
    parser.add_argument("--start-barrier", default=None,
                        help="(set by an N-unit parent) the release files' prefix")
    return parser


def resolve_units(options: Any) -> None:
    """How many compute units, and where the number came from."""
    if options.compute_units is not None:
        options.compute_units_source = "--compute-units on the command line"
    elif os.environ.get(UNITS_ENV):
        declared = os.environ[UNITS_ENV]
        try:
            options.compute_units = int(declared)
        except ValueError as exc:
            raise OdinDeploymentRefusal(
                f"{UNITS_ENV}={declared!r} is not an integer number of compute "
                f"units") from exc
        options.compute_units_source = f"{UNITS_ENV}={declared}"
    else:
        options.compute_units = 1
        options.compute_units_source = "the default (1 unit, the archived session)"
    if int(options.compute_units) < 1:
        raise OdinDeploymentRefusal(
            f"{options.compute_units} compute units: a campaign needs at least one")
    if options.cu_index is not None and not 0 <= int(options.cu_index) < int(
            options.compute_units):
        raise OdinDeploymentRefusal(
            f"unit index {options.cu_index} is outside {options.compute_units} "
            f"compute units")


def resolve_arm(options: Any) -> None:
    """The execution arm, and where it came from: recorded, not implied."""
    if options.arm is not None:
        options.arm_source = "--arm on the command line"
        return
    declared = os.environ.get(ARM_ENV)
    if declared:
        if declared not in ARMS:
            raise OdinDeploymentRefusal(
                f"{ARM_ENV}={declared!r} is not one of {ARMS}")
        options.arm = declared
        options.arm_source = f"{ARM_ENV}={declared}"
        return
    options.arm = ARM_AS_RUN
    options.arm_source = f"the default ({ARM_AS_RUN}, the archived behaviour)"


def resolve_sample_bound(options: Any) -> None:
    """The campaign size, and where the number came from — recorded, not implied."""
    if options.samples is not None:
        options.samples_source = "--samples on the command line"
        return
    declared = os.environ.get(SAMPLES_ENV)
    if declared:
        try:
            options.samples = int(declared)
        except ValueError as exc:
            raise OdinDeploymentRefusal(
                f"{SAMPLES_ENV}={declared!r} is not an integer") from exc
        options.samples_source = f"{SAMPLES_ENV}={declared}"
        return
    options.samples = DEFAULT_SAMPLE_CAP
    options.samples_source = (
        f"the packaged default of {DEFAULT_SAMPLE_CAP}; set {SAMPLES_ENV} to "
        f"widen or narrow the campaign")


def main(argv: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(argv)
    try:
        resolve_sample_bound(options)
        resolve_arm(options)
        resolve_units(options)
        return mode_deploy(options)
    except driver.OdinDriverError as exc:
        print(f"REFUSING: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except bundle.OdinBundleError as exc:
        # The schema module has no driver in scope and raises its own family;
        # a refusal that escaped untyped would look like a crash.
        print(f"REFUSING: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
