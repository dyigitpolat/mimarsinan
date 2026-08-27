#!/usr/bin/env python3
"""HACC NUS - ODIN Deployment: one sealed bundle, one card, one measured verdict.

THE SHAPE OF THE PROBLEM. The shipped bitstream instantiates ONE ODIN core
(PROG_WORDS = NC * 262144 at NC=1, the only geometry build_xclbn.sh builds), and
the chip routes nothing between cores anyway (SPI_OPEN_LOOP). A multi-core
network therefore runs as one host-mediated PASS per core: program the core,
run a sample, read its per-cycle counts back, TRANSCODE them through the axon
source table into the next core's slot counts, build that core's stimulus on the
host, run again. Fabric memories persist across sequencer runs, so a core is
programmed ONCE and stimulated per sample; the per-sample membrane CLEAR is an
op inside each stimulus, not a reprogram.

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

    python3 host/odin_deployment_executor.py --xclbin <path> \\
        --bundle deployment/nc1_two_core_passes.json --results results/deploy
    ODIN_DEPLOY_SAMPLES=8 ...   # bound the campaign; the default is every
                                # shipped sample, capped at 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import odin_board_driver as driver
    import odin_deployment_bundle as bundle
except ImportError as _exc:  # pragma: no cover - a package missing its own halves
    raise SystemExit(
        f"REFUSING: {_exc}. odin_deployment_executor.py runs beside "
        f"odin_board_driver.py and odin_deployment_bundle.py in the package's "
        f"host/ directory; re-unzip the package rather than moving one half")

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
                 capture_events: int) -> None:
        self.session = session
        self.plan = plan
        self.core = int(plan["core"])
        self.neurons = int(plan["neurons"])
        self.latency = int(plan["latency"])
        self.capture_events = int(capture_events)
        self.capture_bytes = driver.capture_buffer_bytes(self.capture_events)
        self.program = decode(plan["program"], what=f"core {self.core}/program")
        self.program_words = len(self.program) // driver.WORD_BYTES
        self.stimulus_capacity = int(plan["max_stimulus_words"]) * driver.WORD_BYTES
        self._program_bo: Any = None
        self._stimulus_bo: Any = None
        self._capture_bo: Any = None

    def program_once(self) -> Dict[str, float]:
        """Allocate this core's three buffers and write its program in."""
        started = time.perf_counter()
        self._program_bo = self.session.allocate(
            len(self.program), driver.ARG_PROGRAM)
        self._stimulus_bo = self.session.allocate(
            self.stimulus_capacity, driver.ARG_STIMULUS)
        self._capture_bo = self.session.allocate(
            self.capture_bytes, driver.ARG_CAPTURE)
        allocated = time.perf_counter()
        walls = self.session.dma_in(self._program_bo, self.program)
        return {
            "core": self.core,
            "program_bytes": len(self.program),
            "program_words": self.program_words,
            "allocate_s": allocated - started,
            "program_bo_write_s": walls["bo_write_s"],
            "program_sync_s": walls["sync_s"],
            "core_program_s": time.perf_counter() - started,
        }

    def run_stimulus(self, stimulus: bytes) -> Tuple[List[int], Dict[str, float]]:
        """One sample on the resident core: rewrite, poison, start, read back."""
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
        stimulus_words = len(stimulus) // driver.WORD_BYTES
        driver.require_program_fits(
            self.program_words, stimulus_words, self.session.capacity,
            transport=self.session.name)
        write = self.session.dma_in(self._stimulus_bo, stimulus)
        poison = self.session.poison_capture(self._capture_bo)
        run_s = self.session.start_and_wait(
            self._program_bo, self._stimulus_bo, self._capture_bo,
            program_words=self.program_words, stimulus_words=stimulus_words,
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

    def release(self) -> None:
        self._program_bo = self._stimulus_bo = self._capture_bo = None


# ---------------------------------------------------------------------------
# The campaign
# ---------------------------------------------------------------------------


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

    def __init__(self, document: Dict[str, Any], session: Any, *,
                 samples: Sequence[int]) -> None:
        self.document = document
        self.session = session
        self.samples = list(samples)
        self.cycles = int(document["cycles_per_sample"])
        self.length = int(document["chip_config"]["simulation_length"])
        self.certified = set(bundle.certification_samples(document))
        self.rasters = {int(entry["index"]): entry["entry_raster"]
                        for entry in document["samples"]}
        self.frozen_stimulus = {
            int(entry["index"]): decode(
                entry["stimulus"], what=f"sample {entry['index']}/stimulus")
            for entry in document["samples"]}
        self.outputs: Dict[int, Dict[int, List[Tuple[int, ...]]]] = {
            index: {} for index in self.samples}
        self.windows: Dict[int, Dict[int, List[int]]] = {
            index: {} for index in self.samples}
        self.pass_rows: List[Dict[str, Any]] = []
        self.core_rows: List[Dict[str, Any]] = []
        self.certificates: List[Dict[str, Any]] = []
        self.first_core = int(document["pass_order"][0])

    def run(self) -> None:
        for plan in bundle.pass_plans(self.document):
            resident = ResidentPass(
                self.session, plan,
                capture_events=self.session.capture_capacity)
            row = resident.program_once()
            print(f"[deploy] core {row['core']}: programmed once, "
                  f"{row['program_words']} words in "
                  f"{row['core_program_s'] * 1e3:.2f} ms")
            self.core_rows.append(row)
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
            plan, slots, sample=0, cycles_per_sample=self.cycles)
        stimulus = bundle.tokens_to_bytes(tokens)
        transcode_s = time.perf_counter() - transcode_started
        if core == self.first_core:
            self._require_frozen_stimulus(sample, stimulus)

        words, walls = resident.run_stimulus(stimulus)
        decode_started = time.perf_counter()
        events, device_cycles = driver.decode_capture(
            words, resident.capture_events, transport=self.session.name)
        run = _run_dict(plan, self.cycles, self.length)
        counts = driver.fold_events(events, run)
        rows = [
            tuple(counts.get((0, cycle, 0, neuron), 0)
                  for neuron in range(resident.neurons))
            for cycle in range(self.cycles)
        ]
        window = driver.window_counts(counts, run)[0][0]
        decode_s = time.perf_counter() - decode_started

        self.outputs[sample][core] = rows
        self.windows[sample][core] = list(window)
        row = {
            "sample": int(sample), "core": core,
            "capture_events": len(events), "device_cycles": int(device_cycles),
            "transcode_s": transcode_s, "decode_s": decode_s,
            "pass_total_s": time.perf_counter() - started,
        }
        row.update({key: value for key, value in walls.items()})
        self.pass_rows.append(row)
        if sample in self.certified:
            self.certificates.append(self._certify(sample, core, window))

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


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

TIMING_FIELDS = (
    "bo_write_s", "sync_s", "run_s", "readback_s", "decode_s", "transcode_s",
    "pass_total_s",
)


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
           wall_s: float, tsv_rows: int) -> Dict[str, Any]:
    failed = [row for row in campaign.certificates if not row["passed"]]
    drifted = [row for row in readout_rows if not row["matches_frozen"]]
    return {
        "schema": "odin_hacc_deployment_report/1",
        "bundle": {
            "name": document["name"], "self_hash": document["self_hash"],
            "provenance": document["provenance"], "model": document["model"],
            "chip_config": document["chip_config"],
            "pass_order": document["pass_order"],
            "cycles_per_sample": document["cycles_per_sample"],
        },
        "host": driver.host_stamp(),
        "device": campaign.session.device_stamp(),
        "campaign": {
            "samples_run": campaign.samples,
            "samples_shipped": len(document["samples"]),
            "sample_bound": options.samples,
            "sample_bound_source": options.samples_source,
            "certification_samples": sorted(campaign.certified),
        },
        "accuracy": accuracy,
        "correct": sum(1 for row in readout_rows if row["correct"]),
        "readout": list(readout_rows),
        "certificates": campaign.certificates,
        "per_core": campaign.core_rows,
        "walls": aggregates(campaign),
        "wall_s": wall_s,
        "tsv_rows": tsv_rows,
        "passed": not failed and not drifted and bool(campaign.samples),
    }


def mode_deploy(options: Any) -> int:
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
    if options.replay:
        # A FAKE pyxrt is handed the frozen per-pass answers to replay, keyed by
        # the stimulus that earned each one; the real binding has no arm().
        with open(options.replay, "r", encoding="utf-8") as handle:
            replay = json.load(handle)
        verify_replay(replay, document, source=options.replay)
        if session.arm_fake(deployment=replay):
            print(f"[deploy] REPLAY   : {len(replay['runs'])} frozen answers "
                  f"from {options.replay} — structure, not silicon")
    started = time.perf_counter()
    try:
        campaign = Campaign(document, session, samples=samples)
        campaign.run()
        readout_rows, accuracy = campaign.readout()
    finally:
        session.close()
    wall_s = time.perf_counter() - started
    os.makedirs(options.results, exist_ok=True)
    tsv_rows = write_tsv(os.path.join(options.results, TSV_NAME), campaign)
    payload = report(
        campaign, document=document, options=options,
        readout_rows=readout_rows, accuracy=accuracy, wall_s=wall_s,
        tsv_rows=tsv_rows)
    path = driver.write_result(options.results, REPORT_NAME, payload)
    walls = payload["walls"]["per_stage"]
    print(f"[deploy] ACCURACY : {accuracy:.6f} "
          f"({payload['correct']}/{len(samples)} samples)")
    for field in TIMING_FIELDS:
        stage = walls[field]
        print(f"[deploy] {field:<14}: total {stage['total'] * 1e3:9.3f} ms  "
              f"p50 {stage['p50'] * 1e3:8.3f}  p90 {stage['p90'] * 1e3:8.3f}  "
              f"max {stage['max'] * 1e3:8.3f}")
    print(f"[deploy] wall     : {wall_s:.3f}s over "
          f"{payload['walls']['passes']} pass(es)")
    print(f"[deploy] {'PASS' if payload['passed'] else 'FAIL'}: wrote {path}")
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
    parser.add_argument("--program-words", type=int, default=0)
    parser.add_argument("--capture-ram-events", type=int, default=0)
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
    return parser


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
