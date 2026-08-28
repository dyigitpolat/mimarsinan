#!/usr/bin/env python3
"""Regenerate the committed deployment bundle the package ships and the gates run.

    env/bin/python scripts/hacc/make_deployment_bundle.py [--out PATH] [--check]

The bundle is COMMITTED EVIDENCE, not a build product: the local gates and the
board executor both run these exact bytes, and regenerating them needs an RTL
simulator that a board node does not have. ``--check`` rebuilds and refuses if
the committed file has drifted, which is how the suite notices rot without
paying for a cosimulation on every run.

THE NETWORK. A deliberately tiny two-core segment in the same family as the
R11a/e2e fixtures: a producer whose theta sits low enough that one cycle carries
MULTIPLE spikes on one slot (a per-cycle law could not reproduce it), feeding a
consumer whose whole axon table is host-delivered. Because the shipped bitstream
is NC=1, the consumer is a SECOND PASS whose stimulus does not exist until the
producer's counts come back — which is exactly the mechanism under test.

THE LABELS are synthetic and chosen so the accuracy accumulator is exercised on
both a hit and a miss; they are not a claim about a trained model.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "hacc"))

from deployment_bundle_builder import build_bundle  # noqa: E402

from mimarsinan.chip_simulation.odin_fpga import kernel_registers  # noqa: E402
from mimarsinan.chip_simulation.odin_rtl.toolchain import available_engine  # noqa: E402
from mimarsinan.chip_simulation.soma_law import SomaLaw  # noqa: E402
from mimarsinan.code_generation.cpp_chip_model import SpikeSource  # noqa: E402
from mimarsinan.mapping.latency.chip import ChipLatency  # noqa: E402
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping  # noqa: E402

DEPLOYMENT = REPO / "scripts" / "hacc" / "package" / "deployment"
DEFAULT_OUT = DEPLOYMENT / "nc1_two_core_passes.json"
#: The replay evidence the local gates' fake pyxrt answers with. It never
#: reaches a board node: on a card the answers come from the card.
DEFAULT_CAPTURE_OUT = DEPLOYMENT / "nc1_two_core_passes_capture.json"

#: The stock-ODIN point, as every other ODIN gate resolves it.
ODIN_LAW = SomaLaw.resolve({
    "spiking_family": "lif", "spiking_variant": "streamed",
    "firing_mode": "Novena", "thresholding_mode": "<=",
    "firing_granularity": "per_event", "membrane_bits": 8,
})

INPUT_LINES = 4
NEURONS = 3
TIMESTEPS = 4
WEIGHT_BITS = 4
EFFECTIVE_MAX_AXONS = 127

#: Four entry rasters and the synthetic labels the readout is scored against.
RASTERS = (
    ((1, 0, 1, 1),) * TIMESTEPS,
    ((1, 1, 1, 1),) * TIMESTEPS,
    tuple(tuple(1 if (cycle + line) % 2 == 0 else 0 for line in range(INPUT_LINES))
          for cycle in range(TIMESTEPS)),
    ((0, 1, 0, 1),) * TIMESTEPS,
)
LABELS = (1, 2, 0, 0)

#: The samples whose PER-PASS counts are frozen. A subset by design: per-pass
#: certification is what proves the transcode, and it does not need every sample.
CERTIFICATION = (0, 2)


def _core(matrix, *, threshold: float, sources) -> HardCore:
    values = np.asarray(matrix, dtype=np.float64)
    core = HardCore(
        axons_per_core=values.shape[0], neurons_per_core=values.shape[1],
        has_bias_capability=False)
    core.core_matrix = values
    core.axon_sources = list(sources)
    core.threshold = float(threshold)
    core.available_axons = 0
    core.available_neurons = 0
    return core


def two_core_network() -> HardCoreMapping:
    """The producer/consumer pair, with the producer emitting multiplicity > 1.

    The seed and the two thetas are CHOSEN, and what they were chosen for is a
    fixture with teeth: the producer emits four spikes on one slot in one cycle
    (a per-cycle law could not reproduce it), the consumer carries inhibitory
    weights (so the signed row pair is exercised), and the readout answers three
    DIFFERENT classes over the four samples — a broken transcode cannot pass by
    guessing a constant.
    """
    rng = np.random.default_rng(32)
    producer = rng.integers(1, 6, size=(INPUT_LINES + 1, NEURONS)).astype(np.float64)
    producer[:, 0] = 3.0
    consumer = rng.integers(-4, 6, size=(NEURONS, NEURONS)).astype(np.float64)
    mapping = HardCoreMapping(chip_cores=[])
    mapping.cores = [
        _core(producer, threshold=4.0,
              sources=[SpikeSource(-2, index, is_input=True)
                       for index in range(INPUT_LINES)]
              + [SpikeSource(-3, 0, is_always_on=True)]),
        _core(consumer, threshold=4.0,
              sources=[SpikeSource(0, index) for index in range(NEURONS)]),
    ]
    mapping.output_sources = np.asarray(
        [SpikeSource(1, index) for index in range(NEURONS)], dtype=object)
    return mapping


def git_head():
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        capture_output=True, text=True, check=True).stdout.strip())
    return head, dirty


def build():
    mapping = two_core_network()
    chip_latency = int(ChipLatency(mapping).calculate())
    head, dirty = git_head()
    engine = available_engine()
    return build_bundle(
        name="nc1_two_core_passes",
        title="NC=1: a two-core segment as two host-mediated passes",
        description=(
            "The producer runs first; its per-cycle counts are transcoded through "
            "the axon-source table into the consumer's slot counts, and the "
            "consumer's stimulus is built on the host from that transcode. Every "
            "count below was MEASURED on the vendored RTL, one pass at a time, "
            "and equals the cycle-accurate twin at every cycle."),
        mapping=mapping,
        rasters=[[list(row) for row in raster] for raster in RASTERS],
        labels=list(LABELS),
        simulation_length=TIMESTEPS,
        chip_latency=chip_latency,
        soma_law=ODIN_LAW,
        weight_bits=WEIGHT_BITS,
        effective_max_axons=EFFECTIVE_MAX_AXONS,
        weight_sign_granularity="per_axon",
        membrane_init=0,
        readout_core=1,
        certification=CERTIFICATION,
        provenance={
            "generating_commit": head,
            "worktree_dirty": dirty,
            "cosim_engine": engine,
            "generator": "scripts/hacc/make_deployment_bundle.py",
            "derivation": (
                "per-pass counts measured by mimarsinan.chip_simulation.odin_rtl"
                ".cosim.run_cosim on the vendored ODIN core and gated against "
                "the cycle-accurate twin at every cycle"),
            "labels": (
                "SYNTHETIC: chosen so the accuracy accumulator sees both a hit "
                "and a miss; not a claim about a trained model"),
        },
        kernel_table=dict(_kernel_table()),
        model={
            "name": "two_core_witness",
            "input_lines": INPUT_LINES,
            "classes": NEURONS,
            "timesteps": TIMESTEPS,
            "samples": len(RASTERS),
            "synthetic": True,
        },
    )


def _kernel_table() -> dict:
    """The device-protocol table, read from the repository SSOT."""
    return {
        "name": kernel_registers.KERNEL_NAME,
        "arg_program": kernel_registers.ARG_PROGRAM,
        "arg_stimulus": kernel_registers.ARG_STIMULUS,
        "arg_capture": kernel_registers.ARG_CAPTURE,
        "arg_program_words": kernel_registers.ARG_PROGRAM_WORDS,
        "arg_stimulus_words": kernel_registers.ARG_STIMULUS_WORDS,
        "arg_capture_words": kernel_registers.ARG_CAPTURE_WORDS,
        "kernel_args": kernel_registers.KERNEL_ARGS,
        "capture_header_words": kernel_registers.CAPTURE_HEADER_WORDS,
        "header_events_seen": kernel_registers.HEADER_EVENTS_SEEN,
        "header_device_cycles": kernel_registers.HEADER_DEVICE_CYCLES,
        "capture_record_words": kernel_registers.CAPTURE_RECORD_WORDS,
        "capture_no_verdict": kernel_registers.CAPTURE_NO_VERDICT,
        "word_bytes": kernel_registers.WORD_BYTES,
    }


def rendered(document: dict) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--capture-out", default=str(DEFAULT_CAPTURE_OUT))
    parser.add_argument("--check", action="store_true",
                        help="rebuild and refuse if the committed bytes drifted")
    options = parser.parse_args(argv)
    document, capture = build()
    text = rendered(document)
    target = Path(options.out)
    if options.check:
        if not target.is_file():
            print(f"REFUSING: {target} does not exist", file=sys.stderr)
            return 2
        committed = target.read_text(encoding="utf-8")

        # The provenance carries the generating commit and the worktree's dirty
        # flag, which move on every commit, and the self-hash moves with them;
        # the EVIDENCE is everything else — programs, stimuli, routing, and
        # above all the frozen expectations certificates are scored against.
        def evidence(doc: dict) -> dict:
            body = json.loads(json.dumps(doc))
            body.pop("self_hash", None)
            for volatile in ("generating_commit", "worktree_dirty"):
                body.get("provenance", {}).pop(volatile, None)
            return body

        if evidence(json.loads(committed)) != evidence(document):
            print(f"REFUSING: {target} no longer matches a fresh build",
                  file=sys.stderr)
            return 1
        print(f"[bundle] {target} still matches a fresh build")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    replay = Path(options.capture_out)
    replay.write_text(rendered(capture), encoding="utf-8")
    print(f"[bundle] wrote {replay} ({len(capture['runs'])} replayable runs)")
    print(f"[bundle] wrote {target} ({len(text)} bytes, "
          f"self_hash {document['self_hash']})")
    print(f"[bundle] passes  : {document['pass_order']}, "
          f"{document['cycles_per_sample']} cycles/sample")
    print(f"[bundle] samples : {len(document['samples'])}, certified "
          f"{document['certification']['samples']}")
    print(f"[bundle] expected: "
          f"{[row['predicted'] for row in document['expected']['final']]} "
          f"against labels {list(LABELS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
