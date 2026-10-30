"""The bundle as a FILE: its canonical rendering, its campaign, its statistics."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import json

from mimarsinan.chip_simulation import odin_deployment_bundle as bundle
from mimarsinan.chip_simulation import odin_deployment_encoding as encoding
from mimarsinan.chip_simulation.odin_fpga import kernel_registers
from mimarsinan.chip_simulation.odin_fpga.chip_selection import named_chip_of_bundle

BUNDLE_BASENAME = "deployment_bundle.json"
CAPTURE_BASENAME = "deployment_bundle_capture.json"


def render_bundle(document: Mapping[str, Any]) -> str:
    """The canonical byte image: sorted, unspaced, ASCII, one trailing newline."""
    return json.dumps(document, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True) + "\n"


def campaign_indices(config: Mapping[str, Any]) -> List[int]:
    """Which test-set samples the bundle SHIPS, as declared by the config."""
    count = int(config["odin_hacc_bundle_samples"])
    if count <= 0:
        raise ValueError(
            f"odin_hacc_bundle_samples={count} ships no samples at all; a "
            f"bundle that carries no sample cannot be executed on a board")
    certified = int(config["odin_hacc_certification_samples"])
    if certified <= 0 or certified > count:
        raise ValueError(
            f"odin_hacc_certification_samples={certified} must be in "
            f"[1, odin_hacc_bundle_samples={count}]: the certification subset "
            f"is the samples whose PER-PASS counts are frozen, and it is drawn "
            f"from the shipped samples")
    return list(range(count))


def kernel_table() -> Dict[str, Any]:
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


def frozen_accuracy(document: Mapping[str, Any]) -> float:
    """What the FROZEN expectations claim this network scores on its samples."""
    rows = document["expected"]["final"]
    if not rows:
        return 0.0
    hits = sum(1 for row in rows if int(row["predicted"]) == int(row["label"]))
    return hits / len(rows)


def bundle_fabric(document: Mapping[str, Any]) -> Dict[str, Any]:
    """WHICH FABRIC this bundle names, DERIVED from its own claims.

    Nothing in the sealed document says a chip name — it says the claims that
    identify one — so the name a package and a build read is resolved here and
    never declared. ``None`` means the declared envelope is narrower than any
    fabric this build produces, which is executable on a wider one and is only
    a refusal where a bitstream has to be chosen.
    """
    return {
        "chip": named_chip_of_bundle(document),
        "aer_encoding": encoding.encoding_of_bundle(document).as_dict(),
    }


def bundle_stats(document: Mapping[str, Any], capture: Mapping[str, Any],
                 paths: Mapping[str, str]) -> Dict[str, Any]:
    """The JSON-safe projection a pipeline entry and a record fragment carry."""
    cores = document["cores"]
    fabric = bundle_fabric(document)
    return {
        "name": document["name"],
        "self_hash": document["self_hash"],
        "schema": document["schema"],
        "chip": fabric["chip"],
        "aer_encoding": fabric["aer_encoding"],
        "samples": len(document["samples"]),
        "cores": len(cores),
        "pass_order": list(document["pass_order"]),
        "cycles_per_sample": int(document["cycles_per_sample"]),
        "certification_samples": len(document["certification"]["samples"]),
        "readout_core": int(document["readout"]["core"]),
        "readout_neurons": len(document["readout"]["neurons"]),
        "program_words": [int(plan["program"]["words"]) for plan in cores],
        "max_stimulus_words": [int(plan["max_stimulus_words"]) for plan in cores],
        "frozen_accuracy": frozen_accuracy(document),
        "replay_runs": len(capture["runs"]),
        "witness": document["provenance"]["derivation"],
        "paths": dict(paths),
        "bytes": len(render_bundle(document).encode("utf-8")),
    }


def verify_rendered(text: str, *, source: str) -> Dict[str, Any]:
    """Parse a rendered bundle and REFUSE unless it seals to its own bytes."""
    return bundle.verify_seal(json.loads(text), source=source)
