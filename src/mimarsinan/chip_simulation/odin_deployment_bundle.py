"""The HACC deployment bundle: one sealed document a board node can execute alone.

SHIPPED VERBATIM as ``host/odin_deployment_bundle.py`` (scripts/hacc/make_package.py
copies it and records its sha256), which is why it imports nothing but the
standard library: a board node has python3 and XRT, and nothing else is promised.

It defines the schema (manifest, per-core PROGRAM streams, per-sample core-0
STIMULUS streams, routing plan, frozen expectations), the fixtures' self-hash
seal, the inter-core TRANSCODE — the axon gather that turns one core's spike
counts into the next core's slot counts — and the token arithmetic that makes
those slot counts a stimulus stream.

It sits here rather than under ``odin_fpga/`` because ``odin_rtl/reference.py``
consumes the gather rule, and reaching it through the backend package would
close an import cycle (``odin_fpga/__init__`` -> ``segment`` -> that module).

CROSS-LANGUAGE CONTRACT: the opcodes, the AER-in word layout and the tag
numbering are mirrored from ``odin_rtl/{stimulus,cosim}.py`` and
``mapping/export/odin/program.py``; packaging REFUSES unless a stimulus this
module builds is byte-identical to the one those build.
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import zlib
from typing import Any, Dict, List, Mapping, Sequence, Tuple

SCHEMA = "odin_hacc_deployment/1"
WORD_BYTES = 4

#: Sequencer opcodes (odin_rtl/stimulus.py) and the AER-in address suffixes
#: (ODIN doc/README.md Sec.2.2.1) this module emits.
OP_END, OP_SPI_W, OP_AER, OP_WAIT, OP_TAG = 0, 1, 3, 4, 5
AER_NEURON_SPIKE_SUFFIX = 0x07
AER_ALL_NEURON_TREF = 0x7F

#: Tag 0 is what the sequencer starts with, so windows number from 1 (cosim.py).
FIRST_TAG = 1

#: ``SpikeSource`` sentinels, mirroring nevresim's ``constants.hpp``.
SOURCE_OFF, SOURCE_INPUT, SOURCE_ALWAYS_ON = -1, -2, -3

READOUT_ARGMAX = "argmax"


class OdinBundleError(RuntimeError):
    """The bundle cannot be executed as it stands; the message says why."""


class OdinBundleCorrupt(OdinBundleError):
    """A bundle's self-hash or payload hash does not match its own bytes."""


class OdinRoutingRefusal(ValueError):
    """An axon source names something the geometry it is gathered from lacks."""


# --- the seal: the fixture discipline, applied to a bundle -----------------


def canonical_bytes(document: Mapping[str, Any]) -> bytes:
    """The byte image the self-hash is taken over (``self_hash`` excluded)."""
    body = {key: value for key, value in document.items() if key != "self_hash"}
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")


def self_hash(document: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(document)).hexdigest()


def seal(document: Mapping[str, Any]) -> Dict[str, Any]:
    """Stamp a document with its own hash; the packager and the reader share this."""
    sealed = {key: value for key, value in document.items() if key != "self_hash"}
    sealed["self_hash"] = self_hash(sealed)
    return sealed


def encode_payload(payload: bytes) -> Dict[str, Any]:
    """A token stream as the bundle carries it: compressed, hashed, base64."""
    return {
        "encoding": "zlib+base64",
        "bytes": len(payload),
        "words": len(payload) // WORD_BYTES,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "data": base64.b64encode(zlib.compress(payload, 9)).decode("ascii"),
    }


def decode_payload(entry: Mapping[str, Any], *, what: str) -> bytes:
    if entry.get("encoding") != "zlib+base64":
        raise OdinBundleCorrupt(
            f"{what}: unknown payload encoding {entry.get('encoding')!r}; this "
            f"bundle reader only decodes 'zlib+base64'")
    payload = zlib.decompress(base64.b64decode(entry["data"]))
    digest = hashlib.sha256(payload).hexdigest()
    if digest != entry["sha256"] or len(payload) != int(entry["bytes"]):
        raise OdinBundleCorrupt(
            f"{what}: the decoded payload is {len(payload)} bytes / {digest}, "
            f"but the bundle declares {entry['bytes']} bytes / {entry['sha256']}. "
            f"Re-copy the package; a device programmed from corrupted bytes "
            f"runs a network nobody assembled")
    return payload


def verify_seal(document: Mapping[str, Any], *, source: str) -> Dict[str, Any]:
    """Refuse unless the document hashes to what it says it does."""
    if document.get("schema") != SCHEMA:
        raise OdinBundleCorrupt(
            f"{source}: schema {document.get('schema')!r} is not {SCHEMA!r}; "
            f"this reader executes one bundle format and would misread another")
    recomputed = self_hash(document)
    if document.get("self_hash") != recomputed:
        raise OdinBundleCorrupt(
            f"{source}: self-hash {document.get('self_hash')} but the content "
            f"hashes to {recomputed}. Either the upload is damaged or the file "
            f"was edited by hand; a bundle is frozen evidence, not a config")
    return dict(document)


def load_bundle(path: str) -> Dict[str, Any]:
    """Read one bundle and REFUSE unless it hashes to what it claims."""
    with open(path, "r", encoding="utf-8") as handle:
        return verify_seal(json.load(handle), source=path)


# --- the TRANSCODE: one core's counts become the next core's slot counts ---


def route_of_source(source: Any) -> Tuple[int, int]:
    """``(kind, index)`` for one axon source: a sentinel, or a producing core."""
    if getattr(source, "is_off_", False):
        return (SOURCE_OFF, 0)
    if getattr(source, "is_input_", False):
        return (SOURCE_INPUT, int(source.neuron_))
    if getattr(source, "is_always_on_", False):
        return (SOURCE_ALWAYS_ON, 0)
    return (int(source.core_), int(source.neuron_))


def core_routes(core: Any) -> Tuple[Tuple[int, int], ...]:
    """One core's axon-source table, in canonical slot order."""
    return tuple(route_of_source(s) for s in core.axon_sources)


def _read(values: Sequence[Any], index: int, what: str) -> Any:
    """Index a gather source loudly: an out-of-range read is a mapping defect."""
    if index < 0 or index >= len(values):
        raise OdinRoutingRefusal(
            f"{what} {index}, which does not exist ({len(values)} available). "
            f"The axon-source table and the geometry it is gathered from "
            f"disagree, and silently delivering zero would report a DIFFERENT "
            f"network's counts as the deployed ones.")
    return values[index]


def gather_axon_slots(
    routes: Sequence[Sequence[int]], previous_outputs: Sequence[Sequence[int]],
    input_counts: Sequence[int], *, where: str,
) -> Tuple[int, ...]:
    """One core's per-slot counts for one cycle — THE inter-core transcode.

    ``previous_outputs`` are the producers' counts from the cycle BEFORE (the
    chip routes nothing between cores: v1 routing is host-mediated);
    ``input_counts`` is the entry raster of THIS cycle.
    """
    slots: List[int] = []
    for slot, route in enumerate(routes):
        kind, index = int(route[0]), int(route[1])
        if kind == SOURCE_OFF:
            slots.append(0)
        elif kind == SOURCE_INPUT:
            slots.append(int(_read(
                input_counts, index, f"{where} slot {slot} reads input line")))
        elif kind == SOURCE_ALWAYS_ON:
            slots.append(1)
        else:
            producer = _read(
                previous_outputs, kind, f"{where} slot {slot} reads core")
            slots.append(int(_read(
                producer, index,
                f"{where} slot {slot} reads core {kind} neuron")))
    return tuple(slots)


# --- slot counts become a stimulus stream ---------------------------------


def tag_of(sample: int, cycle: int, cycles_per_sample: int) -> int:
    return FIRST_TAG + int(sample) * int(cycles_per_sample) + int(cycle)


def aer_spike_word(row: int) -> int:
    """The 17-bit AER-in address of one neuron-spike event on ``row``."""
    if row < 0 or row >= (1 << 8):
        raise OdinBundleError(f"pre-synaptic row {row} is not an 8-bit value")
    return (int(row) << 8) | AER_NEURON_SPIKE_SUFFIX


def inject_pairs(
    slot_rows: Mapping[int, Sequence[int]], counts: Sequence[int],
) -> Tuple[Tuple[int, int], ...]:
    """Per-slot counts -> the physical ``(row, multiplicity)`` delivery order.

    Ascending slots with each slot's multiplicity adjacent — the canonical drain
    of ``mapping/platform/event_order.py``; an undriven row contributes nothing.
    """
    return tuple(
        (int(row), int(counts[slot]))
        for slot in range(len(counts)) if int(counts[slot]) > 0
        for row in slot_rows.get(int(slot), ())
    )


def cycle_tokens(
    slot_rows: Mapping[int, Sequence[int]], counts: Sequence[int], *,
    tag: int, barrier_cycles: int, inject: bool, core: int = 0,
) -> List[int]:
    """One cycle of a pass: TAG, the AER deliveries, the TREF, the BARRIER."""
    tokens: List[int] = [OP_TAG, int(tag)]
    if inject:
        for row, multiplicity in inject_pairs(slot_rows, counts):
            word = aer_spike_word(row)
            for _occurrence in range(multiplicity):
                tokens.extend((OP_AER, int(core), word))
    tokens.extend((OP_AER, int(core), AER_ALL_NEURON_TREF))
    tokens.extend((OP_WAIT, int(barrier_cycles)))
    return tokens


def pass_stimulus_tokens(
    plan: Mapping[str, Any], per_cycle_slots: Sequence[Sequence[int]], *,
    sample: int, cycles_per_sample: int,
) -> List[int]:
    """The whole per-sample stimulus of one pass: CLEAR prefix, then the cycles."""
    slot_rows = {int(slot): tuple(int(row) for row in rows)
                 for slot, rows in plan["slot_rows"]}
    latency = int(plan["latency"])
    barrier = int(plan["barrier_cycles"])
    if len(per_cycle_slots) != int(cycles_per_sample):
        raise OdinBundleError(
            f"core {plan['core']}: the transcode produced {len(per_cycle_slots)} "
            f"cycle(s) against the bundle's {cycles_per_sample}; a short pass "
            f"would run a truncated network")
    tokens: List[int] = [int(word) for word in plan["clear_prefix"]]
    for cycle, counts in enumerate(per_cycle_slots):
        tokens.extend(cycle_tokens(
            slot_rows, counts,
            tag=tag_of(sample, cycle, cycles_per_sample),
            barrier_cycles=barrier, inject=cycle >= latency))
    tokens.append(OP_END)
    return tokens


def tokens_to_bytes(tokens: Sequence[int]) -> bytes:
    """Device-endian words: the AXI convention the XDMA shell reads."""
    return struct.pack(f"<{len(tokens)}I", *(int(word) for word in tokens))


# --- reading a bundle ------------------------------------------------------


def pass_plans(document: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The per-core passes in execution order — producers before consumers."""
    by_core = {int(plan["core"]): plan for plan in document["cores"]}
    return [by_core[int(index)] for index in document["pass_order"]]


def readout_scores(document: Mapping[str, Any], windows: Mapping[int, Sequence[int]]):
    """The class scores of one sample, gathered from the readout core's window."""
    row = windows[int(document["readout"]["core"])]
    return [int(_read(row, int(neuron), "the readout reads neuron"))
            for neuron in document["readout"]["neurons"]]


def predicted_label(document: Mapping[str, Any], scores: Sequence[int]) -> int:
    """The declared readout rule applied to one sample's scores (first max wins)."""
    rule = str(document["readout"]["rule"])
    if rule != READOUT_ARGMAX:
        raise OdinBundleError(
            f"readout rule {rule!r} is not one this reader implements "
            f"({READOUT_ARGMAX!r}); a guess would report another net's accuracy")
    return max(range(len(scores)), key=lambda index: (int(scores[index]), -index))


def certification_samples(document: Mapping[str, Any]) -> Tuple[int, ...]:
    """The sample indices whose PER-PASS counts are frozen, not just the readout."""
    return tuple(int(index) for index in document["certification"]["samples"])


def expected_pass_windows(document: Mapping[str, Any], sample: int) -> Dict[int, list]:
    """``{core: [[per-neuron window counts]]}`` for one certified sample."""
    return {int(core): [[int(v) for v in row] for row in rows] for core, rows
            in document["certification"]["windows"][str(int(sample))].items()}
