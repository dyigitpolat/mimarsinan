"""[U-SFR] A neural segment's SANA-FE record persisted at the segment's end, and what the step keeps of it.

The SANA-FE step hands each finalized ``SanafeSegmentRecord`` to a :class:`SanafeSegmentStore`, which writes it to one
file per sample and segment under the run's working directory and returns a :class:`SanafeSegmentFile`: the file's
path, sha256 and size beside the totals and spike counts the step's own readers need (the run's aggregates, the
spike-parity gate and certificate, the TTFS comparisons, the event census's total, the snapshot's trace flag). The
full record is then freed with the segment, so the step's memory does not grow with the segment count.

File format (``SEGMENT_FILE_FORMAT``): consecutive pickles; a header, the record with ``message_trace`` None, the
flattened message trace in chunks (``_flatten_message_trace``'s list, streamed from SANA-FE's per-cycle lists so it is
never built in memory), each chunk a zlib-compressed pickle (a message pickles to about 490 bytes, its 26 keys spelled
out each time; compressed, to a few tens), and a trailer with the message count. :meth:`SanafeSegmentFile.load` checks
the sha256 first and returns the record with its ``message_trace`` exactly as the runner flattens it.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, BinaryIO, Iterator, List, Optional, Tuple

import numpy as np

from mimarsinan.chip_simulation.sanafe.records.energy import SanafeEnergyBreakdown
from mimarsinan.chip_simulation.sanafe.records.run import SanafeSegmentRecord
from mimarsinan.common.refusal import ToolchainRefusal

SEGMENT_FILE_FORMAT = "mimarsinan.sanafe.segment_record/1"
_TRACE_CHUNK = 8192  # flattened messages per pickle frame (bounds the writer's and reader's transient memory)


class SanafeSegmentFileRefusal(ToolchainRefusal, RuntimeError):
    """A persisted SANA-FE segment record that is missing, altered or not in the segment file format."""


@dataclass(frozen=True)
class SanafeCoreCounts:
    """One core's spike counts as the spike-count certificates and the TTFS comparisons read them (the field names of
    ``SanafeCoreRecord``, so ``SanafeRunRecord.to_hcm_subset`` reads either)."""

    core_index: int
    n_neurons: int
    n_axons_used: int
    core_latency: int
    has_hardware_bias: bool
    n_always_on_axons: int
    input_spike_count: np.ndarray
    output_spike_count: np.ndarray
    output_activation: Optional[np.ndarray] = None

    @classmethod
    def of(cls, core: Any) -> "SanafeCoreCounts":
        return cls(core_index=core.core_index, n_neurons=core.n_neurons, n_axons_used=core.n_axons_used,
                   core_latency=core.core_latency, has_hardware_bias=core.has_hardware_bias,
                   n_always_on_axons=core.n_always_on_axons, input_spike_count=core.input_spike_count,
                   output_spike_count=core.output_spike_count, output_activation=core.output_activation)


@dataclass
class SanafeSegmentFile:
    """A neural segment's full record on disk (``root``/``path``, its sha256 and size) and what the step keeps of it:
    the segment's totals, its spike counts per core and segment boundary, and its flattened message count."""

    path: str
    root: str
    sha256: str
    size_bytes: int
    message_count: int
    stage_index: int
    stage_name: str
    schedule_segment_index: Optional[int]
    schedule_pass_index: Optional[int]
    energy: SanafeEnergyBreakdown
    sim_time_s: float
    spikes: int
    packets_sent: int
    synaptic_events: Optional[int]
    seg_input_rates: np.ndarray
    seg_input_spike_count: np.ndarray
    seg_output_spike_count: np.ndarray
    per_core: List[SanafeCoreCounts] = field(default_factory=list)
    contract_ttfs_cores: List[Any] = field(default_factory=list)
    contract_ttfs_seg_output: Optional[np.ndarray] = None

    def load(self, *, with_message_trace: bool = True) -> SanafeSegmentRecord:
        """The full record, read from its file after its sha256 is checked; refused by name otherwise."""
        return read_segment_file(Path(self.root) / self.path, self.sha256, with_message_trace=with_message_trace)

    def relocated(self, root: Any) -> "SanafeSegmentFile":
        """The same file under another root (a moved or copied run directory)."""
        return replace(self, root=str(root))


def flat_messages(message_trace: Any) -> Iterator[dict]:
    """Each message of SANA-FE's per-cycle message lists in order, placeholders dropped, floats as Python floats: the
    entries of ``analysis.noc._flatten_message_trace``'s list, one at a time."""
    for events in message_trace or ():
        for ev in events:
            if isinstance(ev, dict):
                if ev.get("placeholder"):
                    continue
                yield {k: (float(v) if isinstance(v, float) else v)
                       for k, v in ev.items()}


class _Hashing:
    """A write-only file that hashes and counts what it passes on."""

    def __init__(self, out: BinaryIO):
        self.out, self.digest, self.size = out, hashlib.sha256(), 0

    def write(self, data: Any) -> int:
        self.digest.update(data)
        self.size += len(data)
        return self.out.write(data)


def write_segment_file(path: Path, record: SanafeSegmentRecord, message_trace: Any) -> Tuple[str, int, int]:
    """Write ``record`` (its ``message_trace`` None) and SANA-FE's raw ``message_trace`` flattened in chunks, atomically;
    returns the file's sha256, its size and the flattened message count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    count, chunk = 0, []
    try:
        with open(tmp, "wb") as raw:
            out = _Hashing(raw)
            pickle.dump({"format": SEGMENT_FILE_FORMAT}, out)
            pickle.dump(replace(record, message_trace=None), out)
            for message in flat_messages(message_trace):
                chunk.append(message)
                if len(chunk) == _TRACE_CHUNK:
                    _dump_chunk(chunk, out)
                    count, chunk = count + len(chunk), []
            if chunk:
                _dump_chunk(chunk, out)
                count += len(chunk)
            pickle.dump({"end": True, "message_count": count}, out)
            sha256, size = out.digest.hexdigest(), out.size
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return sha256, size, count


def _dump_chunk(chunk: List[dict], out: Any) -> None:
    pickle.dump(zlib.compress(pickle.dumps(chunk, protocol=pickle.HIGHEST_PROTOCOL), 1), out)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_segment_file(path: Path, sha256: str, *, with_message_trace: bool = True) -> SanafeSegmentRecord:
    """The record of a segment file whose bytes hash to ``sha256``, its message trace attached when asked."""
    if not path.is_file():
        raise SanafeSegmentFileRefusal(f"the persisted SANA-FE segment record {path} is missing")
    got = _file_sha256(path)
    if got != sha256:
        raise SanafeSegmentFileRefusal(f"the persisted SANA-FE segment record {path} has sha256 {got}, not the "
                                       f"recorded {sha256}")
    with open(path, "rb") as f:
        head = pickle.load(f)
        if not isinstance(head, dict) or head.get("format") != SEGMENT_FILE_FORMAT:
            raise SanafeSegmentFileRefusal(f"{path} is not a {SEGMENT_FILE_FORMAT} file")
        record = pickle.load(f)
        if not with_message_trace:
            return record
        flat: List[dict] = []
        frame = pickle.load(f)
        while isinstance(frame, bytes):
            flat.extend(pickle.loads(zlib.decompress(frame)))
            frame = pickle.load(f)
    if not (isinstance(frame, dict) and frame.get("end")):
        raise SanafeSegmentFileRefusal(f"{path} holds a frame that is neither a trace chunk nor its trailer")
    if len(flat) != int(frame["message_count"]):
        raise SanafeSegmentFileRefusal(f"{path} holds {len(flat)} messages against its trailer's "
                                       f"{frame['message_count']}")
    record.message_trace = flat or None
    return record


class SanafeSegmentStore:
    """One file per sample and neural segment under ``root``/``directory``."""

    def __init__(self, root: Any, directory: str):
        self.root, self.directory = str(root), str(directory)

    def persist(self, sample_index: int, record: SanafeSegmentRecord, message_trace: Any) -> SanafeSegmentFile:
        """Write ``record`` with SANA-FE's raw ``message_trace`` and return what the step keeps of it."""
        rel = f"{self.directory}/sample{int(sample_index):04d}_stage{int(record.stage_index):05d}.pickle"
        sha256, size, count = write_segment_file(Path(self.root) / rel, record, message_trace)
        return SanafeSegmentFile(
            path=rel, root=self.root, sha256=sha256, size_bytes=size, message_count=count,
            stage_index=record.stage_index, stage_name=record.stage_name,
            schedule_segment_index=record.schedule_segment_index, schedule_pass_index=record.schedule_pass_index,
            energy=record.energy, sim_time_s=record.sim_time_s, spikes=record.spikes,
            packets_sent=record.packets_sent, synaptic_events=record.synaptic_events,
            seg_input_rates=record.seg_input_rates, seg_input_spike_count=record.seg_input_spike_count,
            seg_output_spike_count=record.seg_output_spike_count,
            per_core=[SanafeCoreCounts.of(core) for core in record.per_core],
            contract_ttfs_cores=list(record.contract_ttfs_cores),
            contract_ttfs_seg_output=record.contract_ttfs_seg_output)
