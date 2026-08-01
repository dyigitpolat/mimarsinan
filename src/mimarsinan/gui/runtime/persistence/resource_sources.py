"""On-disk container for a resource's SOURCE data, so a monitor can render it later."""

from __future__ import annotations

import io
import json
import logging
import struct
from pathlib import Path
from typing import IO, Any

import numpy as np

from mimarsinan.gui.resources import ResourceSource, resource_source_from_state
from mimarsinan.gui.runtime.persistence.resource_paths import resource_source_disk_path

logger = logging.getLogger("mimarsinan.gui")

MAGIC = b"MIMRSRC1"
_HEADER_LEN = struct.Struct("<I")
# A zero BIT PATTERN, not a zero VALUE: viewing the buffer as unsigned integers
# keeps -0.0 and NaN out of the omitted set, so the sparse form is bit-exact.
_BITS_BY_ITEMSIZE = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}
_INDEX_BYTES = 4
# Below this the second .npy header the sparse form needs costs more than the
# padding it drops, so small arrays are always stored whole.
_SPARSE_MIN_BYTES = 4096


def _encode_array(stream: IO[bytes], array: np.ndarray) -> dict[str, Any]:
    """Write ``array`` and return its header entry, dropping padding when that pays."""
    array = np.ascontiguousarray(array)
    flat = array.reshape(-1)
    bits_dtype = _BITS_BY_ITEMSIZE.get(array.dtype.itemsize)
    if bits_dtype is not None and array.nbytes >= _SPARSE_MIN_BYTES and flat.size < 2 ** 31:
        indices = np.flatnonzero(flat.view(bits_dtype))
        if indices.size * (_INDEX_BYTES + array.dtype.itemsize) < array.nbytes:
            np.save(stream, indices.astype(np.int32), allow_pickle=False)
            np.save(stream, flat[indices], allow_pickle=False)
            return {
                "encoding": "sparse",
                "shape": [int(d) for d in array.shape],
                "dtype": array.dtype.str,
            }
    np.save(stream, array, allow_pickle=False)
    return {"encoding": "dense"}


def _decode_array(stream: IO[bytes], entry: dict[str, Any]) -> np.ndarray:
    if entry["encoding"] == "dense":
        return np.load(stream, allow_pickle=False)
    indices = np.load(stream, allow_pickle=False)
    values = np.load(stream, allow_pickle=False)
    array = np.zeros(tuple(entry["shape"]), dtype=np.dtype(entry["dtype"]))
    array.reshape(-1)[indices] = values
    return array


def _write_source(stream: IO[bytes], source: ResourceSource) -> None:
    state = source.to_state()
    arrays = {k: v for k, v in state.items() if isinstance(v, np.ndarray)}
    header: dict[str, Any] = {
        "source_type": source.SOURCE_TYPE,
        "meta": {k: v for k, v in state.items() if k not in arrays},
        "arrays": [],
    }
    # Blobs are buffered so the header, which names each array's encoding, can be
    # written first and read back without seeking.
    body = io.BytesIO()
    for name, array in arrays.items():
        entry = _encode_array(body, array)
        entry["name"] = name
        header["arrays"].append(entry)
    encoded = json.dumps(header).encode("utf-8")
    stream.write(MAGIC)
    stream.write(_HEADER_LEN.pack(len(encoded)))
    stream.write(encoded)
    stream.write(body.getvalue())


def _read_source(stream: IO[bytes]) -> ResourceSource:
    if stream.read(len(MAGIC)) != MAGIC:
        raise ValueError("not a resource source file")
    (header_len,) = _HEADER_LEN.unpack(stream.read(_HEADER_LEN.size))
    header = json.loads(stream.read(header_len).decode("utf-8"))
    state: dict[str, Any] = dict(header["meta"])
    for entry in header["arrays"]:
        state[entry["name"]] = _decode_array(stream, entry)
    return resource_source_from_state(header["source_type"], state)


def save_resource_source(
    working_directory: str,
    step_name: str,
    kind: str,
    rid: str,
    source: ResourceSource,
) -> Path:
    """Write ``source`` where the monitor will look for it; the file appears whole or not at all."""
    path = resource_source_disk_path(working_directory, step_name, kind, rid)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "wb") as f:
            _write_source(f, source)
        tmp.replace(path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return path


def load_resource_source(
    working_directory: str,
    step_name: str,
    kind: str,
    rid: str,
) -> ResourceSource | None:
    """The persisted source for one resource, or ``None`` when there is nothing readable."""
    try:
        path = resource_source_disk_path(working_directory, step_name, kind, rid)
    except ValueError:
        return None
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as f:
            return _read_source(f)
    except (OSError, ValueError, KeyError, struct.error, json.JSONDecodeError) as e:
        logger.debug("Failed to load resource source %s: %s", path, e)
        return None


__all__ = ["load_resource_source", "save_resource_source"]
