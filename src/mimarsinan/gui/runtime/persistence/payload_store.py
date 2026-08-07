"""Content-addressed store for the large arrays resource sources carry.

Without it every core persisted its own dense copy of a shared payload: the
real vehicle wrote 179 GB of resource sources per run for 27 distinct grids.
Sources now reference payloads by content digest, so equal content is written
once and a monitor resolves it at render time.

Hashing stays cheap because the in-process session keys on the array's MEMORY
REGION first (data pointer + shape + strides + dtype). Materialization sharing
upstream (``matrix_memo``) and bank-view sharing (``resolve_pre_pruning_heatmap``)
mean the same region recurs thousands of times, so only distinct regions are
ever hashed. Session entries retain their arrays, so a freed-and-reallocated
buffer cannot alias a stale pointer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

__all__ = [
    "PAYLOAD_DIRNAME",
    "PAYLOAD_MIN_BYTES",
    "load_payload",
    "reset_payload_session",
    "store_payload",
]

PAYLOAD_DIRNAME = "_payloads"
# Below this, a payload reference costs more than the bytes it saves.
PAYLOAD_MIN_BYTES = 64 * 1024

_session: Dict[Tuple, Tuple[str, Any]] = {}


def reset_payload_session() -> None:
    """Forget region→digest memoization (and release the retained arrays)."""
    _session.clear()


def _region_key(array: np.ndarray) -> Tuple:
    interface = array.__array_interface__
    return (
        int(interface["data"][0]), array.shape, array.strides,
        array.dtype.str, int(array.nbytes),
    )


def _digest_of(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.blake2b(contiguous.tobytes(), digest_size=20).hexdigest()


def store_payload(root: Path, array: np.ndarray) -> str:
    """Write ``array`` under its content digest if absent; return the digest."""
    key = _region_key(array)
    cached = _session.get(key)
    if cached is not None:
        return cached[0]
    digest = _digest_of(array)
    path = root / PAYLOAD_DIRNAME / f"{digest}.npy"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".npy.tmp")
        try:
            with open(tmp, "wb") as f:
                np.save(f, np.ascontiguousarray(array), allow_pickle=False)
            tmp.replace(path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
    # Retaining the array pins its region, so the pointer key stays truthful.
    _session[key] = (digest, array)
    return digest


def load_payload(root: Path, digest: str) -> np.ndarray:
    """Read a payload by digest; raises like any missing/corrupt source file."""
    path = root / PAYLOAD_DIRNAME / f"{digest}.npy"
    with open(path, "rb") as f:
        return np.load(f, allow_pickle=False)
