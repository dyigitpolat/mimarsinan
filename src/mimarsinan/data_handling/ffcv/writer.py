"""Materialize a torch ``Dataset`` into an FFCV ``.beton`` file."""

from __future__ import annotations

import errno
import os
import time
from pathlib import Path
from typing import Any, Callable

from mimarsinan.data_handling.ffcv.cache import beton_path_for
from mimarsinan.data_handling.ffcv.pipeline_spec import (
    PipelineSpec,
    normalize_split_name,
)


def _instantiate_field(field_spec):
    from ffcv import fields as ffcv_fields

    cls = getattr(ffcv_fields, field_spec.write_type)
    return cls(**field_spec.write_kwargs)


def _acquire_lockfile(path: Path, *, timeout: float = 600.0) -> int:
    """Blocking flock on a sibling ``<path>.lock``; concurrent writers wait."""
    import fcntl

    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except OSError as exc:
            if exc.errno != errno.EWOULDBLOCK:
                os.close(fd)
                raise
            if time.monotonic() > deadline:
                os.close(fd)
                raise TimeoutError(f"could not acquire lock on {lock_path} in {timeout}s") from exc
            time.sleep(1.0)


def _release_lockfile(fd: int) -> None:
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def ensure_beton(
    spec: PipelineSpec,
    split: str,
    dataset_factory: Callable[[], Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Return the cached beton path; build it from ``dataset_factory()`` if missing.

    ``dataset_factory`` must return a ``torch.utils.data.Dataset`` whose
    ``__getitem__`` yields tuples matching ``spec.fields`` in order.
    Writes go to a sibling tmp file then atomic-rename, so concurrent
    readers never see a partial beton.
    """
    from ffcv.writer import DatasetWriter

    split = normalize_split_name(split)
    path = beton_path_for(spec, split)
    if path.exists() and not overwrite:
        return path

    lock_fd = _acquire_lockfile(path)
    try:
        if path.exists() and not overwrite:
            return path
        fields = {fs.name: _instantiate_field(fs) for fs in spec.fields}
        dataset = dataset_factory()
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        writer = DatasetWriter(str(tmp_path), fields)
        writer.from_indexed_dataset(dataset)
        os.replace(str(tmp_path), str(path))
        return path
    finally:
        _release_lockfile(lock_fd)
