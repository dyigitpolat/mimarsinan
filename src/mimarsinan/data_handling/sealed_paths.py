"""Sealed paths: a process-wide audit hook that refuses, by name, every open of a declared held-out file."""

from __future__ import annotations

import fnmatch
import os
import sys
import threading
from typing import Iterable, Optional, Tuple

from mimarsinan.common.refusal import ToolchainRefusal

_GLOB_CHARS = frozenset("*?[")
_SEALED: list[str] = []
_STATE = {"hook": False}
_REENTRY = threading.local()


class SealedPathRefusal(ToolchainRefusal, PermissionError):
    """An open of a path the process sealed (a held-out file is read only by its declared read)."""


class SealDeclarationRefusal(ToolchainRefusal, ValueError):
    """A seal entry that names no absolute path or anchored pattern."""


def _normalize(entry: str) -> str:
    text = os.fsdecode(entry)
    if not any(c in _GLOB_CHARS for c in text):
        return os.path.realpath(os.path.abspath(text))
    if not (os.path.isabs(text) or text.startswith("*")):
        raise SealDeclarationRefusal(
            f"sealed path pattern {text!r}: a pattern is absolute or starts with '*', so it cannot depend on the "
            "working directory"
        )
    return text


def _covers(pattern: str, path: str) -> bool:
    if any(c in _GLOB_CHARS for c in pattern):
        return fnmatch.fnmatchcase(path, pattern) or fnmatch.fnmatchcase(path, pattern + os.sep + "*")
    return path == pattern or path.startswith(pattern + os.sep)


def sealed_by(path) -> Optional[str]:
    """The seal entry that covers ``path`` as named (made absolute) or as resolved through its links, else ``None``."""
    named = os.path.abspath(os.fsdecode(path))
    forms = (named, os.path.realpath(named))
    return next((pattern for pattern in _SEALED if any(_covers(pattern, form) for form in forms)), None)


def _audit(event: str, args: tuple) -> None:
    if event != "open" or getattr(_REENTRY, "active", False):
        return
    path = args[0] if args else None
    if not isinstance(path, (str, bytes, os.PathLike)):
        return  # an open by file descriptor names no path
    _REENTRY.active = True
    try:
        pattern = sealed_by(path)
    finally:
        _REENTRY.active = False
    if pattern is not None:
        raise SealedPathRefusal(
            f"{os.fsdecode(path)} is sealed by {pattern!r} for this process (sealed_paths): a held-out file is "
            "opened only by its declared read, never by a run"
        )


def seal_paths(entries: Iterable[str]) -> Tuple[str, ...]:
    """Seal ``entries`` (files, directories, or patterns anchored at '/' or '*') for the rest of the process and
    return their normalized form; the audit hook is installed on the first non-empty seal and seals only tighten."""
    added = tuple(_normalize(e) for e in entries)
    if not added:
        return ()
    for pattern in added:
        if pattern not in _SEALED:
            _SEALED.append(pattern)
    if not _STATE["hook"]:
        sys.addaudithook(_audit)
        _STATE["hook"] = True
    return added


def sealed_patterns() -> Tuple[str, ...]:
    """Every entry sealed in this process, in sealing order."""
    return tuple(_SEALED)


def hook_installed() -> bool:
    return bool(_STATE["hook"])
