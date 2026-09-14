"""Reads what ``pyproject.toml`` declares, so every third-party pin has one place to live."""

from __future__ import annotations

import os
import re
from typing import Dict, Iterator, List, Optional, Tuple

_SECTION = re.compile(r"^\[(?P<name>[^\]]+)\]\s*$")
_ARRAY_OPEN = re.compile(r"^(?P<key>[A-Za-z0-9_.\-]+)\s*=\s*\[")
_QUOTED = re.compile(r'"([^"]+)"')
# ``name[extra1,extra2] <specifier or @ direct-reference>``
_REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"\s*(?P<extras>\[[^\]]*\])?"
    r"\s*(?P<rest>.*)$"
)
BASE_KEY = "dependencies"
# Requirement arrays live in these tables; nothing else in them is a requirement.
_REQUIREMENT_TABLES = ("project", "project.optional-dependencies")


def manifest_path(path: Optional[str] = None) -> str:
    """Absolute path to the project's ``pyproject.toml``."""
    if path is not None:
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "..", "pyproject.toml"))


def _manifest_text(path: Optional[str]) -> str:
    manifest = manifest_path(path)
    if not os.path.isfile(manifest):
        return ""
    with open(manifest, "r", encoding="utf-8") as fh:
        return fh.read()


def _entries(text: str) -> Iterator[Tuple[str, str]]:
    """``(array key, requirement string)`` for every requirement under ``[project]``.

    A line scanner, not a TOML parser: Python 3.10 ships no ``tomllib`` and a
    parser is not worth a runtime dependency for one file whose shape this
    project owns. Quoted strings are lifted out before the closing bracket is
    looked for, so ``"uvicorn[standard]"`` does not read as the end of an array.
    """
    table: Optional[str] = None
    key: Optional[str] = None
    for raw in text.splitlines():
        stripped = raw.strip()
        section = _SECTION.match(stripped)
        if section:
            table, key = section.group("name"), None
            continue
        if table not in _REQUIREMENT_TABLES or stripped.startswith("#"):
            continue
        if key is None:
            opened = _ARRAY_OPEN.match(stripped)
            if opened is None:
                continue
            array_key = str(opened.group("key"))
            stripped = stripped.split("[", 1)[1]
        else:
            array_key = key
        key = array_key
        for requirement in _QUOTED.findall(stripped):
            yield array_key, requirement
        if "]" in _QUOTED.sub("", stripped):
            key = None


def declared_requirements(path: Optional[str] = None) -> List[str]:
    """Every requirement string declared under ``[project]``, base and extras alike."""
    return [requirement for _, requirement in _entries(_manifest_text(path))]


def declared_extras(path: Optional[str] = None) -> Dict[str, List[str]]:
    """``extra name -> its requirement strings`` from ``[project.optional-dependencies]``."""
    extras: Dict[str, List[str]] = {}
    for key, requirement in _entries(_manifest_text(path)):
        if key == BASE_KEY:
            continue
        extras.setdefault(key, []).append(requirement)
    return extras


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def declared_requirement(distribution: str, path: Optional[str] = None) -> Optional[str]:
    """The manifest's requirement string for ``distribution``, or ``None``."""
    wanted = _canonical(distribution)
    for requirement in declared_requirements(path):
        match = _REQUIREMENT.match(requirement.strip())
        if match and _canonical(match.group("name")) == wanted:
            return requirement.strip()
    return None


def declared_specifier(distribution: str, path: Optional[str] = None) -> Optional[str]:
    """The version specifier declared for ``distribution`` (``""`` when unbounded)."""
    requirement = declared_requirement(distribution, path)
    if requirement is None:
        return None
    match = _REQUIREMENT.match(requirement)
    assert match is not None
    return match.group("rest").strip()


def declared_pin(distribution: str, path: Optional[str] = None) -> Optional[str]:
    """The exact version declared as ``distribution==X``; ``None`` for any other form."""
    specifier = declared_specifier(distribution, path)
    if specifier is None:
        return None
    match = re.fullmatch(r"==\s*([0-9][0-9A-Za-z.\-]*)", specifier)
    return match.group(1) if match else None


def declared_direct_reference(
    distribution: str, path: Optional[str] = None
) -> Optional[str]:
    """The URL declared as ``distribution @ <url>``; ``None`` for a version specifier."""
    specifier = declared_specifier(distribution, path)
    if specifier is None or not specifier.startswith("@"):
        return None
    return specifier[1:].strip()


__all__ = [
    "BASE_KEY",
    "manifest_path",
    "declared_requirements",
    "declared_extras",
    "declared_requirement",
    "declared_specifier",
    "declared_pin",
    "declared_direct_reference",
]
