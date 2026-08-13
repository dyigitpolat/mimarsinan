"""The shipped reference cases, loaded from data next to the profiles they test."""

from __future__ import annotations

import json
import os
from typing import Dict, Mapping, Optional, Sequence, Tuple

from mimarsinan.deployment_record.correlation.case import ReferenceCase
from mimarsinan.deployment_record.correlation.run import (
    CaseCorrelation,
    correlate,
)

CASES_DIR = os.path.join(os.path.dirname(__file__), "cases")


def _load() -> Mapping[str, ReferenceCase]:
    cases: Dict[str, ReferenceCase] = {}
    for filename in sorted(os.listdir(CASES_DIR)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(CASES_DIR, filename)
        with open(path, encoding="utf-8") as handle:
            case = ReferenceCase.from_dict(json.load(handle))
        stem = filename[: -len(".json")]
        if case.name != stem:
            raise ValueError(
                f"{path}: declares name {case.name!r} but is filed as {stem!r}; the "
                f"filename is the case's identity"
            )
        cases[case.name] = case
    return cases


_CASES = _load()


def available_cases() -> Tuple[str, ...]:
    return tuple(_CASES)


def get_case(name: str) -> ReferenceCase:
    try:
        return _CASES[name]
    except KeyError:
        raise KeyError(
            f"{name!r} is not a shipped reference case; the cases are "
            f"{sorted(_CASES)}"
        ) from None


def correlate_all(names: Optional[Sequence[str]] = None) -> Tuple[CaseCorrelation, ...]:
    """Every shipped case (or the named ones), in a stable order."""
    selected = sorted(available_cases()) if names is None else list(names)
    return tuple(correlate(get_case(name)) for name in selected)
