"""The fresh-wizard starter draft: the packaged baseline document + a fresh run name."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# The baseline is a deployment-config DOCUMENT (data, like tier configs and
# templates) — workload facts live in the document, never in framework code.
# It mirrors the tier-0 anchor cell: the measured fast-passing family.
_BASELINE_PATH = Path(__file__).with_name("starter_baseline.json")

_last_issued_name: Optional[str] = None
_collision_counter = 0


def load_starter_baseline() -> Dict[str, Any]:
    """The raw baseline document (deterministic; callers own the copy)."""
    with open(_BASELINE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _fresh_name(base: str, now: datetime) -> str:
    global _last_issued_name, _collision_counter
    name = f"{base}_{now:%Y%m%d_%H%M%S}"
    if _last_issued_name is not None and _last_issued_name.startswith(name):
        _collision_counter += 1
        name = f"{name}_{_collision_counter}"
    else:
        _collision_counter = 0
    _last_issued_name = name
    return name


def starter_draft(now: Optional[datetime] = None) -> Dict[str, Any]:
    """A ready-to-launch wizard draft: the baseline document with a fresh
    experiment name. The fresh-state contract (tests/unit/gui/
    test_wizard_starter.py) pins it resolvable, emittable, and mappable."""
    draft = deepcopy(load_starter_baseline())
    draft["experiment_name"] = _fresh_name(
        str(draft.get("experiment_name") or "baseline"), now or datetime.now()
    )
    return draft
