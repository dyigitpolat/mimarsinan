""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("w0_price_e2", HERE / "w0_price.py")
W0 = importlib.util.module_from_spec(_spec)
sys.modules["w0_price_e2"] = W0
_spec.loader.exec_module(W0)

_members_for = W0.members_for


def members_for(group: str):
    from mimarsinan.mapping.platform.target_bundles import get_target_bundle
    slots = get_target_bundle(W0.CHIP).values.get("carry_input_slots")
    members = _members_for(group)
    if slots is not None:
        for declare in members[W0.MEMBER]["declare"].values():
            if declare is not None:
                declare["carry_input_slots"] = slots
    return members


W0.members_for = members_for

if __name__ == "__main__":
    raise SystemExit(W0.main())
