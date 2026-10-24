"""[W08] The lane block's quantities, one rule on both planes: lanes per tile and the lane work, or the lane count
withheld by name.

A chip declaring ``onchip_arithmetic`` prices its lane block on the record plane (from the sealed output stage) and
on the candidate plane (from the declaration and the lane plan's estimate), through this one function, so the joint
search ranks the chip every deployed record prices. Lanes declared without ``lanes_per_core`` withhold
``lanes_per_tile`` by name (the block's area and time are named unpriced) and still price the lane work they run.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from mimarsinan.deployment_record.quantities.spec import QuantityValue

#: The reason ``lanes_per_tile`` is withheld on a chip that declares lane classes but no lane count.
NO_LANE_COUNT = ("lanes are declared without lanes_per_core: no lane count sizes the lane block, so its area and "
                 "time are unpriced")


def lane_block_quantities(*, classes: Sequence[str], lanes_per_core: Optional[int], cores_per_tile: int,
                          lane_ops: int, provenance: str) -> Tuple[Dict[str, QuantityValue], Dict[str, str]]:
    """``(values, withheld)`` of a lane-declaring chip: ``lane_ops`` (the lane work per inference, at
    ``provenance``) and ``lanes_per_tile`` (lanes per core times the resolved cores per tile; 0 on the off ablation,
    which declares no class), or ``lanes_per_tile`` withheld by name without a lane count."""
    values = {"lane_ops": QuantityValue(float(lane_ops), provenance)}
    if classes and lanes_per_core is None:
        return values, {"lanes_per_tile": NO_LANE_COUNT}
    lanes = int(lanes_per_core) * int(cores_per_tile) if classes else 0  # type: ignore[arg-type]
    values["lanes_per_tile"] = QuantityValue(float(lanes), "static")
    return values, {}
