"""The declarations a residency plan (``residency_plan``) is not composed with on this tree: why a declared plan
refuses beside them, the one rule both doors call over the layout they judge (``core_type_doors``) and the builder's
message (``hybrid_build_scheduled``), so the doors refuse what the build would refuse after training.

Each rule is keyed on the declaration, as the other operand and lane rules are (``position_multiplexing``'s legal set,
``lane_operand_conflict``), and judges a plan the run reads (scheduling allowed):

- an operand core type (N2): the plan pins cores by payload, and its units do not yet state which cores hold dynamic
  operand banks (rewritten every inference, never pinned);
- executed fused-add lanes (WI-5e class E on the integer path): the planner takes each layer's tile groups as units in
  layer order (``schedule/residency_units.py``), and a fused add places its lane core in no program order of those
  units, so the build refuses every step ("layer order is not a program order"). Clamp and pool lanes without E, and
  lanes off the integer path, keep the plan. A malformed lane declaration is its own rule's."""

from __future__ import annotations

from typing import Any, List, Mapping

from mimarsinan.mapping.lanes.declaration import ADD, LANE_KEY, LaneDeclarationError, lanes_executed

#: The builder's refusal of a plan for a program with dynamic operand banks, and the doors' (one text).
OPERAND_RESIDENCY_REFUSAL = (
    "residency_plan pins cores by payload, and its units do not yet state which cores hold dynamic operand banks "
    "(rewritten every inference, never pinned): refused for a program with operand banks")


def residency_plan_conflicts(config: Mapping[str, Any]) -> List[str]:
    """Why a declared ``residency_plan`` refuses beside the chip's other declarations, else []."""
    if config.get("residency_plan") is None or not config.get("allow_scheduling"):
        return []
    out: List[str] = []
    operand = [i for i, c in enumerate(config.get("cores") or ()) if isinstance(c, Mapping) and c.get("operand")]
    if operand:
        out.append(f"cores {operand} declare an operand core type: {OPERAND_RESIDENCY_REFUSAL} (declare no "
                   "residency_plan, or no operand core type)")
    try:
        lanes = lanes_executed(config)
    except LaneDeclarationError:
        lanes = None
    if lanes is not None and lanes.declares(ADD):
        out.append(f"residency_plan is not composed with executed fused-add lanes on this tree: {LANE_KEY} declares "
                   f"class E on the integer path (readout_law), and the residency planner takes each layer's tile "
                   "groups as units in layer order, in which a fused add's lane core runs in no program order (the "
                   f"build refuses every step: layer order is not a program order); declare {LANE_KEY} without E, "
                   "or no residency_plan")
    return out
