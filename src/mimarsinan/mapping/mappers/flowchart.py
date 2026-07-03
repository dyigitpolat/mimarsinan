"""Per-node flowchart estimate spec returned by each Mapper.flowchart_node_estimate (softcore viz)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlowchartFCSpec:
    """Inputs to the FC core estimator for one mapper node."""

    in_features: int
    out_features: int
    instances: int
    has_bias: bool


@dataclass(frozen=True)
class FlowchartNodeEstimate:
    """A node's flowchart annotation: software summary + optional FC estimate spec."""

    sw_text: str = "SW: n/a"
    fc_spec: FlowchartFCSpec | None = None
