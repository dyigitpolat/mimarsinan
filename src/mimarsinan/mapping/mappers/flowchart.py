"""Per-node flowchart estimate spec (V6 polymorphism for softcore viz).

The softcore DOT flowchart annotates each mapper node with a software (perceptron
count) summary and an optional FC hardware-estimate spec. The per-kind decision —
formerly an ``isinstance`` chain in ``softcore_flowchart_dot`` — is now each
``Mapper``'s own ``flowchart_node_estimate`` method, returning this value object.
The visualization layer turns :class:`FlowchartFCSpec` into the actual core
estimate (keeping the mapping package free of any visualization dependency).
"""

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
