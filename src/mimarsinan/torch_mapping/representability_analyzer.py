"""
Representability analysis for torch.fx graphs.

Walks an FX graph, classifies every node as supported / absorbable /
unsupported, and produces a ``RepresentabilityReport``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import torch.nn as nn
import torch.fx as fx


# ── Data types ───────────────────────────────────────────────────────────────

@dataclass
class OpInfo:
    """Description of a single operation in the FX graph."""
    node_name: str
    op_type: str
    module_type: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class RepresentabilityReport:
    """Result of analysing a traced model for mimarsinan representability."""
    is_representable: bool
    supported_ops: List[OpInfo] = field(default_factory=list)
    unsupported_ops: List[OpInfo] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    absorption_plan: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"Representable: {self.is_representable}"]
        lines.append(f"  Supported ops   : {len(self.supported_ops)}")
        lines.append(f"  Unsupported ops : {len(self.unsupported_ops)}")
        for op in self.unsupported_ops:
            lines.append(f"    - {op.node_name} ({op.op_type}): {op.reason}")
        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


class RepresentabilityError(Exception):
    """Raised when a model cannot be represented in mimarsinan IR."""

    def __init__(self, report: RepresentabilityReport):
        self.report = report
        super().__init__(report.summary())


# ── Module classification tables ─────────────────────────────────────────────

# Modules representable as NeuralCore (via Perceptron / Conv mappers)
_NEURAL_CORE_MODULES: set[type] = {
    nn.Linear,
    nn.Conv2d,
    nn.Conv1d,
}

# Modules that get absorbed into the preceding Perceptron.
# nn.Identity is absorbable so that chains like mm → Identity → BN → act
# can be normalized into a single Perceptron package.
_ABSORBABLE_MODULES: set[type] = {
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.ReLU,
    nn.LeakyReLU,
    nn.GELU,
    nn.Identity,
}


# ── Analyzer ─────────────────────────────────────────────────────────────────

class RepresentabilityAnalyzer:
    """Analyse an FX graph for mimarsinan representability."""

    def __init__(self, graph_module: fx.GraphModule):
        self.gm = graph_module
        self._modules: Dict[str, nn.Module] = dict(graph_module.named_modules())

    def analyze(self) -> RepresentabilityReport:
        report = RepresentabilityReport(is_representable=True)

        for node in self.gm.graph.nodes:
            if node.op == "placeholder" or node.op == "output":
                continue

            if node.op == "call_module":
                self._classify_module_node(node, report)
            elif node.op == "call_function":
                self._classify_function_node(node, report)
            elif node.op == "call_method":
                self._classify_method_node(node, report)
            elif node.op == "get_attr":
                report.supported_ops.append(
                    OpInfo(node.name, "get_attr")
                )

        self._build_absorption_plan(report)

        if report.unsupported_ops:
            report.is_representable = False

        return report

    def _classify_module_node(
        self, node: fx.Node, report: RepresentabilityReport
    ) -> None:
        mod = self._modules.get(node.target)
        if mod is None:
            report.unsupported_ops.append(
                OpInfo(node.name, "call_module", str(node.target),
                       reason=f"Module '{node.target}' not found")
            )
            report.is_representable = False
            return

        mod_type = type(mod)

        if mod_type in _NEURAL_CORE_MODULES:
            if isinstance(mod, (nn.Conv1d, nn.Conv2d)):
                groups = getattr(mod, "groups", 1)
                if groups > 1:
                    report.unsupported_ops.append(
                        OpInfo(node.name, "call_module", mod_type.__name__,
                               reason=f"Grouped convolution (groups={groups}) not supported")
                    )
                    return

        # Everything is supported: neural-core candidates become Perceptrons,
        # absorbable modules get folded into preceding Perceptrons,
        # and everything else becomes a generic ComputeOp or passthrough.
        report.supported_ops.append(
            OpInfo(node.name, "call_module", mod_type.__name__)
        )

    def _classify_function_node(
        self, node: fx.Node, report: RepresentabilityReport
    ) -> None:
        fn = node.target
        fn_name = getattr(fn, "__name__", str(fn))
        # All functions are supported: the converter handles them generically
        # (structural ops, passthrough, or ModuleComputeMapper).
        report.supported_ops.append(
            OpInfo(node.name, "call_function", fn_name)
        )

    def _classify_method_node(
        self, node: fx.Node, report: RepresentabilityReport
    ) -> None:
        method_name = node.target
        # All tensor methods are supported: the converter handles them
        # (view, reshape, permute, mean, etc. or passthrough for unknown).
        report.supported_ops.append(
            OpInfo(node.name, "call_method", method_name)
        )

    def _build_absorption_plan(self, report: RepresentabilityReport) -> None:
        """Determine which BN / activation nodes should be absorbed into a preceding Perceptron."""
        nodes = list(self.gm.graph.nodes)
        node_by_name = {n.name: n for n in nodes}

        for node in nodes:
            if node.op != "call_module":
                continue
            mod = self._modules.get(node.target)
            if mod is None:
                continue

            if not isinstance(mod, (*_ABSORBABLE_MODULES,)):
                continue

            # Check if the single input comes from a neural-core module
            if len(node.args) < 1:
                continue
            input_node = node.args[0]
            if not isinstance(input_node, fx.Node):
                continue

            target_node = self._find_absorption_target(input_node, report)
            if target_node is not None:
                report.absorption_plan[node.name] = f"absorbed_into:{target_node.name}"

    def _find_absorption_target(
        self, node: fx.Node, report: RepresentabilityReport
    ) -> Optional[fx.Node]:
        """Walk back through already-absorbed nodes to find the originating neural-core node."""
        if node.op == "call_module":
            mod = self._modules.get(node.target)
            if mod is not None:
                if type(mod) in _NEURAL_CORE_MODULES:
                    return node
                if node.name in report.absorption_plan:
                    if len(node.args) >= 1 and isinstance(node.args[0], fx.Node):
                        return self._find_absorption_target(node.args[0], report)
        return None
