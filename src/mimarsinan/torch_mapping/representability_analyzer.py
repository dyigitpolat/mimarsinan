"""Representability analysis for torch.fx graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional

import torch.nn as nn
import torch.fx as fx

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.capability.requirements import GROUPED_CONVOLUTION, is_grouped_convolution, requirement_of
from mimarsinan.mapping.mappers.conv_helpers import perceptron_conv_padding
from mimarsinan.mapping.platform.packaging_contract import (
    SPIKING_PACKAGING,
    PackagingContract,
)
from mimarsinan.torch_mapping.converter_handlers.batchnorm_lowering import (
    BATCHNORM_TYPES as BATCHNORM_TYPES,
    batchnorm_channel_axis as batchnorm_channel_axis,
    batchnorm_folds_into as batchnorm_folds_into,
    has_running_statistics as has_running_statistics,
)
from mimarsinan.torch_mapping.converter_handlers.rules.method_rules import has_method_rule
from mimarsinan.torch_mapping.fx_shape_utils import node_output_shape, node_target_str, shared_state_sites


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


class RepresentabilityError(ToolchainRefusal):
    """Raised when a model cannot be represented in mimarsinan IR."""

    def __init__(self, report: RepresentabilityReport):
        self.report = report
        super().__init__(report.summary())


_NEURAL_CORE_MODULES: set[type] = {nn.Linear, nn.Conv2d, nn.Conv1d}

# The activations a Perceptron absorbs, with the name it rebuilds each from (perceptron.make_activation).
_ABSORBABLE_ACTIVATION_NAMES = ((nn.ReLU, "ReLU"), (nn.LeakyReLU, "LeakyReLU"), (nn.GELU, "GELU"))
ABSORBABLE_ACTIVATIONS = tuple(cls for cls, _name in _ABSORBABLE_ACTIVATION_NAMES)


def absorbed_activation_name(act: Optional[nn.Module]) -> Optional[str]:
    """The name an absorbed activation is rebuilt from; None for no activation or a non-absorbable one."""
    return next((name for cls, name in _ABSORBABLE_ACTIVATION_NAMES if isinstance(act, cls)), None)


def activation_folds_exactly(act: nn.Module) -> bool:
    """The Perceptron rebuilds an absorbed activation by name with default parameters; only those fold exactly."""
    if isinstance(act, nn.LeakyReLU):
        return act.negative_slope == nn.LeakyReLU().negative_slope
    if isinstance(act, nn.GELU):
        return act.approximate == "none"
    return isinstance(act, nn.ReLU)


@dataclass(frozen=True)
class LoweringRequirement:
    """A capability class a module's core lowering needs and today's cores do not declare: the module is the exact
    host op of its own call, its reason a stated boundary (never a refusal)."""
    capability: str
    reason: str


def core_lowering_requirement(module: nn.Module) -> Optional[LoweringRequirement]:
    """The one conversion-time capability rule, read from the requirement table (``mapping/capability``): a
    convolution with ``groups > 1`` needs the classes of the table's grouped-convolution row (G, work item N5), so it
    runs on the host with the row's stated boundary; ``None`` for a module the core lowering takes."""
    if not isinstance(module, (nn.Conv1d, nn.Conv2d)) or not is_grouped_convolution(module):
        return None
    row = requirement_of(GROUPED_CONVOLUTION)
    classes = "".join(sorted(row.classes))
    return LoweringRequirement(classes, f"grouped convolution (groups={int(module.groups)}) is the exact host op of "
                                        f"its module: its core lowering needs capability class {classes} "
                                        f"({row.host_boundary})")


def lowers_to_perceptron(absorbs_activation: bool, packaging: PackagingContract) -> bool:
    """A weight op becomes a perceptron unless the contract requires an activation and none folds into it."""
    return absorbs_activation or not packaging.require_activation


# The tensor call_methods with a dedicated rule (SSOT shared with ``MapperGraphConverter._handle_call_method``);
# every other method converts by the method rule (``rules/method_rules.has_method_rule``: a torch function of its
# name) or is rejected: an unknown method used to fall through to identity and silently DROP the op.
CONVERTIBLE_CALL_METHODS: frozenset[str] = frozenset({
    # shape/layout rewrites with dedicated mappers
    "view", "reshape", "flatten", "permute", "transpose",
    # reductions/arithmetic emitted as generic host ComputeOps
    "mean", "add", "__add__",
    # deliberately structural no-ops (metadata or handled at use sites,
    # e.g. ``expand`` feeding a constant-concat)
    "contiguous", "size", "dim", "unsqueeze", "squeeze", "expand",
})


CALL_SITE_WEIGHT_SHARING_REASON = "call-site weight sharing is not supported yet"
"""[D13] Each site would get its own Perceptron and an untied weight copy; shared banks are work item N7."""


def call_site_sharing(gm: fx.GraphModule, modules: Dict[str, nn.Module]) -> List[OpInfo]:
    """One refusal per node target whose state is also reached from another FX call site."""
    refusals = []
    for node, sites in shared_state_sites(gm, modules):
        kind = type(modules[node_target_str(node)]).__name__ if node.op == "call_module" else "Tensor"
        refusals.append(OpInfo(node.name, node.op, kind, reason=(
            f"{CALL_SITE_WEIGHT_SHARING_REASON} (work item N7): the state of {str(node.target)!r} "
            f"({kind}) is reached from {len(sites)} FX call sites ({', '.join(sites)}); "
            "conversion would give each site an untied copy"
        )))
    return refusals


class RepresentabilityAnalyzer:
    """Analyse an FX graph for mimarsinan representability (per packaging contract). ``foreign_modules``: the module
    targets outside the declared operator set (``symbolic.purity``), host ops of their own call no rule reads."""

    def __init__(
        self,
        graph_module: fx.GraphModule,
        packaging: PackagingContract = SPIKING_PACKAGING,
        foreign_modules: FrozenSet[str] = frozenset(),
    ):
        self.gm = graph_module
        self._packaging = packaging
        self._modules: Dict[str, nn.Module] = dict(graph_module.named_modules())
        self._foreign = foreign_modules

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

        report.unsupported_ops.extend(call_site_sharing(self.gm, self._modules))
        self._build_absorption_plan(report)
        report.unsupported_ops.extend(self._perceptron_conv_padding_refusals())

        if report.unsupported_ops:
            report.is_representable = False

        return report

    def _classify_module_node(
        self, node: fx.Node, report: RepresentabilityReport
    ) -> None:
        mod = self._modules.get(node_target_str(node))
        if mod is None:
            report.unsupported_ops.append(
                OpInfo(node.name, "call_module", str(node.target),
                       reason=f"Module '{node.target}' not found")
            )
            report.is_representable = False
            return

        mod_type = type(mod)

        if isinstance(mod, BATCHNORM_TYPES) and not has_running_statistics(mod):
            report.unsupported_ops.append(
                OpInfo(node.name, "call_module", mod_type.__name__,
                       reason="BatchNorm without running statistics (track_running_stats=False) "
                              "normalizes with batch statistics in eval mode; it is not a per-channel affine op")
            )
            return

        requirement = core_lowering_requirement(mod)
        report.supported_ops.append(OpInfo(node.name, "call_module", mod_type.__name__,
                                           reason=None if requirement is None else requirement.reason))

    def _classify_function_node(
        self, node: fx.Node, report: RepresentabilityReport
    ) -> None:
        fn = node.target
        fn_name = getattr(fn, "__name__", str(fn))
        report.supported_ops.append(
            OpInfo(node.name, "call_function", fn_name)
        )

    def _classify_method_node(
        self, node: fx.Node, report: RepresentabilityReport
    ) -> None:
        method_name = node_target_str(node)
        if method_name not in CONVERTIBLE_CALL_METHODS and not has_method_rule(node):
            report.unsupported_ops.append(
                OpInfo(
                    node.name, "call_method", method_name,
                    reason=(
                        f"Tensor method '.{method_name}(...)' has no conversion "
                        "rule (it would be silently dropped). Use a supported "
                        "module/functional form, or add an explicit rule."
                    ),
                )
            )
            return
        report.supported_ops.append(
            OpInfo(node.name, "call_method", method_name)
        )

    def _build_absorption_plan(self, report: RepresentabilityReport) -> None:
        """Fold BN / activation nodes into the weight op whose single-consumer chain they lie on."""
        targets: Dict[str, str] = {}
        for node in self.gm.graph.nodes:
            module = self._call_module(node, nn.Module)
            if module is not None and type(module) in _NEURAL_CORE_MODULES and core_lowering_requirement(module) is None:
                for member in self._foldable_chain(node):
                    targets[member.name] = node.name
        for node in self.gm.graph.nodes:
            if node.name in targets:
                report.absorption_plan[node.name] = f"absorbed_into:{targets[node.name]}"

    def _perceptron_conv_padding_refusals(self) -> List[OpInfo]:
        """[E273.R] A conv lowered to a perceptron pads with zeros only; any other padding is refused by name."""
        refusals = []
        for node in self.gm.graph.nodes:
            conv = self._call_module(node, (nn.Conv1d, nn.Conv2d))
            reason = perceptron_conv_padding(conv)[1] if conv is not None and conv.groups == 1 else None
            chain = self._foldable_chain(node) if reason is not None else []
            absorbs = any(isinstance(self._call_module(m, nn.Module), ABSORBABLE_ACTIVATIONS) for m in chain)
            if reason is not None and conv is not None and lowers_to_perceptron(absorbs, self._packaging):
                refusals.append(OpInfo(node.name, "call_module", type(conv).__name__, reason=(
                    f"{type(conv).__name__} (FX node {node.name!r}): {reason}; the perceptron conv mapping "
                    "implements numeric zero padding only (E273.R)")))
        return refusals

    def _foldable_chain(self, weight_node: fx.Node) -> List[fx.Node]:
        """``Identity* [BN] Identity* [act] Identity*`` after the weight op; every member but the last has one user."""
        weight = self._modules[node_target_str(weight_node)]
        shape = node_output_shape(weight_node)
        rank = len(shape) if shape is not None else None
        chain: List[fx.Node] = []
        cur, folded_bn, folded_act = weight_node, False, False
        while len(cur.users) == 1:
            user = next(iter(cur.users))
            mod = self._call_module(user, (nn.Identity, *BATCHNORM_TYPES, *ABSORBABLE_ACTIVATIONS))
            if mod is None or not user.args or user.args[0] is not cur:
                break
            if isinstance(mod, BATCHNORM_TYPES):
                if folded_bn or folded_act or not batchnorm_folds_into(mod, weight, rank):
                    break
                folded_bn = True
            elif isinstance(mod, ABSORBABLE_ACTIVATIONS):
                if folded_act or not self._packaging.absorb_activation or not activation_folds_exactly(mod):
                    break
                folded_act = True
            chain.append(user)
            cur = user
        return chain

    def _call_module(self, node: fx.Node, types) -> Optional[nn.Module]:
        if node.op != "call_module" or node_target_str(node) in self._foreign:
            return None
        mod = self._modules.get(node_target_str(node))
        return mod if isinstance(mod, types) else None
