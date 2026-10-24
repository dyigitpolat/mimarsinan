"""N2 lowering: every Contraction of the operand form onto the declared operand core type (design §2.1-§2.5).

Runs on the mapper graph after Torch Mapping, so every calibration sees the bank perceptrons. With no operand core
type declared nothing changes. G1's converter is not edited: each lowered product's ``ComputeOpMapper`` is
replaced in place, its consumers re-pointed, its absorbable followers folded into its law.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Tuple, cast

import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.operand.followers import absorbable_follower
from mimarsinan.mapping.mappers.operand.gather_chain import gather_chain, mapper_output_shapes
from mimarsinan.mapping.mappers.operand.operand_mapper import OperandContractionMapper
from mimarsinan.mapping.support.contraction import Contraction
from mimarsinan.mapping.support.operand_form import (
    AT_STATIONARY,
    B_STATIONARY,
    OPERAND_FORM_OUTSIDE,
    Lowering,
    OperandForm,
    operand_rule_form,
    stationary_operand,
)
from mimarsinan.mapping.support.operand_price import lowering_price
from mimarsinan.mapping.support.period_census import OPERAND_WRITE_SOURCES, READOUT_RANGE_RULES
from mimarsinan.mapping.support.residual_merge import _rebind_source

__all__ = [
    "AT_STATIONARY", "B_STATIONARY", "OPERAND_FORM_OUTSIDE", "Lowering", "LoweringReport", "OperandDeclaration",
    "OperandDeclarationError", "OperandForm", "PER_CONSUMER_GROUP", "PER_PERCEPTRON", "READOUT_RANGE_RULES",
    "WRITE_SOURCES", "bank_resnaps", "lower_contractions", "lowering_price", "operand_declaration",
    "operand_rule_form", "product_choice", "stationary_operand",
]

WRITE_SOURCES = OPERAND_WRITE_SOURCES
PER_PERCEPTRON, PER_CONSUMER_GROUP = READOUT_RANGE_RULES


class OperandDeclarationError(ToolchainRefusal, ValueError):
    """An operand core-type declaration the lowering cannot read as one payload width and one write source."""


@dataclass(frozen=True)
class OperandDeclaration:
    payload_bits: int
    write_source: str
    activation_bits: Optional[int]
    readout_ranges: str
    readout_law: bool

    @property
    def consumer_groups(self) -> bool:
        return self.readout_ranges == PER_CONSUMER_GROUP

    @property
    def gridless(self) -> bool:
        return not self.readout_law and self.activation_bits is None

    @property
    def choice_is_cost_only(self) -> bool:
        """The declaration's half of design §2.2's condition: no grid at all, or consumer groups at one width.
        The product's half (both operands core-fed, ``product_choice``) is read where the product is lowered."""
        return self.gridless or (self.consumer_groups and self.payload_bits == self.activation_bits)


def operand_declaration(platform: Mapping[str, Any]) -> Optional[OperandDeclaration]:
    """The platform's operand core type, or None when no core type declares ``operand``."""
    declared = [dict(ct["operand"]) for ct in platform.get("cores") or () if ct.get("operand") is not None]
    if not declared:
        return None
    if any("payload_bits" not in d for d in declared):
        raise OperandDeclarationError("cores[i].operand needs payload_bits: the payload grid width")
    widths = sorted({int(d["payload_bits"]) for d in declared})
    sources = sorted({str(d.get("write_source", "host")) for d in declared})
    if len(widths) > 1 or len(sources) > 1:
        raise OperandDeclarationError(
            f"the operand core types declare payload_bits {widths} and write_source {sources}: one payload width "
            "and one write source per program")
    if sources[0] not in WRITE_SOURCES:
        raise OperandDeclarationError(f"cores[i].operand.write_source {sources[0]!r} is not one of {WRITE_SOURCES}")
    rule = str(platform.get("readout_ranges") or PER_PERCEPTRON)
    if rule not in READOUT_RANGE_RULES:
        raise OperandDeclarationError(f"readout_ranges {rule!r} is not one of {READOUT_RANGE_RULES}")
    bits = platform.get("activation_bits")
    return OperandDeclaration(payload_bits=widths[0], write_source=sources[0],
                              activation_bits=None if bits is None else int(bits), readout_ranges=rule,
                              readout_law=platform.get("readout_law") is not None)


@dataclass(frozen=True)
class LoweredProduct:
    name: str
    stationary: str
    reason: str
    banks: int
    cells: int
    follower_kinds: Tuple[str, ...]


@dataclass(frozen=True)
class HostProduct:
    name: str
    reason: str


@dataclass(frozen=True)
class LoweringReport:
    lowered: Tuple[LoweredProduct, ...] = ()
    host: Tuple[HostProduct, ...] = ()


def _replace(model_repr, old, new) -> None:
    for consumer in model_repr.consumer_map().get(id(old), []):
        _rebind_source(consumer, old, new)
    if model_repr.output_layer_mapper is old:
        model_repr.output_layer_mapper = new
    model_repr._exec_order = model_repr._deps = model_repr._consumer_count = None


def _core_fed(root) -> bool:
    return getattr(root, "perceptron", None) is not None or isinstance(root, OperandContractionMapper)


def product_choice(declaration: OperandDeclaration, roots) -> Tuple[bool, str]:
    """(whether the stationary choice of one product is cost-only, and if not why): design §2.2's condition on
    the declaration and on the product. A host-fed operand streams on its entry grid but would be written on a
    re-snap grid calibrated by another seam, so its role changes its codes."""
    if declaration.gridless:
        return True, ""
    if not declaration.consumer_groups:
        return False, "no consumer groups: the role decides which operand is re-snapped onto a payload grid"
    if declaration.payload_bits != declaration.activation_bits:
        return False, "payload_bits differs from activation_bits: the role decides which operand takes the payload grid"
    host = [name for name, root in zip("AB", roots) if not _core_fed(root)]
    if host:
        return False, (f"operand {' and '.join(host)} is a host value: streamed on its entry grid, written on a "
                       "re-snap grid of another calibration seam, so its role changes its codes")
    return True, ""


def bank_resnaps(declaration: OperandDeclaration, stationary_root) -> Optional[str]:
    """Why a bank re-snaps its payload onto a ``payload_bits`` grid in the write, or None where the write copies
    its producer's codes: no consumer groups, a host-produced stationary operand, or an operand product's output
    (its readout lies on the ``activation_bits`` grid) read at another payload width."""
    if not declaration.consumer_groups:
        return "no consumer groups"
    if not _core_fed(stationary_root):
        return "a host-produced stationary operand"
    if isinstance(stationary_root, OperandContractionMapper) and not declaration.gridless \
            and declaration.payload_bits != declaration.activation_bits:
        return "an operand product's output read at another payload width"
    return None


def _lower_one(node: ComputeOpMapper, form: OperandForm, lowering: Lowering, reason: str, chains, shapes,
               declaration: OperandDeclaration) -> OperandContractionMapper:
    roots = []
    for root, _index in chains:
        if root not in roots:
            roots.append(root)
    stationary_role = 1 if lowering.stationary == B_STATIONARY else 0
    resnap = bank_resnaps(declaration, chains[stationary_role][0])
    if resnap is not None and declaration.write_source == "on_chip" and not declaration.gridless:
        raise OperandDeclarationError(
            f"write_source on_chip on {node.name}: its banks re-snap their payload ({resnap}), so an on-chip "
            "write-back of the producer's codes computes another function than the certified host-sourced program; "
            "on_chip is priced only where every payload write copies codes (readout_ranges per_consumer_group, "
            "each stationary operand core-fed at payload_bits)")
    return OperandContractionMapper(
        name=str(node.name or "product"), lowering=lowering, reason=reason, roots=roots,
        operand_roots=(roots.index(chains[0][0]), roots.index(chains[1][0])),
        operand_maps=(chains[0][1], chains[1][1]), root_shapes=[shapes[id(r)] for r in roots],
        operand_shapes=[tuple(s) for s in cast(Contraction, node.module).operand_shapes],
        output_shape=tuple(node.output_shape or ()),
        payload_bits=declaration.payload_bits, resnap=[resnap is not None] * form.banks,
        host_fed=(not _core_fed(chains[0][0]), not _core_fed(chains[1][0])),
    )


def _absorb_followers(model_repr, mapper: OperandContractionMapper) -> None:
    while True:
        readers = model_repr.consumer_map().get(id(mapper), [])
        follower = absorbable_follower(readers[0]) if len(readers) == 1 else None
        if follower is None or model_repr.output_layer_mapper is mapper:
            return
        mapper.followers.append(follower)
        _replace(model_repr, readers[0], mapper)


def lower_contractions(model_repr, platform: Mapping[str, Any], *, price: Optional[Callable[[Lowering], float]] = None,
                       owner: Optional[nn.Module] = None, input_shape=None) -> LoweringReport:
    """Lower every Contraction of the operand form; the others keep G1's host op and are named. ``owner`` (the
    flow) registers each new mapper as a submodule so device and dtype moves reach its quantizers."""
    declaration = operand_declaration(platform)
    if declaration is None:
        return LoweringReport()
    price = price or lowering_price(platform)
    shapes = mapper_output_shapes(model_repr, input_shape)
    lowered, host = [], []
    for node in list(model_repr.execution_order()):
        if not (isinstance(node, ComputeOpMapper) and isinstance(node.module, Contraction)):
            continue
        form = operand_rule_form(node.module)
        if form is None or len(node.get_source_mappers()) != 2 or node.output_shape is None:
            host.append(HostProduct(str(node.name), OPERAND_FORM_OUTSIDE))
            continue
        chains = [gather_chain(source, shapes) for source in node.get_source_mappers()]
        cost_only, why = product_choice(declaration, [root for root, _index in chains])
        lowering, reason = stationary_operand(form, price, cost_only, why)
        mapper = _lower_one(node, form, lowering, reason, chains, shapes, declaration)
        shapes[id(mapper)] = shapes[id(node)]
        _replace(model_repr, node, mapper)
        _absorb_followers(model_repr, mapper)
        if owner is not None:
            owner.add_module(f"operand_product_{len(lowered)}", mapper)
        lowered.append(LoweredProduct(mapper.name, lowering.stationary, reason, form.banks, lowering.cells,
                                      mapper.follower_kinds))
    model_repr._ensure_exec_graph()
    return LoweringReport(tuple(lowered), tuple(host))
