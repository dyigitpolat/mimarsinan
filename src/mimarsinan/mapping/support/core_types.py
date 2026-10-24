"""Core types with their own weight bits and the layer-to-type rule (WI-5g).

A core type of ``cores`` may declare ``weight_bits``, the width of its cells (absent: the platform's ``weight_bits``).
``layer_core_types`` is the layer-to-type rule: a list of ``{"layers": [glob, ...], "weight_bits": b}``; every
perceptron whose FX node name matches a glob of a rule (and every pool lane whose op does) runs on the core types of
width ``b``, the others on the types of the platform's width. A layer's weights quantize on its type's grid (Weight
Quantization, the IR, the law's stamp) and its cores pack only onto its type's cores. Without a width on any core type
and without the rule, nothing is bound and every program is the base's, byte for byte.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.lanes.ir.pool_cores import LANE_SOURCE_NAME
from mimarsinan.mapping.lanes.model_lanes import pool_lane_quantizers
from mimarsinan.mapping.packing.type_binding import CoreTypeBindingError
from mimarsinan.mapping.support.quantizer_state import perceptron_fx_names
from mimarsinan.transformations.quantization_bounds import (
    CORE_WEIGHT_BITS,
    LAYER_WEIGHT_BITS,
    layer_weight_bits,
    quantization_bounds,
)

LAYER_RULE_KEY = "layer_core_types"
TYPE_BITS_FIELD = "weight_bits"
_RULE_FIELDS = frozenset({"layers", "weight_bits"})


@dataclass(frozen=True)
class LayerRule:
    layers: Tuple[str, ...]
    weight_bits: int


def _bits(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise CoreTypeBindingError(f"{where} must be an integer weight width of at least 2 bits, got {value!r}")
    return int(value)


def _rules(declared: Any) -> Tuple[LayerRule, ...]:
    if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes)):
        raise CoreTypeBindingError(f"{LAYER_RULE_KEY} must be a list of rules, got {declared!r}")
    out: List[LayerRule] = []
    for i, rule in enumerate(declared):
        if not isinstance(rule, Mapping) or set(rule) != _RULE_FIELDS:
            raise CoreTypeBindingError(f"{LAYER_RULE_KEY}[{i}] must hold exactly {sorted(_RULE_FIELDS)}, got {rule!r}")
        layers = rule["layers"]
        if (not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)) or not layers
                or not all(isinstance(g, str) and g for g in layers)):
            raise CoreTypeBindingError(f"{LAYER_RULE_KEY}[{i}].layers must be a non-empty list of FX-name globs, "
                                       f"got {layers!r}")
        out.append(LayerRule(tuple(layers), _bits(rule["weight_bits"], f"{LAYER_RULE_KEY}[{i}].weight_bits")))
    return tuple(out)


@dataclass(frozen=True)
class CoreTypeBinding:
    """The chip's core-type widths (one per declared type) and the layer-to-type rule over the platform width."""

    default_bits: int
    type_bits: Tuple[int, ...]
    rules: Tuple[LayerRule, ...]

    @classmethod
    def from_config(cls, platform: Mapping[str, Any]) -> Optional["CoreTypeBinding"]:
        """The declared binding, or ``None`` when no core type declares a width and no rule is declared."""
        cores = list(platform.get("cores") or ())
        declared = platform.get(LAYER_RULE_KEY)
        if declared is None and not any(isinstance(c, Mapping) and TYPE_BITS_FIELD in c for c in cores):
            return None
        if platform.get("weight_bits") is None:
            raise CoreTypeBindingError("core types with their own weight_bits need the platform weight_bits (the "
                                       "width of the layers no rule selects)")
        default = _bits(platform["weight_bits"], "weight_bits")
        widths = tuple(_bits(c.get(TYPE_BITS_FIELD, default), f"cores[{i}].weight_bits") for i, c in enumerate(cores))
        rules = _rules(declared if declared is not None else ())
        orphans = sorted({r.weight_bits for r in rules} - set(widths))
        if orphans:
            raise CoreTypeBindingError(f"{LAYER_RULE_KEY} puts layers on weight widths {orphans}, which no core type "
                                       f"declares (types: {sorted(set(widths))})")
        return cls(default_bits=default, type_bits=widths, rules=rules)

    def bits_of(self, name: Optional[str]) -> int:
        """The width of the layer (or pool lane) named ``name``: the rule selecting it, else the platform width."""
        if name is not None:
            for rule in self.rules:
                if any(fnmatchcase(name, glob) for glob in rule.layers):
                    return rule.weight_bits
        return self.default_bits

    def layer_bits(self, names: Iterable[str], lanes: Iterable[str] = ()) -> Dict[str, int]:
        """{layer FX name: width}; refused by name: a glob no layer or pool lane matches (a stale or foreign rule),
        a layer two rules of different widths select, and a layer whose width no core type declares."""
        layers = list(names)
        known = layers + list(lanes)
        for i, rule in enumerate(self.rules):
            for glob in rule.layers:
                if not any(fnmatchcase(n, glob) for n in known):
                    raise CoreTypeBindingError(f"{LAYER_RULE_KEY}[{i}] glob {glob!r} matches no layer of the network")
        out: Dict[str, int] = {}
        for name in layers:
            widths = {r.weight_bits for r in self.rules if any(fnmatchcase(name, g) for g in r.layers)}
            if len(widths) > 1:
                raise CoreTypeBindingError(f"layer {name!r} is selected by rules of weight widths {sorted(widths)}")
            out[name] = self.bits_of(name)
        unplaced = sorted({b for b in out.values()} - set(self.type_bits))
        if unplaced:
            raise CoreTypeBindingError(f"layers of weight width {unplaced} (the platform weight_bits) have no core "
                                       f"type of that width (types: {sorted(set(self.type_bits))})")
        return out

    def typed_cores(self, cores: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """``cores`` with each type's width written out (the packer binds on it)."""
        return [{**dict(core), TYPE_BITS_FIELD: bits} for core, bits in zip(cores, self.type_bits)]


def binding_of(platform: Mapping[str, Any]) -> Optional[CoreTypeBinding]:
    return CoreTypeBinding.from_config(platform)


def typed_cores_config(platform: Mapping[str, Any]) -> Any:
    """The packer's core types: the declared list itself when nothing binds, else each with its width."""
    binding = binding_of(platform)
    return platform["cores"] if binding is None else binding.typed_cores(platform["cores"])


def refuse_unbuilt_domain(platform: Mapping[str, Any], *, value_domain: bool, weight_quantization: bool) -> None:
    """Typed widths are built for value-domain cores with a weight grid; elsewhere they are refused by name."""
    if binding_of(platform) is None:
        return
    if not value_domain:
        raise CoreTypeBindingError("core types with their own weight_bits are built for value-domain cores "
                                   "(core_semantics mvm); spiking cores are not")
    if not weight_quantization:
        raise CoreTypeBindingError("core types with their own weight_bits need weight quantization: float weights "
                                   "have no grid")


def install_layer_weight_bits(model, platform: Mapping[str, Any]) -> Optional[Dict[str, int]]:
    """Stamp every perceptron's width from the rule (Weight Quantization reads it); ``None`` when nothing binds.
    The rule's globs may also name an installed pool lane (its reduction cores' type). A perceptron already stamped
    with another width is refused by name (its weights sit on another grid)."""
    binding = binding_of(platform)
    if binding is None:
        return None
    names = perceptron_fx_names(model.get_mapper_repr())
    widths = binding.layer_bits(names, [name for name, _ in pool_lane_quantizers(model.get_mapper_repr())])
    for name, perceptron in names.items():
        stamped = getattr(perceptron, LAYER_WEIGHT_BITS, None)
        if stamped is not None and int(stamped) != widths[name]:
            raise CoreTypeBindingError(f"layer {name!r} was quantized at {stamped} bits; the rule now gives "
                                       f"{widths[name]} (a stale Weight Quantization)")
        setattr(perceptron, LAYER_WEIGHT_BITS, widths[name])
    return widths


def bind_core_types(ir_graph, model, platform: Mapping[str, Any]) -> int:
    """Bind every IR core and weight bank to its layer's width (a pool lane's core by its op's FX name, a core no
    layer owns to the platform width); the number of cores bound, 0 when nothing binds."""
    binding = binding_of(platform)
    if binding is None:
        return 0
    perceptrons = list(model.get_perceptrons())

    def width(index, name=None) -> int:
        if index is None:
            return binding.bits_of(name)
        perceptron = perceptrons[int(index)]
        if getattr(perceptron, LAYER_WEIGHT_BITS, None) is None:
            raise CoreTypeBindingError(f"perceptron {getattr(perceptron, 'name', index)!r} carries no declared width "
                                       f"(Weight Quantization did not read {LAYER_RULE_KEY})")
        return layer_weight_bits(perceptron, binding.default_bits)

    for bank in (getattr(ir_graph, "weight_banks", None) or {}).values():
        setattr(bank, CORE_WEIGHT_BITS, width(getattr(bank, "perceptron_index", None)))
    bound = 0
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            setattr(node, CORE_WEIGHT_BITS, width(node.perceptron_index, getattr(node, LANE_SOURCE_NAME, None)))
            bound += 1
    return bound


def check_core_codes(ir_graph) -> int:
    """Every bound core's integer weight codes (its rows other than bias rows) lie on its type's grid; refused by
    name otherwise. The number of cores checked."""
    checked = 0
    for node in ir_graph.nodes:
        bits = getattr(node, CORE_WEIGHT_BITS, None)
        if not isinstance(node, NeuralCore) or bits is None:
            continue
        q_min, q_max = quantization_bounds(int(bits))
        matrix = np.asarray(node.get_core_matrix(ir_graph))
        on = np.array([source.is_always_on() for source in node.input_sources.flatten()], dtype=bool)
        data = matrix[~on, :] if on.size == matrix.shape[0] else matrix
        if data.size and (float(data.min()) < q_min or float(data.max()) > q_max):
            raise CoreTypeBindingError(f"core {node.name!r} binds to {bits}-bit cells but holds weight codes in "
                                       f"[{data.min()}, {data.max()}] (its grid is [{q_min}, {q_max}])")
        checked += 1
    return checked
