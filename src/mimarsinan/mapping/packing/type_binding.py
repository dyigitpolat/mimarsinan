"""Core-type binding of the packers (WI-5g): a soft core whose layer declares a weight width packs only onto hard
cores of a core type of that width.

A core type with its own ``weight_bits`` and a soft core bound to one (``CORE_WEIGHT_BITS``, set from the layer-to-
type rule ``layer_core_types``) bind: the canonical predicate refuses a hard core of another width, and the runtime
packer packs each width's soft cores from that width's unused cores only. Without a width on any soft or hard core
nothing changes.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.transformations.quantization_bounds import CORE_WEIGHT_BITS


class CoreTypeBindingError(ToolchainRefusal, ValueError):
    """A layer-to-core-type binding the declaration, the program or the chip cannot honour (WI-5g)."""


def bound_bits(holder: Any) -> Optional[int]:
    """The weight width a soft or hard core binds to, or ``None`` (unbound)."""
    bits = getattr(holder, CORE_WEIGHT_BITS, None)
    return None if bits is None else int(bits)


def binding_admits(softcore: Any, hardcore: Any) -> bool:
    """A bound hard core takes only soft cores of its width; an unbound one takes any."""
    bits = bound_bits(hardcore)
    return bits is None or bound_bits(softcore) == bits


def binding_classes(softcores: Sequence[Any], hardcores: Sequence[Any]) -> Optional[List[int]]:
    """The widths to pack one after another, widest first, or ``None`` when nothing binds.

    Refused by name: a bound soft core beside an unbound unused hard core or the reverse (the chip's widths are
    declared per core type or for none). A width whose unused cores are spent packs from an empty pool, so the
    packer's own capacity refusal (which scheduling reads) speaks, as for an unbound chip.
    """
    soft = [bound_bits(core) for core in softcores]
    hard = [bound_bits(core) for core in hardcores]
    if all(bits is None for bits in soft) and all(bits is None for bits in hard):
        return None
    if any(bits is None for bits in soft) or any(bits is None for bits in hard):
        raise CoreTypeBindingError(
            f"core-type binding: {sum(b is None for b in soft)} of {len(soft)} soft cores and "
            f"{sum(b is None for b in hard)} of {len(hard)} unused hard cores carry no weight width; a chip whose "
            f"core types declare weight_bits binds every layer (fill each type's width, layer_core_types)")
    return sorted({int(bits) for bits in soft if bits is not None}, reverse=True)


def refuse_typed_multi_pass(label: str, passes: int, cores: Sequence[Any]) -> None:
    """A segment of more than one pass on cores bound to their type's width refuses by name: the pass planner plans
    shape-only specs against one core budget, blind to each type's width (moved from ``hybrid_build_scheduled``)."""
    if passes > 1 and any(bound_bits(core) is not None for core in cores):
        raise CoreTypeBindingError(
            f"segment {label!r} needs {passes} passes on core types with their own weight_bits: "
            f"multi-pass packing there is not built (the pass planner plans shape-only specs against one core "
            f"budget, blind to each type's width, WI-5g); declare enough cores of each width for one pass")
