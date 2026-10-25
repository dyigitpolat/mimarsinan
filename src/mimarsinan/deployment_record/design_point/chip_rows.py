"""Chip-row declarations of an instance: the core types of a row, each with its role and write technology, and the
refusals by name of a declaration whose capability this tree does not build.

A row declares its core types (role ``any``, ``static`` or ``operand``; geometry; the physics profile of its write
technology; an ``operand`` declaration on an operand core type). Two rules refuse a row by name before anything is
built or priced:

- ``per_core_type_write_technology``: the core types name more than one write technology. The instance prices one
  physics per chip (one ``platform_physics_profile`` per priced point); the write law's per-core-type technologies
  (N2b: a core type's ``write_physics``, ``quantities/write_technology.py``) are not read by the instance yet, so
  such a row is refused by name until the instance declares its core types' ``write_physics``.
- ``operand_core_type``: a core type declares an operand core while the registry's core-type schema declares no
  ``operand`` field: dynamic-operand cores are not integrated, so products of two activations stay host operations
  (the requirement row of ``dmm``). The rule reads the registry, so it lifts when the capability registers its key.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.config_schema.registry.entries_write import CORE_TYPE_SCHEMA
from mimarsinan.mapping.capability.requirements import requirement_of

PER_CORE_TYPE_WRITE_TECHNOLOGY = "per_core_type_write_technology"
OPERAND_CORE_TYPE = "operand_core_type"
#: The core-type field a dynamic-operand core type is declared by.
OPERAND_FIELD = "operand"
ROLES = ("any", "static", "operand")


class RowDeclarationError(ToolchainRefusal, ValueError):
    """A core-type declaration of a row the instance cannot read (an unknown role, a missing geometry field)."""


def core_type_record(core_type: Mapping[str, Any]) -> Dict[str, Any]:
    """One declared core type in the instance's form (role, geometry, count, write technology, operand)."""
    role = str(core_type.get("role", "any"))
    if role not in ROLES:
        raise RowDeclarationError(f"core type role {role!r} is not one of {ROLES}")
    missing = [key for key in ("max_axons", "max_neurons") if key not in core_type]
    if missing:
        raise RowDeclarationError(f"core type {dict(core_type)} declares no {missing}")
    out = {"role": role, "max_axons": int(core_type["max_axons"]), "max_neurons": int(core_type["max_neurons"]),
           "count": None if core_type.get("count") is None else int(core_type["count"]),
           "write_technology": core_type.get("write_technology")}
    for key in ("operand", "write_granularity", "source"):
        if key in core_type:
            out[key] = core_type[key]
    return out


def declaration_refusals(core_types: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    """The refusals by name of a row's core-type declaration (empty: the row can be built and priced)."""
    out: List[Dict[str, str]] = []
    technologies = sorted({str(ct["write_technology"]) for ct in core_types if ct.get("write_technology")})
    if len(technologies) > 1:
        roles = ", ".join(f"{ct.get('role', 'any')}: {ct.get('write_technology')}" for ct in core_types)
        out.append({"name": PER_CORE_TYPE_WRITE_TECHNOLOGY, "stage": "declare",
                    "reason": (f"the core types declare {len(technologies)} write technologies ({roles}); the "
                               "instance prices one physics per chip (one platform_physics_profile per priced "
                               "point) and does not read the write law's per-core-type write_physics yet, so a "
                               "chip with a write technology per core type is not priced here")})
    if any(ct.get(OPERAND_FIELD) is not None for ct in core_types) and OPERAND_FIELD not in CORE_TYPE_SCHEMA.fields:
        out.append({"name": OPERAND_CORE_TYPE, "stage": "declare",
                    "reason": ("a core type declares an operand core, and the registry's core-type schema declares "
                               f"no {OPERAND_FIELD!r} field: dynamic-operand cores are not integrated on this tree, so "
                               f"products of two activations stay host operations ({requirement_of('dmm').reason})")})
    return out
