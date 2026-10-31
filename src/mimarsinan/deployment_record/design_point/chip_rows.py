"""Chip-row declarations of an instance: the core types of a row, each with its role and write technology, and the
refusals by name of a declaration whose capability this tree does not build.

A row declares its core types (role ``any``, ``static`` or ``operand``; geometry; the physics profile of its write
technology; an ``operand`` declaration on an operand core type). Two rules refuse a row by name before anything is
built or priced:

- ``per_core_type_write_technology``: the core types name more than one write technology and some core type
  declares no ``write_physics``. [E-6, targeted pass] Where every core type declares its ``write_physics`` (N2b,
  ``quantities/write_technology.py``; a row's declaration, e.g. ``tests/fixtures/design_point/
  hybrid_neurram_sram_cim.json``), the instance prices each technology's share of the census by the write law and
  judges each core type's wear (``design_point/wear.technology_wear``), so the row is built; otherwise no technology
  is named for some core type's cells, and the row is refused by name.
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
    """One declared core type in the instance's form (role, geometry, count, write technology, operand; [U-CR C-3] the
    weight-cell block where it states one, so a geometry in weights reads with its cells)."""
    role = str(core_type.get("role", "any"))
    if role not in ROLES:
        raise RowDeclarationError(f"core type role {role!r} is not one of {ROLES}")
    missing = [key for key in ("max_axons", "max_neurons") if key not in core_type]
    if missing:
        raise RowDeclarationError(f"core type {dict(core_type)} declares no {missing}")
    out = {"role": role, "max_axons": int(core_type["max_axons"]), "max_neurons": int(core_type["max_neurons"]),
           "count": None if core_type.get("count") is None else int(core_type["count"]),
           "write_technology": core_type.get("write_technology")}
    for key in ("operand", "write_granularity", "source", "cells_per_weight", "weight_cell_rows"):
        if key in core_type:
            out[key] = core_type[key]
    return out


def declaration_refusals(core_types: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    """The refusals by name of a row's core-type declaration (empty: the row can be built and priced)."""
    out: List[Dict[str, str]] = []
    technologies = sorted({str(ct["write_technology"]) for ct in core_types if ct.get("write_technology")})
    if len(technologies) > 1 and not all(ct.get("write_physics") for ct in core_types):
        roles = ", ".join(f"{ct.get('role', 'any')}: {ct.get('write_technology')}" for ct in core_types)
        out.append({"name": PER_CORE_TYPE_WRITE_TECHNOLOGY, "stage": "declare",
                    "reason": (f"the core types declare {len(technologies)} write technologies ({roles}) and "
                               "not every core type declares its write_physics, so the write law cannot name the "
                               "technology each core type's cells are written under (a row's declaration states "
                               "write_physics per core type)")})
    if any(ct.get(OPERAND_FIELD) is not None for ct in core_types) and OPERAND_FIELD not in CORE_TYPE_SCHEMA.fields:
        out.append({"name": OPERAND_CORE_TYPE, "stage": "declare",
                    "reason": ("a core type declares an operand core, and the registry's core-type schema declares "
                               f"no {OPERAND_FIELD!r} field: dynamic-operand cores are not integrated on this tree, so "
                               f"products of two activations stay host operations ({requirement_of('dmm').reason})")})
    return out
