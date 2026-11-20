"""[U-AREA] Test helpers: a description as every price of it is read, and the rule's price at the rule level.

- ``store_declared`` / ``priced_physics``: a description read only with an external component declared (CL2: the
  SRAM-CIM macro's backing store) prices nothing until a run declares the component's constants
  (``context.external_refusals``, the gate of every pricer, which the area rule leaves unchanged). The area reads none
  of these constants, so a test declares them as test values.
- ``rule_price``: the chip area the description's area_rule block gives for core types ``(count, rows, cols)`` at the
  rule level (``typerule.price_block_types`` on the shipped registry), the expected value of a test that reads the
  area through a toolchain site (the tests the unit moved state their R5 value beside it in a comment).
"""

from __future__ import annotations

from mimarsinan.deployment_record.platform_physics import PHYSICS_CONSTANTS, apply_overrides, get_platform_physics
from mimarsinan.deployment_record.platform_physics.area_rule import registry, rule, typerule


def store_declared(physics):
    missing = [k for c in physics.undeclared_external() for k in physics.missing(c.constants)]
    if not missing:
        return physics
    return apply_overrides(physics, {k: {"nominal": 1.0, "unit": PHYSICS_CONSTANTS[k].display_unit,
                                         "evidence_kind": "estimated", "derivation": "a test store (test)",
                                         "note": "a test store (test)"}
                                     for k in missing})


def priced_physics(name):
    """The shipped description ``name`` with its external components declared as test values."""
    return store_declared(get_platform_physics(name))


def rule_price(name, types, weight_bits=None):
    """The rule-level AreaPrice of description ``name`` over core types ``(count, rows, cols)``."""
    physics = get_platform_physics(name)
    census = tuple((float(n), rule.CensusCore(r, c, weight_bits)) for n, r, c in types)
    cell_bits = physics.validity.geometry.cell_bits if physics.validity.geometry is not None else None
    return typerule.price_block_types(registry.load_registry(), physics.area_rule, census, cell_bits)
