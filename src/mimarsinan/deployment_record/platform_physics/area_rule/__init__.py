"""[U-AREA] The chip area rule from published silicon per technology type (design revision 3, FT/uarea/r3).

One rule for every type, entered three ways: an anchor (a description's own published core, scaled off its
geometry, word width and size), a type rule (a description priced from its type's table), or complete declared
terms (R5's own path). Every published figure is data in ``data/`` with its quote, location and citation key:
``registry`` loads it and ``view`` re-derives it with a published chip held out; ``rule`` holds the equations of a
cell, a converter and the outside part, ``core`` the one core equation and its count form (``core.equation``, which
the type rule, the anchors and the published points read), ``typerule`` the type rule; ``anchor`` the anchors;
``declaration`` the description's block and its judge; ``pricing`` the pricer's rule path
(``cost/absolute/metrics.price_area`` asks it first); ``published`` the held-out estimate of the published points
through the same core equation (``parity``, ``decomposition``); ``check`` the registry's evidence check; ``units`` the
named units.
"""

from mimarsinan.deployment_record.platform_physics.area_rule import (
    anchor, check, core, declaration, evidence, pricing, published, registry, rule, typerule, units, view)
from mimarsinan.deployment_record.platform_physics.area_rule.core import arguments as _arguments, door as _door
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal

_door.install(_arguments)  # the door's table, once every module of the rule is loaded (its judges read them all)

__all__ = ["AreaRuleRefusal", "anchor", "check", "core", "declaration", "evidence", "pricing", "published",
           "registry", "rule", "typerule", "units", "view"]
