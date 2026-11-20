"""[U-AREA] The judges of one declared field (moved from core/fields.py in the fourth delta round for the module budget;
core/fields.py re-exports every public name): a quantity of a kind, a text, a flag, a name the registry holds, an
expression over the evidence, a citation, and the fields that read another (``via``: judged under another holder's name,
walked by the declared domain or read by its own reader).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from mimarsinan.deployment_record.platform_physics.area_components import FIXED
from mimarsinan.deployment_record.platform_physics.area_rule import units as u
from mimarsinan.deployment_record.platform_physics.area_rule.core import door as _door, equation as core_rule
from mimarsinan.deployment_record.platform_physics.area_rule.core.judge import COUNT, POSITIVE, Kind, judged
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal
from mimarsinan.deployment_record.platform_physics.area_rule.rule import AREA_RULE_AXES, AreaFlag

Spec = Callable[[Any, str, Any, Any], None]
#: The census's place in the rule's words.
CENSUS = f"{AREA_RULE_AXES[u.FIRST]}'s census"


def quantity(kind: Kind, absent: bool = False) -> Spec:
    """A quantity of its kind (absent where ``absent``); its kind stated on the judge, so a holder's judge calls the one
    judge with the field's words made only on a refusal (core/fields.judged_holder)."""
    def judge(registry: Any, field: str, value: Any, holder: Any) -> None:
        if not (absent and value is None):
            judged(field, value, kind)
    setattr(judge, "kind", kind)
    setattr(judge, "absent", absent)
    return judge


def _shape(test: Callable[[Any], bool], words: str, absent: bool = False) -> Spec:
    def judge(registry: Any, field: str, value: Any, holder: Any) -> None:
        if not (absent and value is None) and not test(value):
            raise AreaRuleRefusal(f"{field} {value!r} is not {words}")
    return judge


def text(absent: bool = False) -> Spec:
    return _shape(lambda v: isinstance(v, str), "a text", absent)


def texts(absent: bool = False) -> Spec:
    return _shape(lambda v: isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v), "a list of texts",
                  absent)


def flag(absent: bool = False) -> Spec:
    return _shape(lambda v: isinstance(v, bool), "true or false", absent)


def one_of(choices: Any) -> Spec:
    return _shape(lambda v: isinstance(v, str) and v in tuple(choices), f"one of {list(choices)}")


def name(table: str) -> Spec:
    """A name the registry looks up in its table (``table``: 'type', 'cell kind', 'converter kind', 'evidence id'),
    each lookup a named refusal for a name it does not hold."""
    def judge(registry: Any, field: str, value: Any, holder: Any) -> None:
        if value is None and table != "type":
            return
        {"type": registry.type_file, "cell kind": registry.cell_kind, "converter kind": registry.converter_kind,
         "evidence id": registry.evidence.entry}[table](value)
    return judge


def expression(kind: Kind, cited_as: Any = None) -> Spec:
    """An expression over the evidence evaluated to its kind (core.equation.cited: a figure, a cell and a size cited by
    id; a published point's counts and areas over the registry)."""
    def judge(registry: Any, field: str, value: Any, holder: Any) -> None:
        core_rule.cited(getattr(registry, "base", registry), field, value, kind, cited_as)
    return judge


def citation(registry: Any, field: str, value: Any, holder: Any) -> None:
    """A citation: a text, a list of texts, or an object of texts to texts (or lists of texts), not empty (the fifth
    check's citation entries: a citation of no texts carried a number into an anchor's law)."""
    items = (list(value.values()) if isinstance(value, Mapping)
             else value if isinstance(value, (list, tuple)) else [value])
    leaves = [x for item in items for x in (item if isinstance(item, (list, tuple)) else [item])]
    if not value or not leaves or not all(isinstance(x, str) and x for x in leaves) or (
            isinstance(value, Mapping) and not all(isinstance(k, str) for k in value)):
        raise AreaRuleRefusal(f"{field} {value!r:.120} cites nothing: a citation is a text, a list of texts or an "
                              "object of texts")


def held_logic(registry: Any, field: str, value: Any, holder: Any) -> None:
    """mi-K2: absent, or a citation object with a non-empty citation (the published figures that fix the logic)."""
    if value is not None and not (isinstance(value, Mapping) and isinstance(value.get("citation"), (list, tuple))
                                  and value["citation"] and all(isinstance(i, str) for i in value["citation"])):
        raise AreaRuleRefusal(f"{field} {value!r} is not absent or a citation object with its citation (ids)")


def census_extent(registry: Any, field: str, value: Any, holder: Any) -> None:
    """A census core's rows or columns: a whole count, a finite number above 0 for a pooled census's mean core."""
    judged(field, value, POSITIVE if getattr(holder, "pooled", False) is True else COUNT)


def terms_form(registry: Any, field: str, value: Any, holder: Any) -> None:
    forms = sorted({registry.type_table(t)["scaled_terms"] for t in registry.types})
    if not isinstance(value, str) or value not in forms:
        raise AreaRuleRefusal(f"{field} {value!r} is not one of the registry's scaled terms {forms}")


def law_organization(registry: Any, field: str, value: Any, law: Any) -> None:
    """An anchor law's organization: the anchor's own core on its type's terms (core.equation.anchor_counts)."""
    core_rule.anchor_counts(registry, law.type, value)


def memory_beside(registry: Any, field: str, value: Any, holder: Any) -> None:
    if value is not None:
        if not isinstance(value, tuple) or len(value) != len(("bits", "type")):
            raise AreaRuleRefusal(f"{field} {value!r} is not a pair (bits, type)")
        judged(f"{field} bits", value[u.FIRST], POSITIVE)
        registry.type_file(value[u.SECOND])


def area_flags(registry: Any, field: str, value: Any, holder: Any) -> None:
    if not isinstance(value, tuple) or not all(isinstance(f, AreaFlag) for f in value):
        raise AreaRuleRefusal(f"{field} {value!r} is not a tuple of area flags")


def row_ratio(registry: Any, field: str, value: Any, row: Any) -> None:
    """A parity row's ratio: a size above 0, stated by every priced row (one without a refusal: the fifth check's
    MA-C3, strict_form divides by it), absent where the row is refused."""
    if value is None and getattr(row, "refused", None) is None:
        raise AreaRuleRefusal(f"{field}: a priced row (no refusal) states its ratio")
    if value is not None:
        judged(field, value, POSITIVE)


def via(holder_name: str, walked: Any = True, of: Any = None, every: bool = False) -> Spec:
    """A field judged under another holder's name (``holder_name``, the table's judge skips it). The declared domain
    walks its value (``walked`` True), reads it by its own reader (a function), or skips it (False: a property read
    over a field the walk covers); ``of``: the holder it holds, absent or an object of that holder's class; ``every``: a
    field of a frozen holder that holds what is not frozen, read at every call."""
    def judge(registry: Any, field: str, value: Any, holder: Any) -> None:
        return None
    setattr(judge, "via", holder_name)
    setattr(judge, "domain", walked)
    setattr(judge, "of", of)
    setattr(judge, "every_call", every)
    return judge


def _mapping(value: Any, place: Any, words: str) -> Any:
    if type(value) not in _door.MAPPINGS:
        _door.outside(place, value, f"not {words}")
    return value


def beside_constants(physics: Any, place: Any, depth: int) -> None:
    """The physics' constants as the rows beside the block read them (declaration.beside_of), the declared domain's
    reader: its outside part (area_global_fixed) and whether an area_per_adc is stated; never its other constants."""
    _door.in_domain(_mapping(physics.constants, place, "an object of constants").get(FIXED), place + (FIXED,), depth)


def beside_rows(table: Any, place: Any, depth: int) -> None:
    """A converter table's rows as the rows beside the block read them (whether a row states an area_per_adc: its
    keys, never its values), the declared domain's reader: an object of rows, each an object."""
    for bits, row in _mapping(table.rows, place, "an object of rows").items():
        _mapping(row, place + (f"[{bits!r}]",), "an object of constants")


__all__ = ["CENSUS", "Spec", "area_flags", "beside_constants", "census_extent", "citation", "expression", "flag",
           "held_logic", "law_organization", "memory_beside", "name", "one_of", "quantity", "row_ratio", "terms_form",
           "text", "texts", "via"]
