"""[U-AREA] The area rule's one door (the third delta round; UB4/check.md MA-K1, MA-K2, mi-K1 to mi-K3, UB4's mi-E1) and
its declared input domain (the fourth delta round; UB5/check.md MA-C1 to MA-C6).

Every entry by which a caller reaches the rule's arithmetic is a door (``door``): before its body runs, the one function
``judged_input`` judges every argument the call binds, first by the declared input domain (``in_domain``), then by the
argument's judge in one table (core/arguments.py), the declared objects (the block with every holder, the description's
validity geometry and the rows beside the block) by the same judges the loader calls through
``declaration.judge_block``. No check of the rule's input exists at load only.

**The declared input domain** (stated here, once). A leaf is one of the exact types a description file holds:
Python's int, float, str, bool and None (``LEAVES``, by exact type, never by the number tower: a Fraction, a Decimal and
every numpy number are outside it). A container is a list, a tuple or a mapping of the domain (``SEQUENCES``,
``MAPPINGS``: the plain ones and the unit's frozen ones, core/frozen.py), nested at most ``units.NESTING`` deep. A
holder is an object whose exact type is a class the judge's table declares for one of its holders (core/fields.py
``CLASSES``, the only place a class is named); only the fields the table declares are walked. Anything else is a named
refusal naming its type and its field, before any other judge runs. No caller of record hands the rule a census in
another number type (every census value is made a Python float, every core type a Python int, where it is made), so
none is converted.

- A door inside a door does not judge again (a context variable): its whole cost is that read. A door's call is bound
  by a binder made once per door (``_binder``). A HeldOut leaving a door's body is a named refusal (but at the
  registry's own readers, whose signal it is).
- The judge runs once per frozen input (``remembered``, core/memo.py): a frozen object of the unit by its identity,
  the values read off what is no frozen object (the rows beside the block) as its digest, never a pickled text; a
  frozen holder's domain pass is remembered by its identity too. The rest is judged at every call; a load afresh.
"""

from __future__ import annotations

import contextvars
import copy
import functools
import inspect
import weakref
from typing import Any, Callable, Dict, Mapping, NoReturn, Optional, Sequence, Tuple

from mimarsinan.deployment_record.platform_physics.area_rule import units as u
from mimarsinan.deployment_record.platform_physics.area_rule.core.frozen import FrozenDict, FrozenList
from mimarsinan.deployment_record.platform_physics.area_rule.core.memo import forget, remembered
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal, HeldOut

_INSIDE = contextvars.ContextVar("area_rule_inside_the_door", default=False)
#: Every door, by its module and qualified name (filled as the rule's modules load).
DOORS: Dict[str, Callable[..., Any]] = {}
#: The public functions of the rule's pricing modules that are no door, each with its reason (a structural test holds
#: every other public function a door).
NOT_DOORS: Mapping[str, Mapping[str, str]] = {
    "core/equation.py": {"judged_fields": "the judge's form for an organization or a reading",
                         "cited": "the judge's form for an expression over the evidence"},
    "declaration.py": {"judge_block": "the one function (judged_input) as the loader calls it",
                       "judged_block": "the block's judge, which judged_input calls (the door's table)",
                       "judge_beside": "the judge of the rows beside the block, which judged_input calls",
                       "beside_of": "reads the rows beside the block off a physics as the loader reads them: the "
                                    "beside judge judges what it reads",
                       "converter_area_stated": "reads whether a converter's area is stated beside the block (the "
                                                "loader's and the price's one reading)",
                       "block_digest": "names a block by its text: it reads no quantity",
                       "AreaRuleBlock.node_nm": "the block's node, judged as a size where it is read",
                       "AreaRuleBlock.bitcell_um2": "the declared cell as stated: declared_cell judges it",
                       "AreaRuleBlock.to_dict": "the block's own text",
                       "AreaRuleBlock.from_dict": "makes the block the loader then judges"},
    "rule.py": {"AreaPrice.reading": "an answer's reading by its name"},
    "pricing.py": {"rule_area": "reads only whether the physics declares a block (R5's own path prices one that does "
                                "not), then hands its whole input to the door rule_price; its refusal is the answer's"},
}

#: The declared input domain: the leaves' exact types, the containers' exact types.
LEAVES = frozenset({int, float, str, bool, type(None)})
SEQUENCES = frozenset({list, tuple, FrozenList})
MAPPINGS = frozenset({dict, FrozenDict})
OUTSIDE = ("outside the area rule's declared input domain (a leaf is an int, a float, a text, true, false or absent; a "
           "container a list, a tuple or an object of them; a holder an object of one of the rule's declared classes)")

#: The door's table (core/arguments.py), installed by the package area_rule after every module (an import: a cycle).
_TABLE: Dict[str, Any] = {}
_HOLDER_OF: Dict[type, Optional[str]] = {}
#: A frozen holder's domain outcome by its identity: (a weak reference to it, None or its named refusal).
_FROZEN: Dict[int, Tuple[Any, Optional[AreaRuleRefusal]]] = {}


def install(table: Any) -> None:
    """Install the door's table (the package area_rule does, as the last step of its load) and each holder's walk."""
    _TABLE["table"] = table
    for holder in table.fields.FIELDS:
        _plan(holder, table.fields)


def _place(parts: Sequence[Any]) -> str:
    """A field's path as the refusal names it (``area_rule.node.node_nm``, ``rows[0].readings[1]``)."""
    out = ""
    for part in parts:
        text = str(part)
        out += text if text.startswith("[") or not out else f".{text}"
    return out


def outside(place: Sequence[Any], value: Any, why: str = OUTSIDE) -> NoReturn:
    """The domain's named refusal: the field, the value, its type, and why (by default: outside the domain)."""
    kind = type(value)
    raise AreaRuleRefusal(f"{_place(place)} {value!r:.80} (a {kind.__module__}.{kind.__qualname__}) is {why}")


def holder_of(kind: type) -> Optional[str]:
    """The judge's table's holder whose declared class is exactly ``kind`` (None: no declared class)."""
    if kind not in _HOLDER_OF:
        _HOLDER_OF[kind] = _TABLE["table"].fields.HOLDER_OF.get(f"{kind.__module__}.{kind.__qualname__}")
    return _HOLDER_OF[kind]


#: Each holder's walk, made from the judge's table as it is installed: (field, its reader or None, the holder it holds
#: or None, whether it is read at every call: a field of a frozen holder that holds what is not frozen); a holder of
#: plain fields read by attribute only (no reader, no holder held) is walked by its field names alone.
_PLANS: Dict[str, Tuple[Tuple[str, Any, Optional[str], bool], ...]] = {}
_PLAIN: Dict[str, Optional[Tuple[str, ...]]] = {}
_EVERY: Dict[str, bool] = {}


def _plan(holder: str, fields: Any) -> None:
    _PLANS[holder] = plan = tuple(
        (name, None if reads is True else reads, of if of in fields.BY_ATTRIBUTE else None,
         bool(getattr(spec, "every_call", False)))
        for name, spec in fields.FIELDS[holder].items()
        for reads, of in ((getattr(spec, "domain", True), getattr(spec, "of", None)),) if reads is not False)
    plain = holder not in fields.BY_METHOD and all(r is None and o is None for _n, r, o, _e in plan)
    _PLAIN[holder] = tuple(name for name, _r, _o, _e in plan) if plain else None
    _EVERY[holder] = any(every for _n, _r, _o, every in plan)


def in_domain(value: Any, place: Tuple[Any, ...], depth: int = u.TOP) -> None:
    """``value`` at ``place`` (the path of its field) in the declared input domain, else a named refusal naming its type
    and its field: one pass over its leaves, a frozen holder's pass remembered by its identity."""
    kind = type(value)
    if kind in LEAVES:
        return
    if depth >= u.NESTING:
        raise AreaRuleRefusal(f"{_place(place)} nests deeper than {u.NESTING} levels: {OUTSIDE}")
    if kind in SEQUENCES:
        for i, item in enumerate(value):
            if type(item) not in LEAVES:
                in_domain(item, place + (f"[{i}]",), depth + u.LEVEL)
        return
    if kind in MAPPINGS:
        for key, item in value.items():
            if type(key) not in LEAVES:
                raise AreaRuleRefusal(f"{_place(place)}: a key {key!r:.80} (a {type(key).__qualname__}) is {OUTSIDE}")
            if type(item) not in LEAVES:
                in_domain(item, place + (key,), depth + u.LEVEL)
        return
    holder = _HOLDER_OF[kind] if kind in _HOLDER_OF else holder_of(kind)
    if holder is None:
        outside(place, value)
    fields = _TABLE["table"].fields
    place = fields.PLACES.get(holder, place)
    if holder not in fields.FROZEN:
        plain = _PLAIN[holder]
        if plain is None:
            walk_holder(value, holder, place, depth)
            return
        for name in plain:  # a holder of plain fields: its leaves checked here, its holders walked
            item = getattr(value, name)
            if type(item) not in LEAVES:
                in_domain(item, place + (name,), depth + u.LEVEL)
        return
    seen = _FROZEN.get(id(value))
    if seen is None or seen[u.FIRST]() is not value:
        try:
            walk_holder(value, holder, place, depth, False)
            refused = None
        except AreaRuleRefusal as caught:
            refused = caught.with_traceback(None)
        ref = weakref.ref(value, lambda _ref, key=id(value): _FROZEN.pop(key, None))
        seen = _FROZEN[id(value)] = (ref, refused)
    if (refused := seen[u.SECOND]) is not None:
        raise copy.copy(refused)
    if _EVERY[holder]:  # what it holds that is not frozen: read at every call
        walk_holder(value, holder, place, depth, True)


def walk_holder(value: Any, holder: str, place: Tuple[Any, ...], depth: int, every: Optional[bool] = None) -> None:
    """A holder's declared fields (the judge's table) in the domain: each read as the rule reads it (by attribute, by
    the census's has and get, or by the field's own reader); a field the table marks read through another field is
    not walked again; a field that holds a holder read by attribute holds an object of its class, or nothing
    (``every``: True the fields read at every call alone, False the others alone, None all)."""
    fields = _TABLE["table"].fields
    by_method = holder in fields.BY_METHOD
    present = set(value.keys()) if by_method else ()
    for name, reads, of, every_call in _PLANS[holder]:
        if every is not None and every_call is not every:
            continue
        if reads is not None:
            reads(value, place + (name,), depth + u.LEVEL)
            continue
        if by_method:
            if name in present:
                in_domain(value.get(name), place + (name,), depth + u.LEVEL)
            continue
        item = getattr(value, name)
        if type(item) in LEAVES:
            if of is None or item is None:
                continue
        elif of is None or holder_of(type(item)) == of:
            in_domain(item, place + (name,), depth + u.LEVEL)
            continue
        outside(place + (name,), item, f"not a {of} (an object of {list(fields.CLASSES[of])}): {OUTSIDE}")


def forget_domain() -> None:
    """Drop the frozen holders' remembered domain passes (a test's fresh start)."""
    _FROZEN.clear()


def judged_input(arguments: Mapping[str, Any], judges: Optional[Mapping[str, str]] = None, *, registry: Any = None,
                 memo: bool = True, door: str = "the loader") -> None:
    """The one function that judges a whole input of the area rule, before any arithmetic: every argument in the
    declared input domain first, then each by its judge (the door's own, else the table's for its name); ``memo``
    False judges the declared objects afresh (a load)."""
    table = _TABLE.get("table")
    if table is None:  # a defect of the code: the package installs the table as it loads
        raise TypeError(f"{door}: the door's table (core/arguments.py) is not installed")
    token = None if _INSIDE.get() else _INSIDE.set(True)  # the judge's own calls of doors are inside the door
    try:
        for name, value in arguments.items():
            in_domain(value, (name,))
        reg = table.registry_of(arguments, registry)
        for name, value in arguments.items():
            judge = (judges or {}).get(name) or table.ARGUMENTS.get(name)
            if judge is None:  # a defect of the code, which the structural test keeps out
                raise TypeError(f"{door}: the door's table states no judge for its argument {name!r}")
            table.JUDGES[judge](reg, name, value, arguments, memo)
    finally:
        if token is not None:
            _INSIDE.reset(token)


def _binder(signature: inspect.Signature) -> Callable[[tuple, dict], Dict[str, Any]]:
    """A call's arguments by name, in the signature's order, its defaults applied (inspect's bind where the signature
    takes more than named arguments); a call the signature cannot bind is Python's own TypeError."""
    params = list(signature.parameters.values())
    if any(p.kind not in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) for p in params):
        def bound_by_inspect(args: tuple, kwargs: dict) -> Dict[str, Any]:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            return dict(bound.arguments)
        return bound_by_inspect
    names = tuple(p.name for p in params)
    positional = sum(p.kind is p.POSITIONAL_OR_KEYWORD for p in params)
    defaults = {p.name: p.default for p in params if p.default is not p.empty}

    def bound(args: tuple, kwargs: dict) -> Dict[str, Any]:
        if len(args) > positional:
            raise TypeError("too many positional arguments")
        if stray := [k for k in kwargs if k not in names or names.index(k) < len(args)]:
            raise TypeError(f"got an unexpected or repeated keyword argument {stray[u.FIRST]!r}")
        out = {}
        for i, name in enumerate(names):
            if i < len(args):
                out[name] = args[i]
            elif name in kwargs:
                out[name] = kwargs[name]
            elif name in defaults:
                out[name] = defaults[name]
            else:
                raise TypeError(f"missing a required argument: {name!r}")
        return out
    return bound


def door(function: Optional[Callable[..., Any]] = None, *, keep_held_out: bool = False, **judges: str) -> Any:
    """A public entry of the rule made a door: its bound arguments judged by ``judged_input`` before its body
    (``judges``: an argument's judge where its name means another thing at this door)."""
    def make(fn: Callable[..., Any]) -> Callable[..., Any]:
        bind, name = _binder(inspect.signature(fn)), fn.__qualname__

        @functools.wraps(fn)
        def entry(*args: Any, **kwargs: Any) -> Any:
            if _INSIDE.get():
                return fn(*args, **kwargs)
            token = _INSIDE.set(True)
            try:
                bound = bind(args, kwargs)
                registry = bound.pop("self", None)
                bound.pop("cls", None)
                judged_input(bound, judges, registry=registry, door=name)
                return fn(*args, **kwargs)
            except HeldOut as held:
                if keep_held_out:
                    raise
                raise AreaRuleRefusal(f"{name}: it reads an entry the registry holds out ({held})") from None
            finally:
                _INSIDE.reset(token)

        setattr(entry, "__door__", True)
        setattr(entry, "__judges__", dict(judges))
        DOORS[f"{fn.__module__}.{fn.__qualname__}"] = entry
        return entry
    return make if function is None else make(function)


__all__ = ["DOORS", "LEAVES", "MAPPINGS", "NOT_DOORS", "OUTSIDE", "SEQUENCES", "door", "forget", "forget_domain",
           "holder_of", "in_domain", "install", "judged_input", "remembered", "walk_holder"]
