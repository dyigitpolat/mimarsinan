"""[U-AREA] The evidence of the area rule and the restricted arithmetic its registry is written in.

The evidence file (``data/evidence/published_figures.json``) holds every published figure the rule cites (a subset of
the corrected area set's published.json: value, quote, location, source, kind, scope and array scope class), the
page-text and description readings, the kind-basis quotes and the record-only quotes. A registry value is a published
entry, an expression over entries (derived) or an aggregate of such expressions (fitted).

Expressions are Python arithmetic restricted to numbers, ``+ - * / **``, unary minus and five calls:
``v(id[, index][, key=...])`` an entry's value; ``s(id, field, text, value)`` a value the entry's field states in
``text`` (the field must hold it); ``qn(id, field, text)`` the one number ``text`` holds (the field must hold it);
``r(path)`` another registry value; ``sigma(node_nm)`` the node series' reference cell. An expression that reads an
entry of a held-out chip (an id starting with an excluded prefix) is held out (``HeldOut``), never priced; an exclusion
entry ``KEPT + id`` keeps that one id (a chip's own printed unit read as an input of its declaration). Each expression
is parsed once (``parsed``, the fix round's mi-1); the registry is immutable after load.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.platform_physics.area_rule import units as u

#: The calls an expression may make, and those that read an entry directly.
CALLS = ("v", "s", "qn", "r", "sigma")
ENTRY_CALLS = ("v", "s", "qn")
#: The mark of an exclusion entry that keeps one id under an excluded prefix (design 3.3: a held-out chip's
#: organization and own printed units are kept as inputs of its declaration).
KEPT = "!"
_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


class AreaRuleRefusal(ToolchainRefusal, ValueError):
    """A value the area rule needs and its registry does not hold, or a declaration it declines: a named refusal."""


class HeldOut(Exception):
    """An expression reads an entry of the held-out chip; the value is not available under that exclusion."""


class ExpressionRefusal(AreaRuleRefusal):
    """A registry expression its restricted arithmetic cannot evaluate (the third delta round: one guard for every
    expression, declared or data); ``cause`` names the failure, its type and its text."""

    cause: str = ""


def excluded(entry_id: str, excl: Iterable[str]) -> bool:
    excl = tuple(excl)
    return KEPT + entry_id not in excl and any(entry_id.startswith(p) for p in excl if not p.startswith(KEPT))


def expression_size(expression: str) -> Tuple[int, int]:
    """(an expression's length, its depth): its bracket nesting plus its arithmetic operators, counted on its text
    outside its quoted texts, a bound of the depth the restricted arithmetic recurses (the fourth delta round,
    MA-C5)."""
    nesting = deepest = operators = u.TOP
    quote, escaped = None, False
    for char in expression:
        if quote is not None:
            escaped, quote = (False, quote) if escaped else (char == "\\", None if char == quote else quote)
        elif char in "'\"":
            quote = char
        elif char in "([{":
            nesting += u.LEVEL
            deepest = max(deepest, nesting)
        elif char in ")]}":
            nesting -= u.LEVEL
        elif char in "+-*/%":
            operators += u.LEVEL
    return len(expression), deepest + operators


def bounded(expression: str) -> str:
    """The expression, within its declared bounds (units.EXPRESSION_LENGTH, EXPRESSION_DEPTH), else a named refusal
    before it is parsed (the parser's own failure on a nesting it cannot hold is never reached)."""
    why = None
    if len(expression) > u.EXPRESSION_LENGTH:
        why = f"it is longer than {u.EXPRESSION_LENGTH} characters ({len(expression)})"
    elif (depth := expression_size(expression)[u.SECOND]) > u.EXPRESSION_DEPTH:
        why = f"it nests deeper than {u.EXPRESSION_DEPTH} (brackets and operators: {depth})"
    if why is not None:
        refused = ExpressionRefusal(f"registry expression {expression!r:.80} is not an expression "
                                    f"the registry evaluates: {why}")
        refused.cause = why
        raise refused
    return expression


@lru_cache(maxsize=None)
def parsed(expression: str) -> Tuple[ast.expr, Tuple[str, ...]]:
    """An expression's tree and the entry ids it reads directly, in order (``v``, ``s``, ``qn``): parsed once, within
    its declared bounds (``bounded``)."""
    tree = ast.parse(bounded(expression), mode="eval")
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and getattr(node.func, "id", None) in ENTRY_CALLS]
    return tree.body, tuple(str(getattr(next(iter(call.args)), "value", "")) for call in calls)


def read_ids(expression: str) -> Tuple[str, ...]:
    """The entry ids an expression reads directly, in order (``v``, ``s``, ``qn``)."""
    return parsed(expression)[u.SECOND]


def parse_all(objects: Iterable[Mapping[str, Any]]) -> None:
    """Parse every expression of the registry's value objects once (at load: mi-1)."""
    for obj in objects:
        recompute = obj.get("recompute") or {}
        points = [p.get("expression") for p in recompute.get("points") or () if isinstance(p, Mapping)]
        for expression in [recompute.get("expression"), recompute.get("fallback"), obj.get("fallback")] + points + \
                list(recompute.get("bits") or ()):
            if isinstance(expression, str):
                parsed(expression)


def text_number(text: str) -> float:
    """The one number a quoted text holds (thousands separators allowed); a named refusal otherwise."""
    found = _NUMBER.findall(text)
    if len(found) != int(u.UNITY):
        raise AreaRuleRefusal(f"the text {text!r} holds {len(found)} numbers, not one")
    return float(found[u.FIRST].replace(",", ""))


@dataclass(frozen=True)
class Evidence:
    """The cited figures by id (published entries and readings), with the file's other sections."""

    entries: Mapping[str, Mapping[str, Any]]
    readings: Tuple[Mapping[str, Any], ...] = ()
    kind_basis_quotes: Tuple[str, ...] = ()
    record_only: Tuple[Mapping[str, Any], ...] = ()
    array_scope_classes: Mapping[str, str] = field(default_factory=dict)
    source: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Evidence":
        doc = json.loads(Path(path).read_text())
        entries: Dict[str, Mapping[str, Any]] = {e["id"]: e for e in doc["entries"]}
        for reading in doc.get("readings") or ():
            entries[reading["id"]] = {"id": reading["id"], "value": reading["value"], "quote": reading["text"],
                                      "kind": reading["kind"], "scope": reading.get("what", "")}
        return cls(entries=entries, readings=tuple(doc.get("readings") or ()),
                   kind_basis_quotes=tuple(doc.get("kind_basis_quotes") or ()),
                   record_only=tuple(doc.get("record_only") or ()),
                   array_scope_classes=dict(doc.get("array_scope_classes") or {}), source=dict(doc.get("source") or {}))

    def entry(self, entry_id: str) -> Mapping[str, Any]:
        if not isinstance(entry_id, str) or entry_id not in self.entries:  # MA-D1: an id is a text
            raise AreaRuleRefusal(f"evidence id {entry_id!r} is not in the area rule's evidence file")
        return self.entries[entry_id]

    def value(self, entry_id: str, index: Optional[int] = None, key: Optional[str] = None) -> float:
        value = self.entry(entry_id)["value"]
        if index is not None:
            value = value[index]
        if key is not None:
            value = value[key]
        return float(value)

    def holds(self, entry_id: str, field_name: str, text: str) -> bool:
        return text in str(self.entry(entry_id).get(field_name) or "")


Resolve = Callable[[str, Tuple[str, ...]], float]
Sigma = Callable[[float, Tuple[str, ...]], float]


def evaluate(expression: str, evidence: Evidence, excl: Tuple[str, ...], resolve: Resolve, sigma: Sigma) -> float:
    """The value of a registry expression under the exclusions ``excl``; ``HeldOut`` where it reads an excluded
    entry, a named refusal where a stated text is not in its entry's field or a call is not allowed, and where its
    arithmetic fails (a parse, a division by zero, a power, a product or a quotient beyond the floats, an index it does
    not hold): its value is judged where it is made, a finite number (the fourth delta round, MA-C6)."""
    try:
        body, ids = parsed(expression)
        for entry_id in ids:
            evidence.entry(entry_id)
            if excluded(entry_id, excl):
                raise HeldOut(entry_id)
        value = _Evaluator(evidence, excl, resolve, sigma).value(body)
        if not math.isfinite(value):  # a product or a quotient beyond the floats: the arithmetic's overflow
            raise OverflowError(f"its value {value!r} is no finite number")
        return value
    except (HeldOut, AreaRuleRefusal):
        raise
    except (SyntaxError, ArithmeticError, LookupError, TypeError, ValueError, RecursionError) as caught:
        refused = ExpressionRefusal(f"registry expression {expression!r} is not an expression the registry evaluates "
                                    f"({type(caught).__name__}: {caught})")
        refused.cause = f"{type(caught).__name__}: {caught}"
        raise refused from None


@dataclass
class _Evaluator:
    evidence: Evidence
    excl: Tuple[str, ...]
    resolve: Resolve
    sigma: Sigma

    def value(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self.value(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](self.value(node.left), self.value(node.right))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) in CALLS:
            return self.call(node)
        raise AreaRuleRefusal(f"not allowed in a registry expression: {ast.dump(node)}")

    def call(self, node: ast.Call) -> float:
        name = getattr(node.func, "id")
        args = [a.value if isinstance(a, ast.Constant) and isinstance(a.value, str) else self.value(a)
                for a in node.args]
        keys = {k.arg: k.value.value for k in node.keywords if isinstance(k.value, ast.Constant)}
        if name == "v":
            entry_id, *rest = args
            key = keys.get("key")
            return self.evidence.value(str(entry_id), int(rest[u.FIRST]) if rest else None,
                                       None if key is None else str(key))
        if name in ("s", "qn"):
            entry_id, field_name, text, *rest = (str(a) if isinstance(a, str) else a for a in args)
            if not self.evidence.holds(str(entry_id), str(field_name), str(text)):
                raise AreaRuleRefusal(f"{entry_id}.{field_name} does not state {text!r}")
            return float(rest[u.FIRST]) if name == "s" else text_number(str(text))
        if name == "r":
            return self.resolve(str(args[u.FIRST]), self.excl)
        return self.sigma(float(args[u.FIRST]), self.excl)


_OPERATORS: Dict[type, Callable[[float, float], float]] = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b}


def geometric_mean(values: Sequence[float]) -> float:
    return math.exp(sum(math.log(x) for x in values) / len(values))


def log_linear_least_squares(points: Sequence[Tuple[float, float]], reference: float) -> Tuple[float, float]:
    """(the area at the reference bits, alpha per bit) of a(b) = a_ref x 2^(alpha (b - ref)), least squares in log."""
    xs = [bits - reference for bits, _area in points]
    ys = [math.log(area) for _bits, area in points]
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** u.BINARY for x in xs)
    return math.exp(ym - slope * xm), slope / math.log(u.BINARY)


def _numbers(value: Any) -> Tuple[float, ...]:
    if isinstance(value, bool) or value is None:
        return ()
    if isinstance(value, (int, float)):
        return (float(value),)
    if isinstance(value, Mapping):
        return tuple(x for v in value.values() for x in _numbers(v))
    if isinstance(value, (list, tuple)):
        return tuple(x for v in value for x in _numbers(v))
    return ()


def _spelled(number: float, text: str) -> bool:
    """The number is written in the text (thousands separators and trailing zeros allowed)."""
    plain = re.sub(r"(\d),\s?(?=\d{3}(?!\d))", r"\1", text)
    forms = {f"{number:g}", f"{number:.12g}"}
    if number == int(number):
        forms.add(str(int(number)))
    return any(re.search(rf"(?<![\d.]){re.escape(f)}0*(?!\d)", plain) for f in forms)


def figures_without_their_number(evidence: Evidence) -> Tuple[str, ...]:
    """T-16 (unit side): the measured or layout figures whose number is written in none of their quote, scope or
    derivation (a figure read from a table or a figure label carries its number in the quote)."""
    missing = []
    for entry_id, entry in sorted(evidence.entries.items()):
        if entry.get("kind") not in ("measured", "layout") or entry.get("reading") in ("figure labels",):
            continue
        text = " ".join(str(entry.get(f) or "") for f in ("quote", "scope", "derivation"))
        if not all(_spelled(x, text) for x in _numbers(entry.get("value"))):
            missing.append(entry_id)
    return tuple(missing)
