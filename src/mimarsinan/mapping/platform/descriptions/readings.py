"""[U-CHIPKEYS2] How a statement's quote is read (the reading check of the loader rule in ``computation_keys``).

- ``reads`` is the span of the quote that states the value: one span, or for an object value one span per field it
  reads (``{field: span}``); every span is a span of the quote that starts and ends at the quote's token boundaries
  ([U-CHIPKEYS3] m-1: a span never cuts a number, a word or an operator, so "8b outputs" is no span of "128b outputs").
- An integer is read in its span when the span holds that one number (one integer token per integer field: every
  numeral and number word of the span is that number), or when the span's numbers anchored by the field's unit are
  that one number (a width in bits written with its unit, "6-bit output", "32b outputs", or its format, "INT8",
  "FP16"; a count of cells with the cells it counts, "two 4-bit cells"); a field of no unit reads one number. A
  derived integer reads its named derivation's input the same way.
- A closed string value is read where its legal options are spelled words: a span that spells another option of the
  key and not the value refuses (codes and symbols, "L" or "<=", and booleans have no spelling the loader can read;
  the tests' lexicon reads them).
- [U-CHIPKEYS3] m-2: ``assumed_fields`` names the fields of an object value its source does not state, each with why:
  a field of the value, a non-empty reason, never every field (a stated value reads at least one field).
- A weight grid states, beside it, the cells one weight's value is bit-sliced over and the precision of one cell
  (``composition``), each a reading of its own; where both are stated, a named composition law gives the grid.
  [U-CHIPKEYS3] m-3: the slices (``slices_per_weight``, the law's operand) are named apart from the core field
  ``cells_per_weight`` (every cell one stored weight occupies, as the cost model writes, wears and prices it), and
  where both are stated the core count holds whole copies of the slices (a polarity twin or a differential partner).
- [U-CHIPKEYS4] 3(d): a derivation entry takes its three fields only (``DERIVATION_FIELDS``); 3(b): each pattern holds
  no element its result does not depend on (the per-element sweep's equivalents removed, the readings unchanged).
- [U-CHIPKEYS5] Each whitespace run of a quote and a span reads as one space (``plain``, the first act of each reader
  below: every pattern sees single spaces); m-5: a width's one-letter unit is the bit's 'b' ("32b"), never a byte's 'B'.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

#: Number words a quote may spell an integer with (the longest first, so "thirty-two" is one number).
WORDS: Mapping[str, int] = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                            "nine": 9, "ten": 10, "twelve": 12, "sixteen": 16, "thirty-two": 32}
_NUMERAL = re.compile(r"(?<![0-9.,])\d+(?:[.,]\d+)*")
_INTEGER = re.compile(r"\d{1,3}(?:,\d{3})+|\d+")
_WORD = re.compile(r"\b(" + "|".join(sorted(WORDS, key=len, reverse=True)) + r")\b", re.IGNORECASE)
#: A width's unit after its number ("6-bit", "20 bit", "8 bits", "32b") or its format before it ("INT8", "FP16").
_UNIT_AFTER = re.compile(r"\s?-?\s?(?:bits?|(?-i:b))\b", re.IGNORECASE)
_FORMAT_BEFORE = re.compile(r"(?<![a-z])(?:u?int|fp|bf)$", re.IGNORECASE)
#: A count of cells: its number, then at most a width and two words, then the cells counted ("two 4-bit cells", "8
#: 2-bit memristor cells", "4 PCM devices", "8 crossbars"); a number with a width's unit is a width, never a count.
_COUNT_AFTER = re.compile(r"(?!\s-?\s?(?:bits?|b)\b)\s+(?:\d+\s?-?\s?bits?\s+|[a-z]+-bits?\s+)?"
                          r"(?:[a-z+-]+\s+){0,2}(?:cells?|devices?|crossbars?|pcms?)\b", re.IGNORECASE)
#: The unit an integer field is anchored by: a count of cells, else a width in bits (a ``*_bits`` field), else none.
UNITS: Mapping[str, str] = {"slices_per_weight": "cells"}
#: A field's one number is never the number another unit anchors (a width read as a count, or a count as a width).
_OTHER_UNITS: Mapping[str, Tuple[str, ...]] = {"bits": ("cells",), "cells": ("bits",)}
_SPELLED = re.compile(r"[A-Za-z]{2,}(?:_[A-Za-z]+)*")
#: [U-CHIPKEYS3] The characters one token joins: a word or number (letters, digits, underscore), an operator.
_TOKEN_CHARS = (re.compile(r"\w"), re.compile(r"[<>=!]"))
#: [U-CHIPKEYS3] A number's decimal or group mark between two digits ("2.5", "1,024") joins one token.
_NUMBER_MARKS = ".,"
_WHITESPACE = re.compile(r"\s+")  # [U-CHIPKEYS5] a run of whitespace, read as one space (``plain``)

#: Named laws composing a weight grid (the symmetric signed grid's bits) from (slices per weight, bits per cell).
COMPOSITIONS: Mapping[str, Tuple[str, Callable[[int, int], int]]] = {
    "signed_word": ("the slices hold one signed word, bit-sliced over them: slices x cell bits", lambda n, b: n * b),
    "magnitude_signed_apart": ("the slices hold one magnitude, bit-sliced over them, and its sign is held apart (a "
                               "second array or a row's sign bit): the symmetric signed grid of slices x cell bits + "
                               "1", lambda n, b: n * b + 1),
}
#: The readings a key's statement carries beside it: the weight grid's slices per weight and bits per cell.
COMPOSITION_READINGS: Mapping[str, Tuple[str, ...]] = {"weight_bits": ("slices_per_weight", "cell_bits")}
#: [U-CHIPKEYS3] m-3: the reading of a composition a core field holds whole copies of: (reading, core field).
COMPOSITION_TIES: Mapping[str, Tuple[str, str]] = {"weight_bits": ("slices_per_weight", "cells_per_weight")}
_READING_STATED = frozenset({"value", "source", "location", "quote", "reads", "derivation", "note"})
#: [U-CHIPKEYS4] 3(d) A derivation entry's closed fields: the field it derives, the named rule and the integer read.
DERIVATION_FIELDS = ("field", "rule", "reads")
_READING_UNSTATED = frozenset({"note"})


def plain(text: str) -> str:
    """[U-CHIPKEYS5] ``text`` with each run of whitespace read as one space: the form a quote and a span are read in."""
    return _WHITESPACE.sub(" ", text)


def number_tokens(span: str) -> List[Tuple[Optional[int], int, int]]:
    """``(value, start, end)`` of every number in ``span``: numerals (a thousands-grouped one is one number, a decimal
    one has no integer value) and number words, in order."""
    out: List[Tuple[Optional[int], int, int]] = []
    for match in _NUMERAL.finditer(span):
        raw = match.group(0)
        out.append((int(raw.replace(",", "")) if _INTEGER.fullmatch(raw) else None, match.start(), match.end()))
    out += [(WORDS[m.group(1).lower()], m.start(), m.end()) for m in _WORD.finditer(span)]
    return sorted(out, key=lambda token: token[1])


def unit_of(where: str) -> Optional[str]:
    """The unit of the integer field at ``where`` (a key path, its last part the field's name)."""
    name = where.rsplit(".", 1)[-1]
    return UNITS.get(name, "bits" if name.endswith("_bits") else None)


def _anchored(span: str, token: Tuple[Optional[int], int, int], unit: Optional[str]) -> bool:
    value, start, end = token
    if value is None or unit is None:
        return False
    if unit == "cells":
        return _COUNT_AFTER.match(span, end) is not None
    return _UNIT_AFTER.match(span, end) is not None or _FORMAT_BEFORE.search(span[:start]) is not None


def integer_reading_error(span: str, number: int, unit: Optional[str] = "bits") -> Optional[str]:
    """Why ``number`` is not read in ``span`` (None when it is): the span's numbers are that one number (one integer
    token), or the numbers anchored by the field's unit are."""
    span = plain(span)
    tokens = number_tokens(span)
    foreign = [t for t in tokens for other in _OTHER_UNITS.get(unit or "", ()) if _anchored(span, t, other)]
    if {value for value, _, _ in tokens} == {number} and not foreign:  # one number, and not another unit's
        return None
    if {token[0] for token in tokens if _anchored(span, token, unit)} == {number}:
        return None
    return (f"{number} is not the one number of {span!r} (its numbers {[t[0] for t in tokens]}) nor its one "
            f"number anchored by its unit ({unit})")


def _spelling(option: Any) -> Optional[re.Pattern]:
    if not isinstance(option, str) or not _SPELLED.fullmatch(option):
        return None
    return re.compile(r"\b" + r"[ _-]".join(map(re.escape, option.split("_"))) + r"\b", re.IGNORECASE)


def option_reading_error(span: str, value: Any, options: Sequence[Any]) -> Optional[str]:
    """Why a closed string ``value`` is not read in ``span`` (None when it is or when no option is spelled there):
    the span spells another legal option and not the value."""
    span, own = plain(span), _spelling(value)
    others = [option for option in options if option != value and (p := _spelling(option)) and p.search(span)]
    if others and own is not None and not own.search(span):
        return f"{span!r} spells {others}, another legal value, and not {value!r}"
    return None


def _cut(text: str, at: int) -> bool:
    """[U-CHIPKEYS3] Whether a boundary at ``at`` cuts one token of ``text`` (a word, a number or an operator)."""
    if at <= 0 or at >= len(text):
        return False
    left, right = text[at - 1], text[at]
    if any(chars.fullmatch(left) and chars.fullmatch(right) for chars in _TOKEN_CHARS):
        return True
    return ((left in _NUMBER_MARKS and right.isdigit() and at >= 2 and text[at - 2].isdigit())
            or (right in _NUMBER_MARKS and left.isdigit() and at + 1 < len(text) and text[at + 1].isdigit()))


def span_at_token_boundaries(span: str, quote: str) -> bool:
    """[U-CHIPKEYS3] m-1: ``span`` occurs in ``quote`` starting and ending at token boundaries (at any occurrence)."""
    span, quote = plain(span), plain(quote)
    start = quote.find(span)
    while start >= 0:
        if not _cut(quote, start) and not _cut(quote, start + len(span)):
            return True
        start = quote.find(span, start + 1)
    return False


def read_spans(body: Mapping[str, Any], fields: Sequence[Optional[str]]) -> Tuple[Dict[Optional[str], str], List[str]]:
    """``({field: span}, errors)``: one ``reads`` span for every read field, or one span per field (an object's
    ``{field: span}``, which names exactly the read fields); every span is a non-empty span of the quote, at its token
    boundaries."""
    reads, quote = body.get("reads"), body.get("quote")
    if isinstance(reads, str):
        spans: Dict[Optional[str], str] = {field: reads for field in fields}
    elif isinstance(reads, Mapping) and None not in fields:
        if set(reads) != set(fields):
            return {}, [f"reads names the fields {sorted(reads)}, the value reads {sorted(map(str, fields))}"]
        spans = {field: reads[field] for field in fields}
    else:
        return {}, [f"reads is a span, or one span per field of an object value, got {reads!r}"]
    errors = []
    for span in dict.fromkeys(spans.values()):
        if not isinstance(span, str) or not span.strip() or plain(span) not in plain(str(quote)):
            errors.append(f"reads {span!r} is not a span of its quote")
        elif not span_at_token_boundaries(span, str(quote)):
            errors.append(f"reads {span!r} cuts a token of its quote (a span starts and ends at token boundaries)")
    return spans, errors


def _filled(value: Any, spans: bool) -> bool:
    """A non-empty text (or, for ``reads``, a non-empty ``{field: span}``)."""
    return bool(isinstance(value, str) and value.strip()) or bool(spans and isinstance(value, Mapping) and value)


def judged_reading(where: str, entry: Any, stated: frozenset = _READING_STATED,
                   unstated: frozenset = _READING_UNSTATED,
                   forms: Optional[Mapping[str, Tuple[frozenset, Tuple[str, ...]]]] = None
                   ) -> Tuple[Optional[str], Dict[str, Any], List[str]]:
    """``(kind, body, errors)`` of one reading: ``{'stated': {...}}`` with a non-empty source, location, quote and
    reads, or ``{'unstated': {...}}`` with a non-empty note (the forms of every statement); ``forms`` adds a form of
    its own ``{kind: (its fields, its non-empty text fields)}`` (a key's statement: ``computation_keys.DEPARTS``)."""
    kinds = {"stated": (stated, ("source", "location", "quote", "reads")), "unstated": (unstated, ("note",)),
             **(forms or {})}
    if not isinstance(entry, Mapping) or len(entry) != 1 or next(iter(entry)) not in kinds:
        return None, {}, [f"{where}: a statement is one of {[{k: {}} for k in kinds]}, got {entry!r}"]
    kind, body = next(iter(entry.items()))
    allowed, texts = kinds[kind]
    if not isinstance(body, Mapping) or set(body) - allowed:
        return None, {}, [f"{where}: {kind} takes the fields {sorted(allowed)}, got {body!r}"]
    errors = [f"{where}: {kind} needs a non-empty {f}" for f in texts if not _filled(body.get(f), f == "reads")]
    if kind == "stated" and "value" not in body:
        errors.append(f"{where}: stated needs its value")
    return kind, dict(body), errors


def assumed_field_errors(where: str, body: Mapping[str, Any]) -> List[str]:
    """[U-CHIPKEYS3] m-2: why a statement's ``assumed_fields`` does not hold: an object ``{field: why}`` of an object
    value, each a field of the value with a non-empty reason, never every field of it."""
    if "assumed_fields" not in body:
        return []
    assumed, value = body["assumed_fields"], body.get("value")
    if not isinstance(value, Mapping):
        return [f"{where}: assumed_fields names fields of an object value, and {value!r} is none"]
    if not isinstance(assumed, Mapping) or not assumed:
        return [f"{where}: assumed_fields is a non-empty object {{field: why}}, got {assumed!r}"]
    errors = [f"{where}.{field}: assumed_fields names no field of the value (its fields {sorted(value)})"
              for field in assumed if field not in value]
    errors += [f"{where}.{field}: an assumed field states why its source does not state it"
               for field, why in assumed.items() if not (isinstance(why, str) and why.strip())]
    if set(value) <= set(assumed):
        errors.append(f"{where}: assumed_fields names every field of the value; a stated value reads one at least")
    return errors


def value_reading_errors(where: str, body: Mapping[str, Any], options: Callable[[Optional[str]], Sequence[Any]],
                         derivations: Mapping[str, Tuple[str, Any]]) -> List[str]:
    """Why a stated value is not read in its spans: each integer (directly or by its named derivation's input) by the
    integer rule, each closed string by the spelled-option rule; ``options(field)`` gives a field's legal values."""
    if errors := assumed_field_errors(where, body):
        return errors
    value = body["value"]
    assumed = set(body.get("assumed_fields") or ())
    items = [(None, value)] if not isinstance(value, Mapping) else [(f, v) for f, v in value.items()
                                                                   if f not in assumed]
    read = [field for field, _ in items]
    derivation = body.get("derivation", [])
    if not isinstance(derivation, list) or not all(isinstance(d, Mapping) and d.get("field", "?") in read
                                                   for d in derivation) \
            or len({d["field"] for d in derivation}) != len(derivation):
        return [f"{where}: derivation is a list of {{field, rule, reads}}, one per field it derives, each a field the "
                f"value reads ({read}), got {derivation!r}"]  # [U-CHIPKEYS3] never silently ignored
    if extra := sorted({str(key) for entry in derivation for key in entry} - set(DERIVATION_FIELDS)):
        return [f"{where}: a derivation entry takes the fields {list(DERIVATION_FIELDS)}, got {extra}"]  # 3(d)
    spans, errors = read_spans(body, read)
    if errors:
        return [f"{where}: {e}" for e in errors]
    derived = {d["field"]: d for d in derivation}
    for field, item in items:
        at, span = where if field is None else f"{where}.{field}", spans[field]
        for one in item if isinstance(item, list) else [item]:
            if isinstance(one, str):
                if (why := option_reading_error(span, one, options(field))) is not None:
                    errors.append(f"{at}: {why}")
                continue
            if isinstance(one, bool) or not isinstance(one, int):
                continue
            rule = derived.get(field)
            if rule is None:
                if (why := integer_reading_error(span, one, unit_of(at))) is not None:
                    errors.append(f"{at}: the stated {one} is not read: {why}")
            elif rule.get("rule") not in derivations or not isinstance(rule.get("reads"), int):
                errors.append(f"{at}: derivation {rule} names no rule of {sorted(derivations)} with an integer read")
            elif derivations[rule["rule"]][1](rule["reads"]) != one:
                errors.append(f"{at}: {rule['rule']} of {rule['reads']} is not {one}")
            elif (why := integer_reading_error(span, rule["reads"], unit_of(at))) is not None:
                errors.append(f"{at}: {rule['rule']}'s {rule['reads']} is not read: {why}")
    return errors


def composition_errors(where: str, composition: Any, grid: Any, names: Sequence[str],
                       derivations: Mapping[str, Tuple[str, Any]],
                       tie: Optional[Tuple[str, str, Sequence[Any]]] = None) -> List[str]:
    """Why a grid's composition does not hold: each of its readings ``names`` judged (stated and read, or unstated
    with a note), and where both are stated a named law of ``COMPOSITIONS`` that gives the description's ``grid``;
    ``tie`` ``(reading, core field, the core types' stated counts)``: each count holds whole copies of the reading."""
    if not isinstance(composition, Mapping) or set(composition) - {*names, "law"} or set(names) - set(composition):
        return [f"{where}: composition takes {list(names)} (each a statement) and a law, got {composition!r}"]
    errors, stated = [], {}
    for name in names:
        kind, body, problems = judged_reading(f"{where}.{name}", composition[name])
        errors += problems
        if kind == "stated" and not problems:
            if isinstance(body["value"], bool) or not isinstance(body["value"], int) or body["value"] < 1:
                errors.append(f"{where}.{name}: a count of cells or bits is a positive integer, got {body['value']!r}")
                continue
            errors += value_reading_errors(f"{where}.{name}", body, lambda _: (), derivations)
            stated[name] = body["value"]
    if tie is not None and tie[0] in stated:
        if bad := [count for count in tie[2] if count % stated[tie[0]]]:
            errors.append(f"{where}: the cores' {tie[1]} {bad} holds no whole copies of the {stated[tie[0]]} "
                          f"{tie[0]} (each core count is a multiple of the slices: a polarity twin's included)")
    law = composition.get("law")
    if len(stated) < len(names):
        return errors + ([f"{where}: a law composes stated readings only, got {law!r}"] if law is not None else [])
    rule = COMPOSITIONS.get(law) if isinstance(law, str) else None
    if rule is None:
        return errors + [f"{where}: both readings stated, so a law of {sorted(COMPOSITIONS)} composes them, got "
                         f"{law!r}"]
    if (composed := rule[1](*(stated[n] for n in names))) != grid:
        errors.append(f"{where}: {law} of {stated} is a {composed}-bit grid, but the description sets {grid!r}")
    return errors
