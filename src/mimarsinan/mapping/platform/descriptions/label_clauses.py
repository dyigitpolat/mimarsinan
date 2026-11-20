"""[U-CHIPKEYS5] The faithfulness label's derived clauses: one grammar its writers format and its reader reads back.

A label is ``<kind>[, given "<setting>" and given "<setting>"][: <item>; <item>]`` (``target_bundles.Faithfulness``).
Beside the items a description types (its named departures, each a quoted field ``named "<text>"``), the label sites
derive clauses from the run: the pricing it resolves (none, or a physics by name: ``departures.pricing_departures``), a
key it declares (``Faithfulness.with_declared``: the run door's overrides, the design point's build declarations), a key
or consumer that does not apply under it (``target_expansion``) and a statement's departure from its source
(``computation_keys.departures``). Each is written from its template in ``CLAUSES`` (``clause``).

[U-CHIPKEYS6] The label's class closed by construction, in two parts (M-1 of U-CHIPKEYS5's check, as its class):

- **The judge reads words, not characters** (``held_clauses``, ``text_error``). A typed text is read as its words after
  one stated normalization: the compatibility form (NFKC), case folded, split into words at every character that is no
  letter or digit. A quote character, a backtick or any other punctuation mark inside a run of such characters with no
  whitespace is dropped or read as a separator (both readings: a quote never joins or splits a clause's words), and
  whitespace always separates. A typed text that a label site prints in or beside the label refuses at load when its
  words hold, in order, the words of a derived clause's template, each slot one or more words. No character pattern
  or word boundary is read.
- **The label prints typed text as a quoted field named by its key** (``field``: the key, then the text as a JSON
  string, its quote character and backslash escaped), and a derived clause is never inside quotes; a clause's typed
  slot (a profile's name, a source's computation, the value assumed) is a quoted field too. The label's reader
  (``derived_clauses``, ``read_label``) reads clauses outside quoted fields only.

[U-CHIPKEYS7] One declared alphabet (M-1 of U-CHIPKEYS6's check): a typed text the label prints in or beside it holds
only ``ALPHABET``'s characters (printable ASCII without every quote character, the separators and the backslash); any
other refuses at load by its code point and name, as does a value that is no text (``text_error``); a value assumed is
judged by its texts (``texts_of``). The writer escapes every character outside the alphabet in a field (``quoted``),
for a text that reaches the label unjudged. No field holds a character that closes it or separates a clause, for the
reader or a person; an ASCII look-alike word ('profiIe') or a fused one prints inside its field as typed text.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from string import Formatter
from typing import Any, List, Optional, Tuple

#: The clauses a label derives, by kind: each a template whose fields are its slots (``clause`` formats it).
CLAUSES = {
    "no_pricing": "no physics profile",
    "pricing": "physics profile {name} declared",
    "declared": "{key} declared",
    "not_applied": "{key} not applied",
    "departs": "{key} departs from its source's computation {computation}, assumed {assumed}",
}
#: [U-CHIPKEYS6] The slots a description or a run types: each printed as a quoted field (``quoted``), never read.
TYPED_SLOTS = frozenset({"name", "computation", "assumed"})
#: [U-CHIPKEYS6] The label's typed items, each a quoted field named by its key: a named departure (in the clauses' list)
#: and a given setting (in the head).
NAMED, GIVEN = "named", "given"
#: The label's separators: between its head and its items, and between two items.
SEPARATORS = frozenset(":;")
#: [U-CHIPKEYS7] The quote characters: the double quote that opens and closes a field, the apostrophe, the backtick.
QUOTES = frozenset("\"'`")
#: [U-CHIPKEYS7] The one alphabet of a typed text the label prints: printable ASCII but the label's structure characters.
ALPHABET = frozenset(map(chr, range(0x20, 0x7F))) - QUOTES - SEPARATORS - {"\\"}
_STRUCTURE = {**dict.fromkeys(QUOTES, "a quote character"), **dict.fromkeys(SEPARATORS, "a label separator"),
              "\\": "the backslash"}
_HEAD, _ITEM, _AND = ": ", "; ", " and "
#: The place of a quoted field in a label read outside its fields (no label character takes it).
_FIELD = chr(0xFFFC)


def quoted(text: Any) -> str:
    """``text`` as a quoted field's text: a JSON string, [U-CHIPKEYS7] its every character outside ``ALPHABET`` escaped
    (the quote and the backslash as JSON escapes them, any other as ``\\uXXXX``): the second line, so a text the judge
    never read (a run's declared profile name) prints no look-alike, quote character or separator raw."""
    return "".join(char if char in ALPHABET or char in '"\\' else f"\\u{ord(char):04x}"
                   for char in json.dumps(str(text)))


def field(key: str, text: Any) -> str:
    """[U-CHIPKEYS6] A typed text as the label prints it: its key, then the text quoted (``quoted``)."""
    return f"{key} {quoted(text)}"


def clause(kind: str, **slots: Any) -> str:
    """The derived clause of ``kind`` with its slots filled, each typed slot quoted: the label's one writer of it."""
    return CLAUSES[kind].format(**{name: quoted(value) if name in TYPED_SLOTS else value
                                   for name, value in slots.items()})


# --- the judge: a typed text's words -------------------------------------------------------------------------------

def words(text: str) -> Tuple[Tuple[str, ...], Tuple[bool, ...]]:
    """``(words, soft)``: ``text`` normalized (NFKC, case folded) and split into its words (runs of letters and
    digits); ``soft[i]`` is whether the run of other characters between word ``i`` and word ``i + 1`` holds no
    whitespace (a quote or other mark there is dropped or a separator: both readings are read)."""
    out: List[str] = []
    soft: List[bool] = []
    word, joint = "", None
    for char in unicodedata.normalize("NFKC", text).casefold():
        if char.isalnum():
            if word == "" and out:
                soft.append(joint is not None and not joint)
            word += char
            joint = None
            continue
        if word:
            out.append(word)
            word, joint = "", False
        if joint is not None and char.isspace():
            joint = True
    if word:
        out.append(word)
    return tuple(out), tuple(soft)


@lru_cache(maxsize=None)
def templates() -> Tuple[Tuple[str, Tuple[Optional[str], ...]], ...]:
    """``(kind, tokens)`` of each template: its fixed words (each literal word with its non-alphanumeric characters
    dropped and case folded) and None for each slot (one or more words)."""
    out = []
    for kind, template in CLAUSES.items():
        tokens: List[Optional[str]] = []
        for literal, slot, _, _ in Formatter().parse(template):
            for part in literal.split():
                if folded := "".join(c for c in unicodedata.normalize("NFKC", part).casefold() if c.isalnum()):
                    tokens.append(folded)
            if slot is not None:
                tokens.append(None)
        out.append((kind, tuple(tokens)))
    return tuple(out)


def _groups(tokens: Tuple[Optional[str], ...]) -> Tuple[bool, Tuple[Tuple[str, ...], ...], bool]:
    """``(lead, groups, trail)``: a template's runs of fixed words between its slots, and whether a slot leads or
    trails."""
    groups: List[List[str]] = [[]]
    for token in tokens:
        if token is None:
            groups.append([])
        else:
            groups[-1].append(token)
    lead, trail = tokens[0] is None, tokens[-1] is None
    return lead, tuple(tuple(g) for g in groups if g), trail


def _run_end(found: Tuple[str, ...], soft: Tuple[bool, ...], start: int, run: Tuple[str, ...]) -> Optional[int]:
    """The word after ``run`` read from word ``start`` on (each fixed word one word, or several joined across soft
    joints), None where it is not read there."""
    at = start
    for wanted in run:
        joined = ""
        while at < len(found) and wanted.startswith(joined + found[at]) and (joined == "" or soft[at - 1]):
            joined += found[at]
            at += 1
            if joined == wanted:
                break
        if joined != wanted:
            return None
    return at


def _holds(found: Tuple[str, ...], soft: Tuple[bool, ...], tokens: Tuple[Optional[str], ...]) -> bool:
    """Whether the words ``found`` hold ``tokens`` in order: each run of fixed words read where it ends first, a slot
    one or more words between (and before a leading or after a trailing slot)."""
    lead, runs, trail = _groups(tokens)
    earliest = 1 if lead else 0
    for run in runs:
        ends = [end for start in range(earliest, len(found)) if (end := _run_end(found, soft, start, run)) is not None]
        if not ends:
            return False
        earliest = min(ends) + 1
    return earliest - 1 < len(found) if trail else True


def held_clauses(text: Any) -> Tuple[str, ...]:
    """The kinds of the derived clauses whose words ``text`` holds (``CLAUSES`` order); () for none or a value that is
    no text."""
    if not isinstance(text, str):
        return ()
    found, soft = words(text)
    return tuple(kind for kind, tokens in templates() if _holds(found, soft, tokens))


def character_errors(text: Any) -> Tuple[str, ...]:
    """[U-CHIPKEYS7] Each character of ``text`` outside ``ALPHABET``, by its code point and name (in code order)."""
    if not isinstance(text, str):
        return ()
    return tuple(f"U+{ord(char):04X} {unicodedata.name(char, '(unnamed)')} "
                 f"({_STRUCTURE.get(char, 'outside printable ASCII')})" for char in sorted(set(text) - ALPHABET))


def texts_of(value: Any) -> Tuple[str, ...]:
    """[U-CHIPKEYS7] The texts a JSON value holds: itself for a text, each string of a list or an object (its keys
    included) at any depth, none for a number, a boolean or null."""
    if isinstance(value, str):
        return (value,)
    items = [part for key, item in value.items() for part in (key, item)] if isinstance(value, dict) else value
    return tuple(text for item in items for text in texts_of(item)) if isinstance(value, (dict, list)) else ()


def text_error(field_name: str, text: Any) -> Optional[str]:
    """Why a description's ``field_name``, which a label site prints in or beside the label, may not hold ``text``
    (None when it may, or when it is absent): [U-CHIPKEYS7] a value that is no text, each character outside the
    alphabet, and the words of a clause the label derives, each named."""
    if not isinstance(text, str):  # None: the field is absent
        return None if text is None else f"{field_name} {text!r} is no text: a text the label prints is a string"
    reasons = [f"holds {', '.join(chars)}: a text the label prints holds only its alphabet's characters (printable "
               "ASCII but the quote characters, ':', ';' and the backslash)"] if (chars := character_errors(text)) else []
    if kinds := held_clauses(text):
        reasons.append(f"holds the label's derived clause ({', '.join(kinds)}): the label derives it from the run "
                       "(the pricing it resolves, the keys it declares, what does not apply, its statements' "
                       "departures), never from a description's text, read by its words")
    return f"{field_name} {text!r} {', and '.join(reasons)}" if reasons else None


# --- the reader: a label's clauses, outside its quoted fields --------------------------------------------------------

@dataclass(frozen=True)
class LabelReading:
    """A label read outside its quoted fields: its kind, its given settings, its named departures (the typed texts)
    and its derived clauses (each as printed)."""

    kind: str
    given: Tuple[str, ...]
    named: Tuple[str, ...]
    clauses: Tuple[str, ...]


_DECODER = json.JSONDecoder()


def _masked(label: str) -> Tuple[str, Tuple[str, ...]]:
    """``label`` with each quoted field replaced by one placeholder, and the fields' texts in order. A quote that opens
    no complete JSON string refuses (``ValueError``): no label the grammar writes holds one."""
    parts, texts, start = [], [], 0
    while (opening := label.find('"', start)) >= 0:
        text, end = _DECODER.raw_decode(label, opening)
        if not isinstance(text, str):
            raise ValueError(f"label {label!r}: a quote opens no quoted field at {opening}")
        parts.append(label[start:opening] + _FIELD)
        texts.append(text)
        start = end
    parts.append(label[start:])
    return "".join(parts), tuple(texts)


def read_label(label: str) -> LabelReading:
    """[U-CHIPKEYS6] ``label`` read as the grammar writes it: its kind, the given settings of its head, its named
    departures and its derived clauses (each as printed), every quoted field read as typed text and never as a clause.
    A label of another form refuses (``ValueError``)."""
    masked, texts = _masked(label)
    fill = iter(texts)

    def printed(part: str) -> str:
        return "".join(quoted(next(fill)) if char == _FIELD else char for char in part)

    head, has_items, rest = masked.partition(_HEAD)
    kind, has_given, settings = head.partition(", ")
    given = []
    for setting in settings.split(_AND) if has_given else ():
        if setting != f"{GIVEN} {_FIELD}":
            raise ValueError(f"label {label!r}: a given setting is a quoted field ('given \"<text>\"')")
        given.append(next(fill))
    named, clauses = [], []
    for item in rest.split(_ITEM) if has_items else ():
        if item == f"{NAMED} {_FIELD}":
            named.append(next(fill))
        else:
            clauses.append(printed(item))
    if _FIELD in kind:
        raise ValueError(f"label {label!r}: its kind is no quoted field")
    return LabelReading(kind, tuple(given), tuple(named), tuple(clauses))


def derived_clauses(label: Any) -> Tuple[str, ...]:
    """[U-CHIPKEYS6] The derived clauses of a label, read outside its quoted fields (each as printed); () for a value
    that is no text."""
    return read_label(label).clauses if isinstance(label, str) else ()


def clause_kind(text: str) -> Optional[str]:
    """The kind of the template a printed clause was written from (``clause``), None for a text no template writes:
    each literal part in order, each typed slot one quoted field, each key slot a non-empty text with none."""
    masked, _ = _masked(text)
    for kind, template in CLAUSES.items():
        parts, position = list(Formatter().parse(template)), 0
        for index, (literal, slot, _, _) in enumerate(parts):
            if not masked.startswith(literal, position):
                break
            position += len(literal)
            if slot in TYPED_SLOTS:
                if not masked.startswith(_FIELD, position):
                    break
                position += len(_FIELD)
            elif slot is not None:
                following = parts[index + 1][0] if index + 1 < len(parts) else ""
                end = masked.find(following, position + 1) if following else len(masked)
                if end <= position or _FIELD in masked[position:end]:
                    break
                position = end
        else:
            if position == len(masked):
                return kind
    return None
