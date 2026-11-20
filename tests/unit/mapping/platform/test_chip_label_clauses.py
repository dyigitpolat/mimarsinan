"""[U-CHIPKEYS5] The faithfulness label's derived clauses as one grammar (descriptions/label_clauses.py): the writers
format its templates. [U-CHIPKEYS6] M-1 of U-CHIPKEYS5's check, its class closed by construction:

- The judge reads words, not characters: a typed text the label prints in or beside it (a named departure, a given
  setting, the class's note, a display name, a realization's display name, a departing statement's computation and
  value assumed, the profile a description names) refuses at load when its words (compatibility form, case folded,
  split at every character that is no letter or digit, a quote or other mark between two words read joined or split)
  hold the words of a derived clause in order, each slot one or more words. A letter outside the ASCII letters and a
  mark, format or control character refuse by name. The check's forms are red cases: a full stop after the clause,
  typographic and backtick quotes, a name of two words, hyphens between the words, a sentence around the clause.
- The label prints typed text as a quoted field named by its key (``named "<text>"``, ``given "<text>"``, a clause's
  typed slot quoted), its quote character escaped, and the label's reader reads clauses outside quoted fields only.
- The class tested by generators: 2,000 imitations from every template refused at every typed-text site, and 2,000
  clean typed texts that load, print inside their field and leave the reader exactly the resolved pricing's clauses.

m-2 (a realization on a description that states no pricing names the pricing it states), m-3 (the absolute-pricing
refusal's label reads the platform the refusal reads) and m-4 (no shipped note holds a derived clause) are kept.

[U-CHIPKEYS7] M-1 of U-CHIPKEYS6's check (a quote or separator look-alike closes a field to a person's eye), its class
closed by one declared alphabet (``label_clauses.ALPHABET``: printable ASCII without the quote characters, the label's
separators and the backslash), which the judge, the writer's field and these tests read: any other character of a
typed text refuses at load by its code point and name, at its site; the writer prints a field with every character
outside ASCII escaped (a text that reaches the label without the judge). The check's cases: each look-alike at each
site, its 20 break-out texts, ASCII quote pairs used as a closing quote, and the property over every punctuation and
separator character of Unicode and 2,000 generated texts over the ASCII range (refused by name, or printed in a field of
the alphabet with the label's text outside its fields the neutral label's); m-1 (the value assumed judged by its texts
before escaping), m-2 (a value that is no text refuses at a typed-text site) and m-3 (symbol letters refuse by name).
Every case but the shipped-data ones reads hand-made descriptions only."""

import ast
import copy
import json
import random
import re
from pathlib import Path
from string import Formatter

import pytest

import mimarsinan

TARGETS_DIR = Path(mimarsinan.__file__).parent / "mapping" / "platform" / "targets"
REPO = Path(mimarsinan.__file__).resolve().parents[2]
GENERIC = "generic_estimated_22nm"
CONSTANTS = {"e_mac": {"nominal": 1.0, "unit": "pJ", "note": "a test constant"}}
NO_PHYSICS = "no physics profile"
REFUSED_CLAUSE = r"holds the label's derived clause \("


def _clause(priced):
    """The pricing clause a run prints: none, or the profile by name as a quoted field."""
    return NO_PHYSICS if priced is None else f"physics profile {json.dumps(priced)} declared"


def _chip(name="hand", priced=True, faith=None, keys=None, **extra):
    """A hand-made description on a profile of its own (``priced``) or stating no pricing (neither a profile nor its
    absence), with ``keys`` beside its core semantics."""
    from unit.mapping.platform.hand_descriptions import described
    keys = {"core_semantics": "mvm", **({"platform_physics_profile": GENERIC} if priced else {}), **(keys or {})}
    return described({"name": name, "geometry": "recom_64x64_256", "keys": keys,
                      "faithfulness": faith or {"class": "exact" if priced else "refused"}, **extra})


def _realization(**fields):
    return {"geometry": {"name": "hand_probe_geometry", "cores": [{"max_axons": 64, "max_neurons": 64, "count": 2,
                                                                   "has_bias": False}],
                         "weight_bits": 8, "provenance": "a hand-made realization", "claim_class": "quarantined"},
            "computation": {}, "record": {"report": "r.json", "sha256": "0" * 64, "result": "a hand-made record"},
            **fields}


#: A departing statement of the weight grid (its quote read in its own span), the computation the test's.
DEPARTING = {"computation": "a bit times a per-neuron integer", "source": "hand", "location": "this test",
             "quote": "Synaptic weight is a product of two terms: a bit times an integer.",
             "reads": "a product of two terms", "assumed": 8, "reason": "the grid holds one integer per synapse",
             "composition": {"slices_per_weight": {"unstated": {"note": "n"}}, "cell_bits": {"unstated": {"note": "n"}}}}


def _load(tmp_path, data):
    from mimarsinan.mapping.platform.target_bundles import load_bundle
    path = tmp_path / f"{data['name']}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_bundle(path)


def _refused(tmp_path, data, match):
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError
    with pytest.raises(TargetBundleError, match=match):
        _load(tmp_path, data)


# --- (1) the judge reads words: the check's forms, priced or not ---------------------------------------------------

#: Typed forms of the two pricing clauses: the literal, a declared profile, the custom profile, case, spacing, quoting,
#: the clause inside an entry, and [U-CHIPKEYS6] the check's forms (each loaded on revision 5).
PRICING_FORMS = [
    ("no physics profile", "no_pricing"),
    (f"physics profile {GENERIC!r} declared", "pricing"),
    ("physics profile 'custom' declared", "pricing"),
    ("No Physics Profile", "no_pricing"),
    ("PHYSICS PROFILE 'Custom' DECLARED", "pricing"),
    ("no physics  profile", "no_pricing"),
    ("no\tphysics\nprofile", "no_pricing"),
    ("physics  profile 'custom'  declared", "pricing"),
    ('physics profile "custom" declared', "pricing"),
    ("physics profile custom declared", "pricing"),
    ("a run on it finds no physics profile ships", "no_pricing"),
    ("no physics profile.", "no_pricing"),  # a full stop after the clause
    ("physics profile 'vprobe_s' declared.", "pricing"),
    ("the probe chip has no physics profile...", "no_pricing"),
    ("physics profile \u2018vprobe_c\u2019 declared", "pricing"),  # typographic quotes
    ("physics profile \u201cvprobe_c\u201d declared", "pricing"),
    ("physics profile `vprobe_b` declared", "pricing"),  # backtick quotes
    ("physics profile vprobe two declared", "pricing"),  # a name of two words
    ("no-physics-profile", "no_pricing"),  # hyphens between the words
    ("A probe chip with no physics profile.", "no_pricing"),  # a sentence around the clause
    ("A probe chip priced as physics profile 'vprobe_z' declared.", "pricing"),
    ("physics profile'x'declared", "pricing"),  # a quote with no space beside it
    ("no_physics_profile", "no_pricing"),  # underscores between the words
]


@pytest.mark.parametrize("priced", [False, True], ids=["states_no_pricing", "states_its_profile"])
@pytest.mark.parametrize("text, kind", PRICING_FORMS)
def test_a_named_entry_of_a_pricing_clause_s_form_refuses_by_name(text, kind, priced, tmp_path):
    faith = {"class": "approximate", "named": ["a hand-made departure", text]}
    _refused(tmp_path, _chip(priced=priced, faith=faith),
             rf"faithfulness named .* {REFUSED_CLAUSE}[a-z_, ]*{kind}[a-z_, ]*\): the label derives it from the run")


def test_a_realization_s_bundle_naming_a_pricing_clause_refuses(tmp_path):
    """The clause reaches a realization's label through its bundle's faithfulness: the bundle refuses at load."""
    data = _chip(faith={"class": "approximate", "named": ["physics profile \u2018custom\u2019 declared."]})
    data["realizations"] = {"probe": _realization(pricing_absent="a hand-made realization no profile prices")}
    _refused(tmp_path, data, rf"{REFUSED_CLAUSE}pricing")


# --- the sweep: every site that prints a description's typed text in or beside the label, every derived clause -----

#: One typed text of each derived clause's form (each as a description might type it).
DERIVED = {
    "no_pricing": "no physics profile.",
    "pricing": "physics profile \u2018custom\u2019 declared",
    "declared": "weight-bits declared.",
    "not_applied": "weight grid not applied!",
    "departs": "weight_bits departs from its source\u2019s computation `a product`, assumed 8",
}
#: Each typed-text site: where the refusal names it.
SITES = {"named": "faithfulness named", "given": "faithfulness given", "note": "faithfulness note",
         "display_name": "display_name", "realization": "realization 'probe': display_name",
         "departs": "weight_bits: computation", "realization_departs": "realization 'probe': .*weight_bits: computation",
         "profile": "keys.platform_physics_profile", "realization_profile": "realization 'probe': keys.platform_physics_profile"}


def _at_site(site, text):
    """The hand-made description with ``text`` at ``site``: a clause typed in the label (named, given, a departing
    computation, a realization's own profile on a description stating none), or a text printed beside it (the
    class's note, a display name, a realization's display name, the description's profile)."""
    if site == "named":
        return _chip(faith={"class": "approximate", "named": [text]})
    if site == "given":
        return _chip(faith={"class": "exact", "given": [text]})
    if site == "note":
        return _chip(faith={"class": "exact", "note": text})
    if site == "display_name":
        return _chip(display_name=text)
    if site == "profile":
        return _chip(keys={"platform_physics_profile": text})
    if site == "departs":
        data = _chip(faith={"class": "approximate"})
        data["computation"]["weight_bits"] = {"departs": {**DEPARTING, "computation": text}}
        return data
    data = _chip(priced=site == "realization_departs", faith={"class": "approximate", "named": ["a departure of the bundle"]}
                 if site == "realization_departs" else None)
    real = {"realization": {"display_name": text, "pricing_absent": "no profile prices it"},
            "realization_departs": {"pricing_absent": "no profile prices it", "computation": {
                "weight_bits": {"departs": {**DEPARTING, "computation": text}}}},
            "realization_profile": {"keys": {"platform_physics_profile": text}}}[site]
    data["realizations"] = {"probe": _realization(**real)}
    return data


@pytest.mark.parametrize("kind", sorted(DERIVED))
@pytest.mark.parametrize("site", sorted(SITES))
def test_a_typed_text_holding_a_derived_clause_refuses_at_each_site(site, kind, tmp_path):
    _refused(tmp_path, _at_site(site, DERIVED[kind]), rf"{SITES[site]} .*{REFUSED_CLAUSE}[a-z_, ]*{kind}")


@pytest.mark.parametrize("site", sorted(SITES))
@pytest.mark.parametrize("text", [
    "a physics profile is out of scope",  # the pricing clause's words without its form
    "weight_bits departs from its source",  # a departure's head without its form
    "declared at the start of the text",  # a slot needs a word before 'declared'
    "no physics profiles ship",  # a longer word is no word of the clause
    "nophysicsprofile",
    "the profile is not modeled",
    "an applied note, not a clause",
])
def test_a_text_holding_no_derived_clause_loads_at_each_site(site, text, tmp_path):
    bundle = _load(tmp_path, _at_site(site, text))
    assert bundle.name == "hand"


def test_the_value_assumed_by_a_departing_statement_is_judged_by_its_words():
    """The value assumed is printed in the departs clause: [U-CHIPKEYS7] judged by its texts (each string it holds). A
    shipped key's legal values hold no clause, so the case reads the statement's judge directly."""
    from mimarsinan.mapping.platform.descriptions.computation_keys import _departs_errors
    errors = _departs_errors("readout_law", {**DEPARTING, "assumed": "no physics profile"})
    assert any("assumed 'no physics profile' holds the label's derived clause (no_pricing)" in e for e in errors)
    assert not [e for e in _departs_errors("weight_bits", DEPARTING) if "derived clause" in e]


@pytest.mark.parametrize("text, code", [
    ("caf\u00e9 departure", "U+00E9 LATIN SMALL LETTER E WITH ACUTE"),
    ("\uff4eo physics profile", "U+FF4E FULLWIDTH LATIN SMALL LETTER N"),
    ("no phys\u0131cs profile", "U+0131 LATIN SMALL LETTER DOTLESS I"),
    ("\u212aelvin", "U+212A KELVIN SIGN"),
    ("pro\u200bfile", "U+200B ZERO WIDTH SPACE"),
    ("a\u00adb", "U+00AD SOFT HYPHEN"),
    ("no physics pro\u0301file", "U+0301 COMBINING ACUTE ACCENT"),
    ("bell\x07", "U+0007 (unnamed)"),
    ("private \ue000 use", "U+E000 (unnamed)"),
    ("a tab\there", "U+0009 (unnamed)"),
    ("a line\nbreak", "U+000A (unnamed)"),
    ("a return\r\nhere", "U+000A (unnamed) (outside printable ASCII), U+000D"),
    ("plain ASCII 'quoted' text", "U+0027 APOSTROPHE (a quote character)"),
    ('a "quoted" text', "U+0022 QUOTATION MARK (a quote character)"),
    ("a `ticked` text", "U+0060 GRAVE ACCENT (a quote character)"),
    ("a back\\slash", "U+005C REVERSE SOLIDUS (the backslash)"),
])
@pytest.mark.parametrize("site", sorted(SITES))
def test_a_character_outside_the_alphabet_refuses_by_its_code_point_and_name_at_each_site(text, code, site, tmp_path):
    """[U-CHIPKEYS7] One alphabet in place of revision 6's categories: each character outside it, by name, at its site."""
    _refused(tmp_path, _at_site(site, text), rf"{SITES[site]} .*holds .*{re.escape(code)}")


def test_the_alphabet_is_printable_ascii_without_the_label_s_structure():
    """[U-CHIPKEYS7] The declared constant: printable ASCII without every quote character (the double quote, the
    apostrophe, the backtick), the label's separators and the backslash; every other code point of the Basic
    Multilingual Plane is refused by its code point."""
    from mimarsinan.mapping.platform.descriptions.label_clauses import ALPHABET, QUOTES, SEPARATORS, character_errors
    assert QUOTES == frozenset("\"'`") and SEPARATORS == frozenset(":;")
    assert ALPHABET == frozenset(map(chr, range(0x20, 0x7F))) - QUOTES - SEPARATORS - {"\\"} and len(ALPHABET) == 89
    assert character_errors("".join(sorted(ALPHABET))) == ()
    for code in range(0x10000):
        found = character_errors(chr(code))
        if chr(code) in ALPHABET:
            assert found == (), code
        else:
            assert len(found) == 1 and found[0].startswith(f"U+{code:04X} "), (code, found)


def test_no_shipped_typed_text_holds_a_refused_character_or_a_derived_clause():
    """Every shipped description's typed label texts (named, given, note, display names, departing computations and
    the texts of their values assumed, profiles) hold no clause's words (four reworded in revision 6) and [U-CHIPKEYS7]
    only the alphabet's characters (48 reworded: each colon, semicolon, apostrophe and quote of a display name, note,
    named departure and departing computation)."""
    from mimarsinan.mapping.platform.descriptions.label_clauses import ALPHABET, text_error, texts_of
    found = []
    for path in sorted(TARGETS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        faith = data.get("faithfulness") or {}
        texts = [("display_name", data.get("display_name")), ("note", faith.get("note")),
                 ("profile", (data.get("keys") or {}).get("platform_physics_profile")),
                 *(("named", t) for t in faith.get("named", [])), *(("given", t) for t in faith.get("given", []))]
        for spec in [data, *(data.get("realizations") or {}).values()]:
            texts += [("display_name", spec.get("display_name")),
                      ("profile", (spec.get("keys") or {}).get("platform_physics_profile"))]
            for statement in (spec.get("computation") or {}).values():
                if "departs" in statement:
                    texts += [("computation", statement["departs"]["computation"]),
                              *(("assumed", t) for t in texts_of(statement["departs"]["assumed"]))]
        found += [f"{path.name}: {error}" for field, text in texts if (error := text_error(field, text))]
        found += [f"{path.name}: {field} {text!r} outside the alphabet" for field, text in texts
                  if isinstance(text, str) and set(text) - ALPHABET]
    assert found == []


# --- (2) the label prints typed text as a quoted field; its reader reads clauses outside quoted fields ---------------

def test_a_typed_text_prints_as_a_quoted_field_with_its_quote_escaped(tmp_path):
    """[U-CHIPKEYS7] A description's typed text holds no quote character or backslash (refused at load by name); the
    writer still escapes both in a field, for a text no judge reads, and the reader reads it back."""
    from mimarsinan.mapping.platform.descriptions.label_clauses import field, read_label
    text = 'a "quoted" departure \\ with a backslash'
    _refused(tmp_path, _chip(priced=False, faith={"class": "refused", "named": [text]}),
             r"faithfulness named .* holds U\+0022 QUOTATION MARK .*, U\+005C REVERSE SOLIDUS")
    bundle = _load(tmp_path, _chip(priced=False, faith={"class": "refused", "named": ["a (bracketed) departure"],
                                                        "given": ["a setting"]}))
    label = bundle.faithfulness.label()
    assert label == 'refused, given "a setting": no physics profile; named "a (bracketed) departure"'
    reading = read_label(label)
    assert reading.kind == "refused" and reading.given == ("a setting",) and reading.named == ("a (bracketed) departure",)
    assert reading.clauses == (NO_PHYSICS,)
    printed = f"refused: {field('named', text)}"
    assert printed == 'refused: named "a \\"quoted\\" departure \\\\ with a backslash"'
    assert read_label(printed).named == (text,)


def test_each_template_is_written_by_its_kind_and_read_back_as_it():
    """Every derived clause is written outside quotes, each typed slot a quoted field; the reader returns it as a
    clause and names its template; its words hold its own kind."""
    from mimarsinan.mapping.platform.descriptions.label_clauses import (
        CLAUSES, clause, clause_kind, derived_clauses, held_clauses)
    written = {"no_pricing": clause("no_pricing"), "pricing": clause("pricing", name=GENERIC),
               "declared": clause("declared", key="activation_bits"),
               "not_applied": clause("not_applied", key="weight grid"),
               "departs": clause("departs", key="weight_bits", computation='a "product"; of: terms',
                                 assumed=json.dumps({"a": 1}))}
    assert sorted(written) == sorted(CLAUSES)
    assert written["pricing"] == _clause(GENERIC) and written["no_pricing"] == _clause(None)
    assert written["departs"] == ('weight_bits departs from its source\'s computation '  # [U-CHIPKEYS7] ':', ';' escaped
                                  '"a \\"product\\"\\u003b of\\u003a terms", assumed "{\\"a\\"\\u003a 1}"')
    label = "refused: " + "; ".join(written.values())
    assert derived_clauses(label) == tuple(written.values())
    for kind, text in written.items():
        assert clause_kind(text) == kind and kind in held_clauses(text), (kind, text)
    assert derived_clauses(None) == () and held_clauses(8) == () and held_clauses("") == ()
    assert clause_kind('named "no physics profile"') is None


@pytest.mark.parametrize("label", ["exact, given leakage off", 'refused: named "unterminated', 'exact, "kind"'])
def test_a_label_of_another_form_is_no_label_the_reader_reads(label):
    from mimarsinan.mapping.platform.descriptions.label_clauses import read_label
    with pytest.raises(ValueError):
        read_label(label)


@pytest.mark.parametrize("field", ["named", "given"])
@pytest.mark.parametrize("text, match", [
    ("a; b", r"holds U\+003B SEMICOLON \(a label separator\)"),  # [U-CHIPKEYS7] the alphabet names it
    ("a: b", r"holds U\+003A COLON \(a label separator\)"),
    ("", r": a list of .* each named without ':' or ';'"), (" ", r": a list of .* each named without ':' or ';'")])
def test_a_typed_clause_holds_no_label_separator(field, text, match, tmp_path):
    """A typed clause is never empty and holds no ':' or ';' (the label's separators), as in revision 5; [U-CHIPKEYS7]
    a separator is outside the alphabet, refused by its code point."""
    faith = {"class": "approximate", "named": ["a"]} if field == "given" else {"class": "approximate"}
    faith[field] = [text]
    _refused(tmp_path, _chip(faith=faith), rf"faithfulness {field} .*{match}")


@pytest.mark.parametrize("field, value", [("named", "a departure"), ("named", ["a", 1]), ("given", "a setting")])
def test_a_typed_clause_list_is_a_list_of_texts(field, value, tmp_path):
    faith = {"class": "approximate", "named": ["a"], field: value}
    _refused(tmp_path, _chip(faith=faith), rf"faithfulness {field} .*: a list of")


@pytest.mark.parametrize("pc, derived", [
    ({"platform_physics_profile": GENERIC}, [_clause(GENERIC)]),
    ({}, [NO_PHYSICS]),
    ({"platform_physics_overrides": CONSTANTS, "target_overrides": {
        "activation_bits": {"value": 4, "note": "a test override"}}}, [_clause("custom"), "activation_bits declared"]),
])
def test_the_run_door_s_label_reads_back_its_derived_clauses_and_its_typed_fields(pc, derived, tmp_path,
                                                                                  monkeypatch):
    """The run door's label: its derived clauses read back outside the quoted fields, the description's typed texts
    inside them, whatever they hold."""
    from mimarsinan.mapping.platform import target_bundles as bundle_module
    from mimarsinan.mapping.platform.descriptions.label_clauses import read_label
    from mimarsinan.mapping.platform.target_expansion import expand_target
    typed = ["a hand-made departure, (bracketed) and / slashed", "physics profile is a phrase here"]
    data = _chip(priced=False, faith={"class": "refused", "named": typed, "given": ["a setting of the test"]},
                 keys={"activation_bits": 8})
    known = {"hand": _load(tmp_path, data)}
    monkeypatch.setattr(bundle_module, "target_bundles", lambda: known)
    label = expand_target({}, {"target": "hand", **pc}, write_derived=True)[1]["target_faithfulness_class"]
    reading = read_label(label)
    assert reading.clauses == tuple(derived) and reading.named == tuple(typed)
    assert reading.given == ("a setting of the test",)


# --- (3) the class tested by generators ------------------------------------------------------------------------------

#: Quote pairs of every kind (open, close): ASCII, typographic, angle, low-9, corner, prime, fullwidth and backticks.
QUOTES = [("'", "'"), ('"', '"'), ("`", "`"), ("\u2018", "\u2019"), ("\u201c", "\u201d"), ("\u00ab", "\u00bb"),
          ("\u2039", "\u203a"), ("\u201a", "\u2019"), ("\u201e", "\u201c"), ("\u300c", "\u300d"), ("\u300e", "\u300f"),
          ("\u301d", "\u301e"), ("\uff02", "\uff02"), ("\uff07", "\uff07"), ("\u2032", "\u2032"), ("\u2033", "\u2033"),
          ("``", "''"), ("\u201b", "\u2019"), ("\u201f", "\u201d"), ("", "")]
#: Separators between two words of a clause: whitespace of every kind, line breaks, hyphens, punctuation and quotes.
GAPS = [" ", "  ", "\t", "\n", "\r\n", " \t ", "\u00a0", "\u3000", "\u2003", "-", " - ", "--", "_", ", ", ". ", "/",
        " (", ") ", "...", "\u2026", "\u2019 ", " \u2018", "`", ": ", "; ", "!", "?", "\u2014"]
#: Texts around a clause: before it and after it.
BEFORE = ["", "", "A probe chip with ", "(", "\u201c", "Note: ", "- ", "the run finds ", "1. ", "`", "'"]
AFTER = ["", "", ".", "!", "?", "...", ")", "\u201d", " here.", "; more text", ", as typed", ".\n", "`", "'", "--"]
#: Slot fillers: one or more words, keys and names of every form.
FILLERS = ["probe", "vprobe_7nm", "the chip", "weight_bits", "cores.max_axons", "x-y", "two words here", "v1.2",
           "custom", "generic_estimated_22nm", "binary synapse", "8", "true", "a", "weight grid"]


def _cased(rng, text):
    return "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in text) if rng.random() < 0.5 else (
        rng.choice([text, text.upper(), text.title()]))


def _imitation(rng, template):
    """One typed imitation of ``template``: its words in order, each slot one or more words (quoted or not), random
    case, spacing, quotes and punctuation between and around them."""
    parts = []
    for literal, slot, _, _ in Formatter().parse(template):
        for word in literal.split():
            word = word.rstrip(",")
            if "'" in word:  # an apostrophe in a word: any quote character in its place
                word = word.replace("'", rng.choice(["'", "\u2019", "`", "\u2032", "\uff07"]))
            parts.append(_cased(rng, word))
        if slot is not None:
            opening, closing = rng.choice(QUOTES)
            filler = rng.choice(FILLERS)
            filler = filler.replace(" ", rng.choice(GAPS[:9])) if rng.random() < 0.5 else filler
            parts.append(f"{opening}{_cased(rng, filler)}{closing}")
    text = parts[0]
    for part in parts[1:]:
        text += rng.choice(GAPS) + part
    return rng.choice(BEFORE) + text + rng.choice(AFTER)


def _imitations(count=2000, seed=20261112):
    """``count`` imitations, the templates taken in turn (each kind count / 5 times)."""
    from mimarsinan.mapping.platform.descriptions.label_clauses import CLAUSES
    rng, kinds = random.Random(seed), sorted(CLAUSES)
    return [(kinds[i % len(kinds)], _imitation(rng, CLAUSES[kinds[i % len(kinds)]])) for i in range(count)]


def test_the_imitations_cover_every_template_and_form():
    """The generator's own check (no judge read): 2,000 imitations, 400 per template, every quote pair, gap, text
    before and after taken."""
    imitations = _imitations()
    assert len(imitations) == 2000 and len({text for _, text in imitations}) > 1950
    assert {kind: sum(k == kind for k, _ in imitations) for kind, _ in imitations} == dict.fromkeys(
        ("declared", "departs", "no_pricing", "not_applied", "pricing"), 400)
    joined = "".join(text for _, text in imitations)
    assert all(opening in joined for opening, _ in QUOTES if opening) and all(gap in joined for gap in GAPS)


@pytest.mark.parametrize("site", sorted(SITES))
def test_every_imitation_of_every_template_refuses_at_each_site(site, tmp_path):
    """The property: each of 2,000 generated imitations, placed at the site, is a named refusal of the class."""
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError, load_bundle
    loaded = []
    for index, (kind, text) in enumerate(_imitations()):
        data = _at_site(site, text)
        path = tmp_path / "hand.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        try:
            load_bundle(path)
            loaded.append((index, kind, text))
        except TargetBundleError as exc:
            assert "holds the label's derived clause (" in str(exc), (index, kind, text, str(exc)[:300])
    assert loaded == []


#: The clean texts' words: a vocabulary that holds some words of every clause and never all of any (no 'no', 'not',
#: 'declared', 'departs', 'assumed'), so no text of it holds a clause; each joining of words checked below.
VOCABULARY = ["analog", "error", "modeled", "lanes", "estimated", "cell", "array", "energy", "read", "write", "the",
              "a", "of", "per", "chip", "core", "weight", "bits", "grid", "board", "host", "macro", "tile", "study",
              "band", "sourced", "value", "law", "physics", "profile", "applied", "from", "its", "source", "computation",
              "synapse", "binary", "integer", "type", "axon", "neuron", "leakage", "learning", "off", "by", "deployment",
              "ADC", "FP16", "W47", "8-bit", "x", "it", "and", "with", "in", "on"]
EXCLUDED = ("no", "not", "declared", "departs", "assumed")
#: [U-CHIPKEYS7] Joints of the alphabet only (revision 6's tab, quotes, backslash and backticks are refused by name).
CLEAN_GAPS = [" ", "  ", " - ", ", ", " (", ") ", " / ", " | ", ". ", " [q] ", " <q> ", " & ", " {b} "]


def _clean_texts(count=2000, seed=20261113):
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        words = [rng.choice(VOCABULARY) for _ in range(rng.randint(1, 9))]
        text = words[0]
        for word in words[1:]:
            text += rng.choice(CLEAN_GAPS) + word
        out.append(_cased(rng, text) if rng.random() < 0.3 else text)
    return out


def test_the_clean_vocabulary_spells_no_clause_word_in_any_joining():
    """The generator's own check: no word of it and no joining of two or three of its words (a quote or mark between
    them read as dropped) is a word the vocabulary leaves out, so no clean text holds every word of a clause."""
    import itertools
    from mimarsinan.mapping.platform.descriptions.label_clauses import templates
    folded = ["".join(c for c in w.casefold() if c.isalnum()) for w in VOCABULARY]
    pieces = {p for w in VOCABULARY for p in "".join(c if c.isalnum() else " " for c in w.casefold()).split()}
    joined = {"".join(c) for n in (1, 2, 3) for c in itertools.product(sorted(pieces | set(folded)), repeat=n)
              if len("".join(c)) <= 9}
    assert not joined & set(EXCLUDED)
    assert all(any(token in EXCLUDED for token in tokens if token) for _, tokens in templates())


@pytest.mark.parametrize("chunk", range(4))
def test_every_clean_typed_text_loads_prints_in_its_field_and_leaves_the_pricing_s_clauses(chunk, tmp_path):
    """The property's other half: each of 2,000 generated typed texts holding no clause's words loads at every site
    of a description stating no pricing (named, given, note, display name, a departing computation), prints inside its
    quoted field, and the label's reader returns exactly the derived clauses of the pricing each run resolves."""
    from mimarsinan.mapping.platform.descriptions.departures import run_faithfulness
    from mimarsinan.mapping.platform.descriptions.label_clauses import clause, field, read_label
    runs = [({}, [NO_PHYSICS]), ({"platform_physics_profile": GENERIC}, [_clause(GENERIC)]),
            ({"platform_physics_overrides": CONSTANTS}, [_clause("custom")])]
    for text in _clean_texts()[chunk * 500:(chunk + 1) * 500]:
        data = _chip(priced=False, faith={"class": "refused", "named": [text], "given": [text], "note": text},
                     display_name=text)
        data["computation"]["weight_bits"] = {"departs": {**DEPARTING, "computation": text}}
        bundle = _load(tmp_path, data)
        departs = clause("departs", key="weight_bits", computation=text, assumed="8")
        for declared, pricing in runs:
            label = run_faithfulness(bundle, declared).label()
            assert field("named", text) in label and field("given", text) in label, (text, label)
            reading = read_label(label)
            assert reading.named == (text,) and reading.given == (text,), (text, label)
            assert reading.clauses == (*pricing, departs), (text, label)


# --- the writers swept: every derived clause is written from the grammar's templates ------------------------------

def _shape(node):
    """An f-string's text with each formatted value as '{}' (a plain text as itself)."""
    if isinstance(node, ast.Constant):
        return node.value
    return "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)


def test_no_module_writes_a_derived_clause_but_the_grammar():
    """The class swept over the writers: no text of src or scripts outside label_clauses.py takes a template's shape
    (an f-string formatting a template's fields, or a template without fields), so each derived clause the label
    prints is written by the grammar."""
    from mimarsinan.mapping.platform.descriptions.label_clauses import CLAUSES
    shapes = {"".join(lit + ("{}" if field is not None else "") for lit, field, _, _ in Formatter().parse(t))
              for t in CLAUSES.values()}
    found = []
    for path in sorted([*(REPO / "src" / "mimarsinan").rglob("*.py"), *(REPO / "scripts").rglob("*.py")]):
        if path.name == "label_clauses.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.JoinedStr) or (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                text = _shape(node)
                if any(text == s or (isinstance(node, ast.JoinedStr) and text.startswith(s)) for s in shapes):
                    found.append(f"{path.relative_to(REPO)}:{node.lineno}: {text!r}")
    assert found == []


def test_no_module_prints_a_typed_text_in_the_label_but_the_grammar():
    """[U-CHIPKEYS6] The class swept over the printers: a Faithfulness is built with a typed text only by the faithfulness
    judge, which prints each named departure as a quoted field (label_clauses.field); every other constructor call of
    src and scripts rebuilds a stored one (the realization's lookup)."""
    calls = []
    for path in sorted([*(REPO / "src" / "mimarsinan").rglob("*.py"), *(REPO / "scripts").rglob("*.py")]):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", None)) == "Faithfulness":
                calls.append(f"{path.relative_to(REPO / 'src' / 'mimarsinan') if 'src' in path.parts else path.name}")
    assert sorted(calls) == ["mapping/platform/descriptions/departures.py", "mapping/platform/descriptions/realizations.py"]


#: Every writer of a derived clause or of a typed field outside the grammar: (file, the call, its kind or key).
WRITERS = sorted([
    ("mapping/platform/descriptions/computation_keys.py", "clause", "departs"),
    ("mapping/platform/descriptions/departures.py", "clause", "no_pricing"),
    ("mapping/platform/descriptions/departures.py", "clause", "pricing"),
    ("mapping/platform/descriptions/departures.py", "field", "NAMED"),
    ("mapping/platform/target_bundles.py", "clause", "declared"),
    ("mapping/platform/target_bundles.py", "field", "GIVEN"),
    ("mapping/platform/target_expansion.py", "clause", "not_applied"),
    ("mapping/platform/target_expansion.py", "clause", "not_applied"),
    ("scripts/design_point_instance.py", "clause", "declared"),
    ("scripts/design_point_instance.py", "clause", "declared"),
])


def test_every_writer_of_a_clause_or_a_typed_field_is_a_call_of_the_grammar_at_a_known_site():
    """[U-CHIPKEYS6] The sweep of the writers, read from the code: each call of label_clauses.clause and .field in src
    and scripts, by file and kind. A new writer fails here until it is swept (its site added to the label tests)."""
    found = []
    for path in sorted([*(REPO / "src" / "mimarsinan").rglob("*.py"), *(REPO / "scripts").rglob("*.py")]):
        if path.name == "label_clauses.py":
            continue
        rel = (path.relative_to(REPO / "src" / "mimarsinan") if "src" in path.relative_to(REPO).parts[:1]
               else path.relative_to(REPO)).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {alias.asname or alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                    and (node.module or "").endswith("label_clauses") for alias in node.names}
        for node in ast.walk(tree):
            func = node.func if isinstance(node, ast.Call) else None
            name = (func.attr if isinstance(func, ast.Attribute) and getattr(func.value, "id", None) == "label_clauses"
                    else func.id if isinstance(func, ast.Name) and func.id in imported else None)
            if name in ("clause", "field") and node.args:
                first = node.args[0]
                kind = first.value if isinstance(first, ast.Constant) else getattr(first, "attr", ast.dump(first))
                found.append((rel, name, kind))
    assert sorted(found) == WRITERS


def _script(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"{name}_label_sites", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TYPED = ["a hand-made departure, (bracketed) and / slashed", "a phrase about a physics profile"]


@pytest.mark.parametrize("site", ["door", "absolute_refusal", "ladder", "chip_block"])
def test_each_label_site_prints_typed_texts_as_quoted_fields_and_its_clauses_outside(site, tmp_path, monkeypatch):
    """[U-CHIPKEYS6] One case per label site: the description's typed texts read back from the printed label inside their
    quoted fields, and its derived clauses (the pricing the site resolves) outside them."""
    from mimarsinan.mapping.platform import target_bundles as bundle_module
    from mimarsinan.mapping.platform.descriptions.label_clauses import read_label
    data = _chip(priced=False, faith={"class": "refused", "named": TYPED, "given": ["a setting of the test"]})
    known = {**bundle_module.target_bundles(), "hand": _load(tmp_path, data)}
    monkeypatch.setattr(bundle_module, "target_bundles", lambda: known)
    if site == "door":
        from mimarsinan.mapping.platform.target_expansion import expand_target
        label = expand_target({}, {"target": "hand", "platform_physics_profile": GENERIC}, write_derived=True)[1][
            "target_faithfulness_class"]
        clauses = (_clause(GENERIC),)
    elif site == "absolute_refusal":
        reason = bundle_module.absolute_pricing_refusal({"target": "hand", "platform_physics_resolved": None})[1]
        label = reason[reason.index("(") + 1:reason.rindex("): no area")]
        clauses = (NO_PHYSICS,)
    elif site == "ladder":
        ladder = _script("scaling_ladder")
        monkeypatch.setitem(ladder.TARGETS, "hand", "hand")
        label = ladder.target_spec("hand", {"platform_physics_overrides": CONSTANTS})["bundle"]["faithfulness"]
        clauses = (_clause("custom"),)
    else:
        driver = _script("design_point_instance")
        monkeypatch.setattr(driver, "bundle_document", lambda name: data)
        block = driver.chip_block({}, {"target": "hand", "arithmetic_class": "a test class"}, None)["faithfulness"]
        label = f"{block['class']}: {'; '.join(block['named'])}"
        clauses = (NO_PHYSICS,)
    reading = read_label(label)
    assert reading.named == tuple(TYPED) and reading.clauses == clauses, label
    assert reading.given == (("a setting of the test",) if site != "chip_block" else ())


# --- m-2: a realization on a description that states no pricing -----------------------------------------------------

@pytest.mark.parametrize("pricing, priced", [({"keys": {"platform_physics_profile": GENERIC}}, GENERIC),
                                             ({"pricing_absent": "a hand-made realization no profile prices"}, None)],
                         ids=["its_own_profile", "pricing_absent"])
def test_a_realization_on_a_description_stating_no_pricing_loads_and_names_its_own_pricing(pricing, priced, tmp_path):
    """The verify's probe C: revision 4 refused it at load (its bundle's class rested on its derived pricing alone)."""
    from mimarsinan.mapping.platform.target_bundles import get_target_bundle
    from mimarsinan.mapping.platform.target_expansion import expand_target
    data = _chip(priced=False)
    data["realizations"] = {"probe": _realization(**copy.deepcopy(pricing))}
    known = {"hand": _load(tmp_path, data)}
    real = get_target_bundle("hand@probe", known)
    assert real.faithfulness.label() == f"refused: {_clause(priced)}"
    assert known["hand"].faithfulness.label() == f"refused: {NO_PHYSICS}"
    _, pc = expand_target({}, {"target": "hand@probe"}, write_derived=True, bundles=known)
    assert pc["target_faithfulness_class"] == f"refused: {_clause(priced)}"


def test_a_realization_on_a_description_stating_its_pricing_names_no_pricing(tmp_path):
    from mimarsinan.mapping.platform.target_bundles import get_target_bundle
    data = _chip()
    data["realizations"] = {"probe": _realization(pricing_absent="a hand-made realization no profile prices")}
    assert get_target_bundle("hand@probe", {"hand": _load(tmp_path, data)}).faithfulness.label() == "exact"


# --- m-3: the absolute-pricing refusal reads one platform ----------------------------------------------------------

@pytest.fixture
def unpriced(tmp_path, monkeypatch):
    from mimarsinan.mapping.platform import target_bundles as bundle_module
    known = {**bundle_module.target_bundles(), "hand": _load(tmp_path, _chip(priced=False))}
    monkeypatch.setattr(bundle_module, "target_bundles", lambda: known)
    return "hand"


@pytest.mark.parametrize("declared", [{"platform_physics_profile": GENERIC}, {"platform_physics_overrides": CONSTANTS},
                                      {"platform_physics_profile": GENERIC, "platform_physics_resolved": None}, {}])
def test_the_absolute_pricing_refusal_s_label_reads_the_platform_the_refusal_reads(declared, unpriced):
    """The verify's probe G: a platform never resolved (or resolved to none) is unpriced, and its label says so."""
    from mimarsinan.mapping.platform.target_bundles import absolute_pricing_refusal
    assert absolute_pricing_refusal({"target": unpriced, **declared}) == (
        "absolute_pricing", f"target={unpriced!r} declares no physics profile (refused: {NO_PHYSICS}): no area, "
                            "energy or latency is priced")


# --- m-4: the shipped notes hold no derived clause -------------------------------------------------------------------

def test_no_shipped_note_states_a_pricing_the_label_derives():
    """The chip block prints the note beside the label of a priced run: no note holds any derived clause's words."""
    from mimarsinan.mapping.platform.descriptions.label_clauses import held_clauses
    from mimarsinan.mapping.platform.target_bundles import get_target_bundle, states_pricing, target_names
    unpriced = []
    for name in target_names():
        bundle = get_target_bundle(name)
        note = json.loads((TARGETS_DIR / f"{name.split('@')[0]}.json").read_text(encoding="utf-8"))["faithfulness"].get(
            "note", "")
        assert held_clauses(note) == (), name
        if not states_pricing(bundle.values, bundle.pricing_absent):
            unpriced.append(name)
    assert len(unpriced) == 10


# --- the words: the normalization's parts, each with its case ------------------------------------------------------

@pytest.mark.parametrize("text, found", [
    ("No-Physics_Profile.", (("no", "physics", "profile"), (True, True))),  # case folded, marks soft joints
    ("a  b\tc", (("a", "b", "c"), (False, False))),  # whitespace a hard joint
    ("x \u2019y", (("x", "y"), (False,))),  # whitespace beside a mark: hard
    ("\uff41\u00a0b", (("a", "b"), (False,))),  # the compatibility form: a fullwidth letter, a no-break space
    ("", ((), ())),
])
def test_the_words_of_a_text_are_its_letters_and_digits_with_their_joints(text, found):
    from mimarsinan.mapping.platform.descriptions.label_clauses import words
    assert words(text) == found


@pytest.mark.parametrize("text, kinds", [
    ("source's computation", ()),
    ("w departs from its source's computation x, assumed y", ("departs",)),
    ("w departs from its sources computation x assumed y", ("departs",)),  # the apostrophe dropped
    ("w departs from its source\u2019s computation x assumed y", ("departs",)),
    ("w departs from its source s computation x assumed y", ()),  # a space is no dropped mark
    ("w departs from its source's computation x, assumed", ()),  # the value's slot needs a word
    ("physics profile declared", ("declared",)),  # no name: the pricing clause's slot needs a word, the key's holds two
    ("not applied", ()),  # the key's slot needs a word
    ("x not applied", ("not_applied",)),
    ("pro'file", ()), ("no physics pro'file", ("no_pricing",)),  # a mark inside a word: joined
])
def test_a_slot_is_one_or_more_words_and_a_mark_joins_or_splits(text, kinds):
    from mimarsinan.mapping.platform.descriptions.label_clauses import held_clauses
    assert held_clauses(text) == kinds


# --- [U-CHIPKEYS7] (4) the typed text's alphabet: the check's cases, the property, m-1 to m-3 ------------------------

#: The check's quote and separator look-alikes (U-CHIPKEYS6's check, M-1), and the line and paragraph separators.
LOOKALIKES = ["\uff02", "\u05f4", "\u2033", "\u3003", "\u02dd", "\u201c", "\u201d", "\u201e", "\u201f", "\u275d",
              "\u275e", "\u301d", "\u301e", "\u301f", "\u037e", "\uff1b", "\ufe54", "\uff1a", "\u2028", "\u2029"]
#: The sites whose typed text the label prints inside it: a named departure, a given setting, a departing computation
#: of a bundle and of a realization, and a realization's own profile on a description stating no pricing.
IN_LABEL = ["departs", "given", "named", "realization_departs", "realization_profile"]
#: The check's 20 break-out texts (its generator, seed 6006: on revision 6 each loaded, closed its field to a person's
#: eye and printed a second pricing after it), then its two hand texts.
BREAKOUTS = [
    "a probe departure\u3003 - given NoPhysicsprofile. - named \uff02b",
    "a probe departure\u05f4\uff1a The chip has physics profile <generic_estimated_22nm> dec1ared -- end\uff1a named \uff02b",
    "a probe departure\u05f4\uff1b NophysicsProfile...\uff1b named \\\"b",
    "a probe departure\u201c: * physics\u2022profile _lp_ applies (see the notes): named \u201db",
    "a probe departure\u201f, The chip has Nophysicsprofile here, named \u201db",
    "a probe departure\u275e - Per W47: physics profile `my own 7nm profile`\uff61stated - named \u201cb",
    "a probe departure\u05f4\u2028 (physics profile -- \u275dlp\u275e declare d.\n\u2028 named \u05f4b",
    "a probe departure\u275e\n As the source states, NophysicsProfile...\n named \u201db",
    "a probe departure\u275e\uff1a \u00abnoPhysicsprofile?!\uff1a named \\\"b",
    "a probe departure\u275e ; named noPhysicsProfile\t ; named \u275eb",
    "a probe departure\uff02\uff1a * n0; physics/profi1e -- end\uff1a named \u201cb",
    "a probe departure\u05f4\ufe54 named Nophysicsprofile and more\ufe54 named \\\"b",
    "a probe departure\u201f\n [without physics [profile\n named \\\"b",
    "a probe departure\uff02\ufe54 physics'profile%\u301dmy own 7nm profile\u301e.declare d and more\ufe54 named \u201fb",
    "a probe departure\u3003; # without physics\u00b7profile here; named \u201fb",
    "a probe departure\u275e, named nophysicsprofile, named \u05f4b",
    "a probe departure\u275e\u037e refused, zero-physics\u3000profile!\u037e named \u05f4b",
    "a probe departure\u3003\uff1a \u00abNoPhysicsprofile?\uff1a named \\\"b",
    "a probe departure\u3003\n given physicsprofilexDeclared...\n named \u3003b",
    "a probe departure\u201d  \nNoPhysicsprofile'  named \u3003b",
    "a probe departure\uff02\uff1b no physics profiIe\uff1b named \uff02b",
    "a probe departure\uff02\u037e physics profile \uff02custom\uff02 in force\u037e named \uff02b",
]
#: The neutral typed text: the label it prints is the one each generated text's label is compared with outside fields.
NEUTRAL = "a probe departure"
#: A JSON string as printed (read here independently of the unit's reader).
_PRINTED_FIELD = re.compile(r'"(?:[^"\\]|\\.)*"')


def _refusal(site, text, tmp_path):
    """The refusal of ``text`` at ``site`` (its message), or the loaded probe."""
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError
    try:
        return _load(tmp_path, _at_site(site, text))
    except TargetBundleError as exc:
        return str(exc)


@pytest.mark.parametrize("char", LOOKALIKES, ids=lambda c: f"U+{ord(c):04X}")
@pytest.mark.parametrize("site", sorted(SITES))
def test_a_quote_or_separator_look_alike_refuses_by_name_at_each_site(site, char, tmp_path):
    """The check's cases: each look-alike quote and separator it lists, used to close the field and separate a clause,
    refuses at load by its code point at each site (revision 6 loaded every one where no clause's words were read)."""
    text = f"a probe departure{char} no physics profiIe{char} named {char}b"
    _refused(tmp_path, _at_site(site, text), rf"{SITES[site]} .*holds .*U\+{ord(char):04X} ")


@pytest.mark.parametrize("site", sorted(SITES))
def test_the_check_s_break_out_texts_refuse_by_name_at_each_site(site, tmp_path):
    """The check's 20 break-out texts and its two hand texts: each refused at every site, each of its characters
    outside the alphabet named by its code point."""
    from mimarsinan.mapping.platform.descriptions.label_clauses import ALPHABET
    for text in BREAKOUTS:
        message = _refusal(site, text, tmp_path)
        assert isinstance(message, str), (site, text)
        assert all(f"U+{ord(c):04X} " in message for c in set(text) - ALPHABET), (site, text, message[:400])


@pytest.mark.parametrize("quote", ["'", "`", '"'], ids=["apostrophe", "backtick", "double_quote"])
@pytest.mark.parametrize("site", sorted(SITES))
def test_an_ascii_quote_pair_used_as_a_closing_quote_refuses_by_name(site, quote, tmp_path):
    """An apostrophe pair, a backtick pair and a double quote pair, each used as a closing quote and an opening one around
    a pricing: refused by its code point (revision 6 printed the first two raw and the third escaped)."""
    text = f"a probe departure{quote} - no physics profiIe - named {quote}b"
    _refused(tmp_path, _at_site(site, text), rf"{SITES[site]} .*holds .*U\+{ord(quote):04X} ")


def _readings(site, bundle):
    """The labels a loaded probe prints: for a bundle site its own and each run's (none, the generic profile,
    constants), for a realization site the realization's own (it states its pricing)."""
    from mimarsinan.mapping.platform.descriptions.departures import run_faithfulness
    from mimarsinan.mapping.platform.target_bundles import get_target_bundle
    if site.startswith("realization"):
        return [get_target_bundle("hand@probe", {"hand": bundle}).faithfulness.label()]
    return [bundle.faithfulness.label()] + [run_faithfulness(bundle, declared).label() for declared in (
        {}, {"platform_physics_profile": GENERIC}, {"platform_physics_overrides": CONSTANTS})]


def _refused_by_name_or_printed(site, text, tmp_path, neutral):
    """The property for one text at one site: refused by name (each character outside the alphabet by its code
    point, else the clause its words hold, else a blank text), or loaded and printed with every field of the
    alphabet, the label ASCII and its text outside its fields the neutral label's. Returns the outcome."""
    from mimarsinan.mapping.platform.descriptions.label_clauses import ALPHABET, held_clauses
    found, outside = _refusal(site, text, tmp_path), set(text) - ALPHABET
    if isinstance(found, str):
        assert all(f"U+{ord(c):04X} " in found for c in outside), (site, text, found[:400])
        assert outside or "holds the label's derived clause (" in found or not text.strip(), (site, text, found[:400])
        return "refused by its characters" if outside else "refused by its words"
    assert not outside and not held_clauses(text), (site, text)
    for label, base in zip(_readings(site, found), neutral):
        fields = _PRINTED_FIELD.findall(label)
        assert label.isascii() and json.dumps(text) in fields, (site, text, label)
        assert all(set(f[1:-1]) <= ALPHABET for f in fields), (site, text, label)
        assert _PRINTED_FIELD.sub("<field>", label) == _PRINTED_FIELD.sub("<field>", base), (site, text, label, base)
    return "loaded"


def _punctuation():
    """Every character of Unicode's punctuation categories (its quotation marks among them) and separator categories,
    with the check's look-alikes (one of them a modifier symbol)."""
    import sys
    import unicodedata
    found = {chr(c) for c in range(sys.maxunicode + 1) if unicodedata.category(chr(c))[0] in "PZ"}
    return sorted(found | set(LOOKALIKES))


@pytest.mark.parametrize("site", IN_LABEL)
def test_every_punctuation_and_separator_character_refuses_by_name_or_prints_in_a_field_of_the_alphabet(site, tmp_path):
    """The property over characters: each punctuation or separator character of Unicode, closing the field and
    separating a clause, is refused by its code point or (the alphabet's own) prints inside its field, the label's
    text outside its fields the neutral label's."""
    from mimarsinan.mapping.platform.descriptions.label_clauses import ALPHABET
    chars = _punctuation()
    assert len(chars) > 800 and set(LOOKALIKES) <= set(chars)
    neutral = _readings(site, _refusal(site, NEUTRAL, tmp_path))
    outcomes = [(_refused_by_name_or_printed(site, f"{NEUTRAL}{c} no physics profiIe{c} named {c}b", tmp_path,
                                             neutral), c) for c in chars]
    assert {c for outcome, c in outcomes if outcome == "loaded"} == {c for c in chars if c in ALPHABET}


#: The clause words the ASCII generator draws beside its characters (the apostrophe of "source's" dropped, as the
#: judge reads it, so a departs clause of the alphabet is drawn too).
CLAUSE_WORDS = ["no", "physics", "profile", "declared", "not", "applied", "departs", "from", "its", "sources",
                "computation", "assumed", "named", "given", "custom"]


def _ascii_texts(count=2000, seed=20261114):
    """``count`` texts over the full ASCII range: after the neutral text, two to nine parts, each a clause word or a
    character drawn from U+0000 to U+007F."""
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        parts = [rng.choice(CLAUSE_WORDS) + " " if rng.random() < 0.4 else chr(rng.randrange(128))
                 for _ in range(rng.randint(2, 9))]
        out.append(f"{NEUTRAL} {''.join(parts)}")
    return out


def test_the_ascii_texts_cover_the_ascii_range():
    """The generator's own check (no judge read): 2,000 distinct enough texts, every ASCII code point taken."""
    texts = _ascii_texts()
    assert len(texts) == 2000 and len(set(texts)) > 1900
    assert {chr(c) for c in range(128)} <= set("".join(texts))


@pytest.mark.parametrize("site", IN_LABEL)
def test_every_generated_ascii_text_refuses_by_name_or_prints_in_a_field_of_the_alphabet(site, tmp_path):
    """The property over texts: each of 2,000 generated texts over the ASCII range, at the site, is refused by name or
    prints inside its field of the alphabet with the label's text outside its fields the neutral label's; each
    outcome occurs."""
    neutral = _readings(site, _refusal(site, NEUTRAL, tmp_path))
    outcomes = {_refused_by_name_or_printed(site, text, tmp_path, neutral) for text in _ascii_texts()}
    assert outcomes == {"refused by its characters", "refused by its words", "loaded"}


def test_the_writer_escapes_every_character_outside_the_alphabet_in_a_field():
    """The second line: a text that reaches the label without the load judge (a run's declared profile name, a
    Faithfulness built in code) prints with every character outside the alphabet escaped (outside ASCII, the quote
    characters, the separators, the backslash), and the reader reads it back; every code point of the Basic
    Multilingual Plane, each alone in a field, prints the alphabet's characters and escapes only."""
    from mimarsinan.mapping.platform.descriptions.departures import pricing_departures
    from mimarsinan.mapping.platform.descriptions.label_clauses import field, quoted, read_label
    from mimarsinan.mapping.platform.target_bundles import Faithfulness
    name = "a probe\uff02\uff1b no physics profile\uff1b named \uff02b"
    printed = pricing_departures(False, name)[0]
    assert printed == 'physics profile "a probe\\uff02\\uff1b no physics profile\\uff1b named \\uff02b" declared'
    label = Faithfulness("approximate", (field("named", "x\u037ey\u2028z"), printed), ("caf\u00e9",)).label()
    assert label.isascii() and quoted("\U0001f17d") == '"\\ud83c\\udd7d"'
    reading = read_label(label)
    assert reading.named == ("x\u037ey\u2028z",) and reading.given == ("caf\u00e9",) and reading.clauses == (printed,)
    structure = "a; b: 'c' `d` \"e\" \\f"
    assert field("named", structure) == 'named "a\\u003b b\\u003a \\u0027c\\u0027 \\u0060d\\u0060 \\"e\\" \\\\f"'
    assert read_label(f"refused: {field('named', structure)}").named == (structure,)
    from mimarsinan.mapping.platform.descriptions.label_clauses import ALPHABET
    escape = re.compile(r'\\(?:u[0-9a-f]{4}|["\\/bfnrt])')
    for code in range(0x10000):
        printed = quoted(chr(code))
        assert printed[0] == printed[-1] == '"' and set(escape.sub("", printed[1:-1])) <= ALPHABET, (code, printed)
        assert json.loads(printed) == chr(code), code


@pytest.mark.parametrize("assumed, names, kinds", [
    ("no\u00a0physics profile", ["U+00A0 NO-BREAK SPACE"], ["no_pricing"]),  # m-1: a non-ASCII gap
    ({"output_law": "floor\uff1a no physics profile"}, ["U+FF1A FULLWIDTH COLON"], ["no_pricing"]),
    (["a", "b\u037e"], ["U+037E GREEK QUESTION MARK"], []),
    ("Default", [], []), ({"output_law": "floor", "range": 4}, [], []), ([1, 2], [], []), (8, [], []), (True, [], []),
])
def test_the_value_assumed_is_judged_by_its_texts_before_escaping(assumed, names, kinds):
    """m-1: each text the value assumed holds (itself for a text; each string of a list or an object, its keys
    included) is judged raw, before the writer escapes it; the JSON text's own quotes and ':' are the writer's, so a
    legal value of every form loads."""
    from mimarsinan.mapping.platform.descriptions.computation_keys import _departs_errors
    from mimarsinan.mapping.platform.descriptions.label_clauses import texts_of
    errors = [e for e in _departs_errors("readout_law", {**DEPARTING, "assumed": assumed}) if ": assumed " in e]
    assert all(any(name in e for e in errors) for name in names), errors
    assert all(any(f"derived clause ({kind})" in e for e in errors) for kind in kinds), errors
    assert bool(errors) == bool(names or kinds), errors
    assert texts_of({"a": ["b", {"c": 1}]}) == ("a", "b", "c") and texts_of(8) == () and texts_of("x") == ("x",)


@pytest.mark.parametrize("value", [["no physics profile"], {"a": "b"}, 8], ids=["list", "object", "number"])
@pytest.mark.parametrize("site", ["display_name", "note", "realization"])
def test_a_value_that_is_no_text_refuses_at_a_typed_text_site(site, value, tmp_path):
    """m-2: the display name, a realization's display name and the note take a text or nothing; a list, an object or
    a number refuses by name (revision 6 loaded each and the chip block printed its str())."""
    _refused(tmp_path, _at_site(site, value), rf"{SITES[site]} .* is no text")


def _symbol_letters(base, text):
    """``text`` upper-cased with each Latin letter written from the symbol block at ``base`` (its A)."""
    return "".join(chr(base + ord(c) - ord("A")) if "A" <= c <= "Z" else c for c in text.upper())


@pytest.mark.parametrize("base", [0x1F170, 0x1F150, 0x1F1E6],
                         ids=["negative_squared", "negative_circled", "regional_indicator"])
@pytest.mark.parametrize("site", sorted(SITES))
def test_a_symbol_letter_refuses_by_name_at_each_site(site, base, tmp_path):
    """m-3: symbols named as Latin letters, which no compatibility form maps, refuse by their code points (revision 6
    loaded them at every site)."""
    _refused(tmp_path, _at_site(site, _symbol_letters(base, "no physics profile")),
             rf"{SITES[site]} .*holds .*U\+{base + 13:X} ")
