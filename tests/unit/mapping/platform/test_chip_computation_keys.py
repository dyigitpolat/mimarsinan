"""[U-CHIPKEYS] Every chip description states each key that changes the computation from its source, with the source's
quote and location, or names it unstated: one loader rule (rulings D-3 and F-6/F-7 of the chip-description plan).

The oracle below is the paper's chip-description list (Definition 1, "changes the computation"), written here
independently of the toolchain's table, so the per-description test reads, on a tree without the rule, which keys
each description leaves to a registry default.

[U-CHIPKEYS2] The tests pin the rule: one hand-made refusal case per clause of the loader rule (``CLAUSES``, read
off ``computation_keys``, ``readings`` and ``target_bundles``), each red on the mutant that removes its clause; the
reading check is strict (one integer token per integer field in its span, or the one number its unit anchors), and the
lexicon of the other values states each value and excludes its opposite reading; the weight grid's composition.

[U-CHIPKEYS3] The tests pin every condition: one hand-made case per condition of each refusal expression and per element
of each closed set of the loader rule, the readings and the realization rule (the per-condition mutant harness, each
mutant red on its case), the seven named cases of the verify first (MC06, MC10, MC23, MC25, MC42, MC45, MC62); spans
at token boundaries (m-1); assumed_fields judged (m-2); the weight grid's slices named apart from the core field and
tied to it (m-3); the INT8 format clause's own case (m-5); the lexicon's L and E exclude the opposite reading (m-6);
the departing form (a computation the toolchain's key cannot hold, printed by the label).

[U-CHIPKEYS4] A derivation entry's field set is closed (item 3(d)); the pricing in the label is derived from the pricing
a run resolves, never stated (item 3(f)): the hand-made chip of the clause cases states its own profile, so the cases
of the rule read its statements only, and a chip that states no pricing has cases of its own."""

import copy
import json
import re
from pathlib import Path

import pytest

import mimarsinan
from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.config_schema.registry.decision_classes import FUNCTION_BEARING

TARGETS_DIR = Path(mimarsinan.__file__).parent / "mapping" / "platform" / "targets"
SHIPPED = sorted(path.stem for path in TARGETS_DIR.glob("*.json"))
#: The produced realizations the shipped descriptions declare (ruling D-5: ODIN's 1,024 x 256 card realization).
REALIZATIONS = ["odin@odin_wide_1024x256_mb16"]
ORACLE = {
    None: ("core_semantics", "weight_bits", "cores.has_bias"),
    "mvm": ("activation_bits", "readout_law", "onchip_arithmetic", "cores.operand"),
    "spiking": ("spiking_family", "spiking_variant", "simulation_steps", "target_tq", "firing_granularity",
                "firing_mode", "membrane_bits", "membrane_signed", "thresholding_mode"),
}
#: [U-CHIPKEYS2] The words that state each non-integer value in its ``reads`` span, and the words that state its
#: opposite reading (another value of the key), which the span must not hold: regular expressions over the span in
#: lower case. A bias is read only from a statement of a bias.
_NO_BIAS_WORD = r"^(?![\s\S]*\bbias\b)"
LEXICON = {
    ("core_semantics", "mvm"): ((r"\bmvm\b", r"matrix[- –]vector", r"vector[- –]matrix", r"dot[- ]product",
                                 r"multiply[- ]accumulate", r"multiply and accumulate", r"\bmacs?\b",
                                 r"in[- ]memory comput"),
                                (r"\bspik",)),
    ("core_semantics", "spiking"): ((r"\bspik", r"\bneurons?\b[\s\S]*\bevents?\b"),
                                    (r"\bmvm\b", r"matrix[- –]vector", r"multiply[- ]accumulate")),
    ("cores.has_bias", True): ((r"\bbias\b",),
                               (r"\bbias\b[^.;]*\b(rows?|axons?|slots?|tail)\b", r"\bno\b[^.;]*\bbias\b",
                                r"\bwithout\b[^.;]*\bbias\b")),
    ("cores.has_bias", False): ((r"\bbias\b[^.;]*\b(rows?|axons?|slots?|tail)\b", r"\bno\b[^.;]*\bbias\b",
                                 r"\bwithout\b[^.;]*\bbias\b", _NO_BIAS_WORD + r"[\s\S]*\binput\s+wire\b",
                                 _NO_BIAS_WORD + r"[\s\S]*\(\s*membrane\s*\+\s*weight\s*\)"),
                                (r"\bbias (current|registers?)\b", r"\bbias\b[^.;]*\b(programmed into|is added|adds)\b",
                                 r"\boffset\b", r"\bleak\b")),
    ("spiking_family", "lif"): ((r"integrate-and-fire", r"\blif\b"), (r"\bttfs\b", r"time[- ]to[- ]first[- ]spike")),
    ("spiking_variant", "streamed"): ((r"\bevent\b",), (r"time[- ]step", r"synchroni")),
    ("firing_granularity", "per_event"): ((r"\bevent\b",), (r"time[- ]step", r"\bper (time )?step\b")),
    ("firing_granularity", "per_cycle"): ((r"time[- ]step", r"after synaptic events are evaluated"),
                                          (r"\b(every|each) (synaptic )?event\b", r"\bper[- ]event\b")),
    ("firing_mode", "Novena"): ((r"8'd0", r"reset to (0|zero)"), (r"subtract", r"minus the threshold")),
    ("membrane_signed", False): ((r"\bunsigned\b", r"8'hff", r"8'b0"), (r"(?<!un)\bsigned\b", r"two's[- ]complement")),
    ("membrane_signed", True): ((r"(?<!un)\bsigned\b",), (r"\bunsigned\b",)),
    ("thresholding_mode", "<="): ((r">=", r"equals or exceeds", r"reached at equality"),
                                  (r"(?<!equals or )\bexceeds\b", r">(?!=)")),
    ("thresholding_mode", "<"): ((r"(?<!equals or )\bexceeds\b", r">(?!=)"),
                                 (r">=", r"equals or exceeds", r"\bequal")),
    ("readout_law.output_law", "identity"): ((r"\bexact\b", r"full[- ]precision", r"converts nothing"),
                                             (r"\bint\d+\b", r"requantiz")),
    ("readout_law.output_law", "requantize"): ((r"\bint\d+\b", r"requantiz"), (r"full[- ]precision", r"\bexact\b")),
    # [U-CHIPKEYS3] m-6: a class on chip is excluded by a span that keeps it on the host or states none
    ("onchip_arithmetic.classes", "L"): ((r"\brelu\b", r"\bactivation"),
                                         (r"\bhost\b", r"\boff[- ]chip\b", r"\b(no|without)\b[^.;]*\b(relu|activation)")),
    ("onchip_arithmetic.classes", "E"): ((r"\bresidual\b", r"\baddition", r"\badd\b"),
                                         (r"\bhost\b", r"\boff[- ]chip\b", r"\b(no|without)\b[^.;]*\b(residual|addition|add)\b")),
    ("onchip_arithmetic.classes", "P"): ((r"average pool",), (r"max[- ]?pool",)),
    ("readout_law.range", "calibrated"): ((r"calibrat",), (r"\bfixed range\b",)),
}
_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "twelve": 12, "sixteen": 16}


def _raw(name):
    base, _, real = name.partition("@")
    data = json.loads((TARGETS_DIR / f"{base}.json").read_text(encoding="utf-8"))
    block = dict(data.get("computation") or {})
    if real:
        block.update(((data.get("realizations") or {}).get(real) or {}).get("computation") or {})
    return data, block


def _sets(values, path):
    key, _, field = path.partition(".")
    if key == "cores" and field:
        return any(field in core for core in values.get("cores") or ())
    return key in values


def _values(name):
    from mimarsinan.mapping.platform.target_bundles import get_target_bundle
    return get_target_bundle(name).values


@pytest.mark.parametrize("name", SHIPPED + REALIZATIONS)
def test_every_chip_description_states_or_names_each_computation_key(name):
    """Red on a tree without the rule for each description that leaves a computation key to a registry default (or
    sets one with no statement from its source); the message names the keys."""
    _, block = _raw(name)
    values = _values(name)
    semantics = values.get("core_semantics", REGISTRY["core_semantics"].default)
    defaults, unsourced = [], []
    for path in ORACLE[None] + ORACLE[semantics]:
        if path not in block:
            (unsourced if _sets(values, path) else defaults).append(path)
    assert not defaults and not unsourced, (
        f"{name} ({semantics} cores): relies on a registry default for {defaults}; sets {unsourced} with no "
        "statement from its source")


@pytest.mark.parametrize("name", SHIPPED)
def test_a_description_missing_a_statement_refuses_by_name_at_load(name, tmp_path):
    """Each computation key's statement removed in turn: the loader refuses by name, naming the key."""
    from mimarsinan.mapping.platform.target_bundles import ChipKeyUnstatedError, load_bundle
    data, _ = _raw(name)
    semantics = _values(name)["core_semantics"]
    for path in ORACLE[None] + ORACLE[semantics]:
        changed = copy.deepcopy(data)
        changed.setdefault("computation", {}).pop(path, None)
        target = tmp_path / f"{name}.json"
        target.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ChipKeyUnstatedError, match=re.escape(path)):
            load_bundle(target)


# --- the test's own reading of a statement (independent of the loader's) ------------------------------------------

def _numbers(span):
    """(value, start, end) of every number of a span: digit runs at digit boundaries (a thousands-grouped one is one
    number, a decimal one has no integer value) and number words."""
    found = []
    for m in re.finditer(r"(?<![0-9.,])\d+(?:[.,]\d+)*(?![0-9])", span):
        text = m.group(0)
        found.append((int(text.replace(",", "")) if re.fullmatch(r"\d{1,3}(?:,\d{3})+|\d+", text) else None,
                      m.start(), m.end()))
    for m in re.finditer(r"\b(" + "|".join(_WORD_NUMBERS) + r")\b", span, flags=re.IGNORECASE):
        found.append((_WORD_NUMBERS[m.group(1).lower()], m.start(), m.end()))
    return found


def _anchored_by(span, start, end, unit):
    after = span[end:]
    if unit == "bits":
        return bool(re.match(r"\s?-?\s?(bits?|b)\b", after, flags=re.IGNORECASE)
                    or re.search(r"(?<![a-z])(u?int|fp|bf)$", span[:start], flags=re.IGNORECASE))
    if unit == "cells":
        return bool(not re.match(r"\s?-?\s?(bits?|b)\b", after, flags=re.IGNORECASE) and re.match(
            r"\s+(\d+\s?-?\s?bits?\s+|[a-z]+-bits?\s+)?([a-z+-]+\s+){0,2}?(cells?|devices?|crossbars?|pcms?)\b", after,
            flags=re.IGNORECASE))
    return False


def _one_number_read(span, number, unit):
    numbers = _numbers(span)
    return ({v for v, _, _ in numbers} == {number}
            or {v for v, s, e in numbers if v is not None and _anchored_by(span, s, e, unit)} == {number})


def _unit(where):
    name = where.rsplit(".", 1)[-1]
    return "cells" if name == "slices_per_weight" else "bits" if name.endswith("_bits") else None


def _readings(path, stated):
    """Why a stated value is not read in its spans, by the test's reading: integers by one number or their unit,
    the other values by the lexicon (each stating word present, no opposite word)."""
    value, reads = stated["value"], stated["reads"]
    assumed = set(stated.get("assumed_fields") or ())
    derived = {d.get("field"): d for d in stated.get("derivation") or ()}
    items = [(None, value)] if not isinstance(value, dict) else [(f, v) for f, v in value.items() if f not in assumed]
    errors = []
    for field, item in items:
        where = path if field is None else f"{path}.{field}"
        span = reads if isinstance(reads, str) else reads[field]
        assert span in stated["quote"], f"{where}: reads {span!r} is not a span of its quote"
        for one in item if isinstance(item, list) else [item]:
            if isinstance(one, int) and not isinstance(one, bool):
                read = derived[field]["reads"] if field in derived else one
                if not _one_number_read(span, read, _unit(where)):
                    errors.append(f"{where}={one}: {read} is not the one number (or the one {_unit(where)}-anchored "
                                  f"number) of {span!r}")
                continue
            entry = LEXICON.get((where, one))
            if entry is None:
                errors.append(f"{where}={one!r}: no reading of this value is known to the test")
                continue
            stating, opposite = entry
            low = span.lower()
            if not any(re.search(w, low) for w in stating):
                errors.append(f"{where}={one!r}: no word of {stating} in {span!r}")
            if hits := [w for w in opposite if re.search(w, low)]:
                errors.append(f"{where}={one!r}: {span!r} reads the opposite ({hits})")
    return errors


def _composition_readings(path, statement):
    comp = statement.get("composition") or {}
    errors = []
    for name in ("slices_per_weight", "cell_bits"):
        entry = comp.get(name, {})
        if "stated" in entry:
            errors += _readings(f"{path}.composition.{name}", entry["stated"])
        else:
            assert entry.get("unstated", {}).get("note", "").strip(), f"{path}.composition.{name}: why unstated"
    return errors


@pytest.mark.parametrize("name", SHIPPED + REALIZATIONS)
def test_each_stated_value_equals_its_quote_s(name):
    """Every stated value is read in its quote: each span is a span of the quote, each integer is the span's one
    number or the one its unit anchors (or the number its named derivation reads), each other value is stated by its
    words and never by its opposite's; the weight grid's composition readings likewise."""
    from mimarsinan.mapping.platform.target_bundles import get_target_bundle
    statements = get_target_bundle(name).statements
    assert statements, f"{name}: no statements"
    errors = []
    for path, statement in sorted(statements.items()):
        if path == "weight_bits":
            errors += _composition_readings(path, statement)
        if statement["kind"] == "departs":  # [U-CHIPKEYS3] the source's computation, read in its own span
            assert statement["reads"] in statement["quote"] and statement["computation"].strip(), f"{name} {path}"
            continue
        if statement["kind"] != "stated":
            assert statement["note"].strip(), f"{name} {path}: an unstated key names why"
            continue
        assert statement["source"].strip() and statement["location"].strip()
        errors += _readings(path, statement)
    assert not errors, f"{name}: " + "; ".join(errors)


@pytest.mark.parametrize("name", SHIPPED + REALIZATIONS)
def test_every_weight_grid_states_its_composition_beside_it(name):
    """[U-CHIPKEYS2] One definition of the weight grid for every description: the composed weight one synapse
    represents, with the cells per weight and the cell precision beside it, each stated or named unstated."""
    from mimarsinan.mapping.platform.target_bundles import get_target_bundle
    statement = get_target_bundle(name).statements["weight_bits"]
    comp = statement["composition"]
    assert set(comp) - {"law"} == {"slices_per_weight", "cell_bits"}
    if all("stated" in comp[n] for n in ("slices_per_weight", "cell_bits")):
        assert comp["law"] in ("signed_word", "magnitude_signed_apart")


#: [U-CHIPKEYS2] The weight grids the one definition moved, by description: (revision 1's value, revision 2's).
MOVED_GRIDS = {"prime_mat_256x256": (4, 9)}


@pytest.mark.parametrize("name", sorted(MOVED_GRIDS))
def test_a_grid_the_definition_moved_is_the_composed_weight(name):
    from mimarsinan.mapping.platform.target_bundles import get_target_bundle
    assert get_target_bundle(name).values["weight_bits"] == MOVED_GRIDS[name][1]


# --- one hand-made refusal case per clause of the loader rule ------------------------------------------------------

_NOTE = "a test chip"


def _unstated_grid(assumed=8):
    return {"unstated": {"note": _NOTE, "assumed": assumed, "composition": {
        "slices_per_weight": {"unstated": {"note": _NOTE}}, "cell_bits": {"unstated": {"note": _NOTE}}}}}


def _hand(**changes):
    block = {
        "core_semantics": {"stated": {"value": "mvm", "source": "hand", "location": "this test",
                                      "quote": "a matrix-vector multiply per core", "reads": "matrix-vector"}},
        "weight_bits": _unstated_grid(),
        "cores.has_bias": {"unstated": {"note": _NOTE, "assumed": False}},
        "activation_bits": {"unstated": {"note": "the deployment declares it"}},
        "readout_law": {"unstated": {"note": "the deployment declares it"}},
        "onchip_arithmetic": {"unstated": {"note": "the deployment declares it"}},
        "cores.operand": {"unstated": {"note": "no operand core type"}},
    }
    data = {"name": "hand", "geometry": "recom_64x64_256",  # [U-CHIPKEYS4] the chip states its pricing (a profile)
            "keys": {"core_semantics": "mvm", "platform_physics_profile": "generic_estimated_22nm"},
            "faithfulness": {"class": "exact"}, "computation": block}
    for path, value in changes.items():
        if path == "keys":
            data["keys"].update(value)
        elif path == "data":
            data.update(value)
        elif value is None:
            block.pop(path.replace("__", "."), None)
        else:
            block[path.replace("__", ".")] = value
    return data


def _load(data, tmp_path):
    from mimarsinan.mapping.platform.target_bundles import load_bundle
    path = tmp_path / f"{data['name']}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_bundle(path)


def _st(value, quote, reads=None, **extra):
    return {"stated": {"value": value, "source": "s", "location": "l", "quote": quote,
                       "reads": quote if reads is None else reads, **extra}}


def _grid(cells, bits, law=None, assumed=8):
    comp = {"slices_per_weight": cells, "cell_bits": bits, **({"law": law} if law else {})}
    return {"unstated": {"note": _NOTE, "assumed": assumed, "composition": comp}}


_ADC = {"adc_bits": 13, "mult_bits": 24, "output_law": "identity", "range": "calibrated"}
_LAW = {"value": _ADC, "source": "hand", "location": "this test", "quote": "two 12-bit unsigned results",
        "reads": "two 12-bit unsigned results", "assumed_fields": {"mult_bits": "t", "output_law": "t", "range": "t"}}
#: [U-CHIPKEYS3] A departing statement of the hand-made chip's weight grid (its fields, each a case of its own below).
_DEPARTING = {"computation": "binary synapse times a per-neuron integer of the type of its axon", "source": "hand",
              "location": "this test", "quote": "Synaptic weight is a product of two terms: a bit times an integer.",
              "reads": "a product of two terms", "assumed": 8, "reason": "the grid holds one integer per synapse",
              "note": "the test's", "composition": {"slices_per_weight": {"unstated": {"note": "n"}},
                                                    "cell_bits": {"unstated": {"note": "n"}}}}
_TWO_TYPES = {"name": "hand_two_types", "cores": [
    {"max_axons": 8, "max_neurons": 8, "count": 1, "has_bias": True},
    {"max_axons": 8, "max_neurons": 8, "count": 1, "has_bias": False}],
    "weight_bits": 8, "provenance": "a test chip", "claim_class": "quarantined"}

#: clause id -> (what the clause refuses, the changes to the hand-made chip, the refusal's message). Read off the rule's
#: code: computation_keys.statement_errors and _judge_one, readings.judged_reading, read_spans, value_reading_errors
#: and composition_errors, and key_value.
CLAUSES = {
    "K02": ("a computation block that is no object", {"data": {"computation": ["core_semantics"]}},
            r"computation must be an object"),
    "K05": ("a statement for a key the description does not have", {"spiking_variant": {"unstated": {"note": "n"}}},
            r"spiking_variant: not a computation key"),
    "K06": ("a statement of neither form", {"readout_law": {"guessed": {"note": "n"}}}, r"readout_law: a statement is"),
    "K07": ("a statement with a field of neither form",
            {"core_semantics": _st("mvm", "matrix-vector", extra=1)}, r"core_semantics: stated takes the fields"),
    "K08": ("a stated statement with no source",
            {"keys": {"activation_bits": 8}, "activation_bits": {"stated": {
                "value": 8, "source": " ", "location": "l", "quote": "8-bit inputs", "reads": "8-bit"}}},
            r"activation_bits: stated needs a non-empty source"),
    "K09": ("a stated statement with no location",
            {"keys": {"activation_bits": 8}, "activation_bits": {"stated": {
                "value": 8, "source": "s", "location": "", "quote": "8-bit inputs", "reads": "8-bit"}}},
            r"activation_bits: stated needs a non-empty location"),
    "K10": ("a stated statement with no quote",
            {"keys": {"activation_bits": 8}, "activation_bits": {"stated": {
                "value": 8, "source": "s", "location": "l", "quote": None, "reads": "8-bit"}}},
            r"activation_bits: stated needs a non-empty quote"),
    "K11": ("a stated statement with no reads",
            {"keys": {"activation_bits": 8}, "activation_bits": {"stated": {
                "value": 8, "source": "s", "location": "l", "quote": "8-bit inputs", "reads": ""}}},
            r"activation_bits: stated needs a non-empty reads"),
    "K12": ("an unstated statement with no note", {"readout_law": {"unstated": {"note": " "}}},
            r"readout_law: unstated needs a non-empty note"),
    "K13": ("a stated statement with no value",
            {"keys": {"activation_bits": 8}, "activation_bits": {"stated": {
                "source": "s", "location": "l", "quote": "8-bit inputs", "reads": "8-bit"}}},
            r"activation_bits: stated needs its value"),
    "K14": ("a stated value the description does not set", {"activation_bits": _st(8, "8-bit inputs")},
            r"activation_bits: stated 8, but the description sets nothing"),
    "K15": ("a stated value the description sets otherwise", {"keys": {"activation_bits": 4},
                                                            "activation_bits": _st(8, "8-bit inputs")},
            r"activation_bits: stated 8, but the description sets 4"),
    "K16": ("an assumed value the description does not set",
            {"activation_bits": {"unstated": {"note": "n", "assumed": 8}}},
            r"activation_bits: assumed 8, but the description sets nothing"),
    "K17": ("an assumed value the description sets otherwise", {"weight_bits": _unstated_grid(4)},
            r"weight_bits: assumed 4, but the description sets 8"),
    "K18": ("a key named unstated with no assumption that the description sets", {"keys": {"activation_bits": 8}},
            r"activation_bits: named unstated without its assumed value"),
    "K19": ("a weight grid with no composition beside it", {"weight_bits": {"unstated": {"note": "n", "assumed": 8}}},
            r"weight_bits: states its composition beside it"),
    "K20": ("a reads span that is not in its quote", {"keys": {"activation_bits": 8},
                                                     "activation_bits": _st(8, "8-bit inputs", "8 bit")},
            r"activation_bits: reads '8 bit' is not a span of its quote"),
    "K21": ("one span per field that names another field",
            {"keys": {"readout_law": dict(_ADC)}, "readout_law": {"stated": {
                **_LAW, "reads": {"adc_bits": "12-bit", "range": "results"},
                "derivation": [{"field": "adc_bits", "rule": "sign_and_magnitude", "reads": 12}]}}},
            r"readout_law: reads names the fields \['adc_bits', 'range'\]"),
    "K22": ("a reads that is neither a span nor one span per field",
            {"keys": {"activation_bits": 8}, "activation_bits": _st(8, "8-bit inputs", {"value": "8-bit"})},
            r"activation_bits: reads is a span, or one span per field"),
    "K23": ("an integer the span does not hold", {"keys": {"activation_bits": 8},
                                                 "activation_bits": _st(8, "4-bit inputs")},
            r"activation_bits: the stated 8 is not read"),
    "K24": ("an integer read inside a longer number", {"keys": {"activation_bits": 8},
                                                      "activation_bits": _st(8, "128b outputs")},
            r"activation_bits: the stated 8 is not read"),
    "K25": ("an integer among two unit-anchored numbers (one token per integer field)",
            {"keys": {"activation_bits": 6}, "activation_bits": _st(6, "under 4-bit input and 6-bit output")},
            r"activation_bits: the stated 6 is not read"),
    "K26": ("a decimal's last digits read as an integer",
            {"keys": {"activation_bits": 5}, "activation_bits": _st(5, "a 2.5-bit effective precision")},
            r"activation_bits: the stated 5 is not read"),
    "K27": ("a derivation that names no rule",
            {"keys": {"readout_law": dict(_ADC)}, "readout_law": {"stated": {**_LAW, "derivation": [
                {"field": "adc_bits", "rule": "guessed", "reads": 12}]}}},
            r"readout_law.adc_bits: derivation .* names no rule"),
    "K28": ("a derivation whose result is not the value",
            {"keys": {"readout_law": dict(_ADC)}, "readout_law": {"stated": {**_LAW, "derivation": [
                {"field": "adc_bits", "rule": "sign_and_magnitude", "reads": 11}]}}},
            r"readout_law.adc_bits: sign_and_magnitude of 11 is not 13"),
    "K29": ("a derivation whose input the span does not hold",
            {"keys": {"readout_law": dict(_ADC)}, "readout_law": {"stated": {
                **_LAW, "quote": "two 10-bit unsigned results", "reads": "two 10-bit unsigned results",
                "derivation": [{"field": "adc_bits", "rule": "sign_and_magnitude", "reads": 12}]}}},
            r"readout_law.adc_bits: sign_and_magnitude's 12 is not read"),
    "K30": ("a closed value whose span spells another option and not itself",
            {"core_semantics": _st("mvm", "a group of spiking neurons per core")},
            r"core_semantics: 'a group of spiking neurons per core' spells \['spiking'\]"),
    "K31": ("a composition that is no object of its two readings",
            {"weight_bits": {"unstated": {"note": "n", "assumed": 8, "composition": {
                "slices_per_weight": {"unstated": {"note": "n"}}}}}},
            r"weight_bits.composition: composition takes \['slices_per_weight', 'cell_bits'\]"),
    "K32": ("a composition reading of neither form",
            {"weight_bits": _grid({"guessed": {}}, {"unstated": {"note": "n"}})},
            r"weight_bits.composition.slices_per_weight: a statement is"),
    "K33": ("a composition reading that is no positive integer",
            {"weight_bits": _grid(_st(0, "0 cells"), {"unstated": {"note": "n"}})},
            r"weight_bits.composition.slices_per_weight: a count of cells or bits is a positive integer"),
    "K34": ("a composition reading the span does not hold",
            {"weight_bits": _grid({"unstated": {"note": "n"}}, _st(4, "2-bit cells"))},
            r"weight_bits.composition.cell_bits: the stated 4 is not read"),
    "K35": ("a width read as a count of cells",
            {"weight_bits": _grid(_st(2, "We use 2 bits per device"), {"unstated": {"note": "n"}})},
            r"weight_bits.composition.slices_per_weight: the stated 2 is not read"),
    "K36": ("a law beside a reading that is not stated",
            {"weight_bits": _grid({"unstated": {"note": "n"}}, _st(8, "8-bit cells"), "signed_word")},
            r"weight_bits.composition: a law composes stated readings only"),
    "K37": ("two stated readings and no law",
            {"weight_bits": _grid(_st(1, "one 8-bit weight per cell"), _st(8, "8-bit cells"))},
            r"weight_bits.composition: both readings stated, so a law"),
    "K38": ("a composed grid the description does not set (one cell's precision recorded as the grid)",
            {"weight_bits": _grid(_st(2, "two 4-bit cells"), _st(4, "two 4-bit cells"), "magnitude_signed_apart")},
            r"weight_bits.composition: magnitude_signed_apart of .* is a 9-bit grid, but the description sets 8"),
    "K39": ("one value for core types that differ",
            {"data": {"geometry": _TWO_TYPES}, "cores__has_bias": {"unstated": {"note": "n", "assumed": True}}},
            r"cores.has_bias: assumed True, but the description sets \[True, False\]"),
    # --- [U-CHIPKEYS3] the verify's seven surviving conditions (MC23, MC25, MC42, MC45, MC10; MC06 and MC62 below)
    "K40": ("a composition with a field of no reading (MC23)",
            {"weight_bits": {"unstated": {"note": "n", "assumed": 8, "composition": {
                "slices_per_weight": {"unstated": {"note": "n"}}, "cell_bits": {"unstated": {"note": "n"}},
                "cells_per_wieght": {"unstated": {"note": "n"}}}}}},
            r"weight_bits.composition: composition takes \['slices_per_weight', 'cell_bits'\]"),
    "K41": ("a boolean count of slices (MC25)",
            {"weight_bits": _grid(_st(True, "two 4-bit cells"), {"unstated": {"note": "n"}})},
            r"weight_bits.composition.slices_per_weight: a count of cells or bits is a positive integer, got True"),
    "K42": ("an object field's value whose span spells another option of the field (MC42)",
            {"keys": {"readout_law": dict(_ADC)}, "readout_law": {"stated": {
                **_LAW, "quote": "two 12-bit unsigned results requantize at the core output",
                "reads": {"adc_bits": "two 12-bit unsigned results", "output_law": "requantize at the core output"},
                "assumed_fields": {"mult_bits": "t", "range": "t"},
                "derivation": [{"field": "adc_bits", "rule": "sign_and_magnitude", "reads": 12}]}}},
            r"readout_law.output_law: 'requantize at the core output' spells \['requantize'\]"),
    "K42b": ("an object field's value whose span spells another option of the field (the range)",
             {"keys": {"readout_law": dict(_ADC)}, "readout_law": {"stated": {
                 **_LAW, "quote": "two 12-bit unsigned results over a fixed range",
                 "reads": {"adc_bits": "two 12-bit unsigned results", "range": "a fixed range"},
                 "assumed_fields": {"mult_bits": "t", "output_law": "t"},
                 "derivation": [{"field": "adc_bits", "rule": "sign_and_magnitude", "reads": 12}]}}},
             r"readout_law.range: 'a fixed range' spells \['fixed'\]"),
    "K43": ("a derivation whose read is no integer (MC45)",
            {"keys": {"readout_law": dict(_ADC)}, "readout_law": {"stated": {**_LAW, "derivation": [
                {"field": "adc_bits", "rule": "sign_and_magnitude", "reads": "12"}]}}},
            r"readout_law.adc_bits: derivation .* names no rule .* with an integer read"),
    "K44": ("a width field read where its one number is a count (MC10)",
            {"weight_bits": _grid({"unstated": {"note": "n"}}, _st(4, "4 PCM devices"))},
            r"weight_bits.composition.cell_bits: the stated 4 is not read"),
    # --- [U-CHIPKEYS3] m-1: a span starts and ends at its quote's token boundaries
    "K45": ("a span cut inside a number", {"keys": {"activation_bits": 8},
                                          "activation_bits": _st(8, "In total 128b outputs", "8b outputs")},
            r"activation_bits: reads '8b outputs' cuts a token of its quote"),
    "K46": ("a span cut inside a word", {"core_semantics": _st("mvm", "a matrix-vector multiply per core", "atrix-vector")},
            r"core_semantics: reads 'atrix-vector' cuts a token of its quote"),
    "K47": ("a span cut inside an operator (>=)", {"keys": {"activation_bits": 8},
                                                  "activation_bits": _st(8, "width >= 8-bit inputs", "= 8-bit inputs")},
            r"activation_bits: reads '= 8-bit inputs' cuts a token"),
    "K47b": ("a span cut inside an operator (<=)", {"keys": {"activation_bits": 8},
                                                   "activation_bits": _st(8, "width <= 8-bit inputs", "= 8-bit inputs")},
             r"activation_bits: reads '= 8-bit inputs' cuts a token"),
    "K47c": ("a span cut inside an operator (!=)", {"keys": {"activation_bits": 8},
                                                   "activation_bits": _st(8, "width != 8-bit inputs", "= 8-bit inputs")},
             r"activation_bits: reads '= 8-bit inputs' cuts a token"),
    "K48": ("a span cut after a number's decimal mark", {"keys": {"activation_bits": 5},
                                                        "activation_bits": _st(5, "a 2.5-bit effective precision",
                                                                               "5-bit effective")},
            r"activation_bits: reads '5-bit effective' cuts a token"),
    "K49": ("a span cut before a number's group mark", {"keys": {"activation_bits": 2},
                                                       "activation_bits": _st(2, "a 2,048-bit bus", "a 2")},
            r"activation_bits: reads 'a 2' cuts a token"),
    # --- [U-CHIPKEYS3] m-2: assumed_fields is judged
    "K50": ("assumed_fields beside a value that is no object",
            {"keys": {"activation_bits": 8}, "activation_bits": _st(8, "8-bit inputs", assumed_fields={"x": "t"})},
            r"activation_bits: assumed_fields names fields of an object value"),
    "K51": ("assumed_fields that is no object of reasons",
            {"keys": {"readout_law": dict(_ADC)}, "readout_law": {"stated": {
                **_LAW, "assumed_fields": ["mult_bits", "output_law", "range"],
                "derivation": [{"field": "adc_bits", "rule": "sign_and_magnitude", "reads": 12}]}}},
            r"readout_law: assumed_fields is a non-empty object"),
    "K51b": ("assumed_fields that names no field",
             {"keys": {"readout_law": dict(_ADC)}, "readout_law": {"stated": {
                 **_LAW, "assumed_fields": {},
                 "derivation": [{"field": "adc_bits", "rule": "sign_and_magnitude", "reads": 12}]}}},
             r"readout_law: assumed_fields is a non-empty object"),
    "K52": ("an assumed field the value does not have",
            {"keys": {"readout_law": dict(_ADC)}, "readout_law": {"stated": {
                **_LAW, "assumed_fields": {"mult_bits": "t", "output_law": "t", "range": "t", "gain": "t"},
                "derivation": [{"field": "adc_bits", "rule": "sign_and_magnitude", "reads": 12}]}}},
            r"readout_law.gain: assumed_fields names no field of the value"),
    "K53": ("an assumed field with no reason",
            {"keys": {"readout_law": dict(_ADC)}, "readout_law": {"stated": {
                **_LAW, "assumed_fields": {"mult_bits": " ", "output_law": "t", "range": "t"},
                "derivation": [{"field": "adc_bits", "rule": "sign_and_magnitude", "reads": 12}]}}},
            r"readout_law.mult_bits: an assumed field states why"),
    "K53b": ("an assumed field whose reason is no text",
             {"keys": {"readout_law": dict(_ADC)}, "readout_law": {"stated": {
                 **_LAW, "assumed_fields": {"mult_bits": 1, "output_law": "t", "range": "t"},
                 "derivation": [{"field": "adc_bits", "rule": "sign_and_magnitude", "reads": 12}]}}},
             r"readout_law.mult_bits: an assumed field states why"),
    "K54": ("every field of the value assumed",
            {"keys": {"readout_law": dict(_ADC)}, "readout_law": {"stated": {
                **_LAW, "assumed_fields": {"adc_bits": "t", "mult_bits": "t", "output_law": "t", "range": "t"}}}},
            r"readout_law: assumed_fields names every field of the value"),
    # --- [U-CHIPKEYS3] m-3: the core field holds whole copies of the stated slices
    "K55": ("a core count of cells that holds no whole copies of the slices",
            {"data": {"core_fields": {"cells_per_weight": 4, "weight_cell_rows": 1}},
             "weight_bits": _grid(_st(3, "three 4-bit cells"), {"unstated": {"note": "n"}})},
            r"weight_bits.composition: the cores' cells_per_weight \[4\] holds no whole copies of the 3 "
            r"slices_per_weight"),
    # --- [U-CHIPKEYS3] the departing form
    "K56": ("a departing statement with no value assumed in the source's place",
            {"data": {"faithfulness": {"class": "approximate"}},
             "weight_bits": {"departs": {k: v for k, v in _DEPARTING.items() if k != "assumed"}}},
            r"weight_bits: departs needs the value assumed in the source's place"),
    "K57": ("a departing statement whose reads is no span of its quote",
            {"data": {"faithfulness": {"class": "approximate"}},
             "weight_bits": {"departs": {**_DEPARTING, "reads": "a product of three terms"}}},
            r"weight_bits: reads 'a product of three terms' is not a span of its quote"),
    "K58": ("a departing computation holding the label's separator ';'",
            {"data": {"faithfulness": {"class": "approximate"}},
             "weight_bits": {"departs": {**_DEPARTING, "computation": "a binary synapse; its type's integer"}}},
            r"weight_bits: a departing computation is printed in the label, so it holds no"),
    "K58b": ("a departing computation holding the label's separator ':'",
             {"data": {"faithfulness": {"class": "approximate"}},
              "weight_bits": {"departs": {**_DEPARTING, "computation": "a binary synapse: its type's integer"}}},
             r"weight_bits: a departing computation is printed in the label, so it holds no"),
    "K59": ("a departing statement assuming another value than the description sets",
            {"data": {"faithfulness": {"class": "approximate"}},
             "weight_bits": {"departs": {**_DEPARTING, "assumed": 4}}},
            r"weight_bits: assumed 4, but the description sets 8"),
    "K60": ("a departing statement with a field of no kind",
            {"data": {"faithfulness": {"class": "approximate"}},
             "weight_bits": {"departs": {**_DEPARTING, "value": 8}}},
            r"weight_bits: departs takes the fields"),
    "K61": ("a departing statement on an exact faithfulness (the label would print exact beside a departure)",
            {"weight_bits": {"departs": dict(_DEPARTING)}},
            r"faithfulness 'exact' \[.*weight_bits departs from its source\\'s computation .*\]: one of"),
    "K62": ("a departing statement on a key with its composition beside it, missing it",
            {"data": {"faithfulness": {"class": "approximate"}},
             "weight_bits": {"departs": {k: v for k, v in _DEPARTING.items() if k != "composition"}}},
            r"weight_bits: states its composition beside it"),
}


@pytest.mark.parametrize("clause", sorted(CLAUSES))
def test_the_loader_refuses_each_clause_of_its_rule(clause, tmp_path):
    from mimarsinan.mapping.platform.target_bundles import ChipKeyUnstatedError, TargetBundleError
    _, changes, match = CLAUSES[clause]
    with pytest.raises(TargetBundleError, match=match) as raised:
        _load(_hand(**changes), tmp_path)
    assert not isinstance(raised.value, ChipKeyUnstatedError), "a defect of a statement is no unstated key"


def test_k01_a_description_that_states_no_core_semantics_refuses(tmp_path):
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError
    data = _hand()
    data["keys"] = {}
    with pytest.raises(TargetBundleError, match=r"core_semantics: a chip description states its core semantics"):
        _load(data, tmp_path)


def test_k03_a_description_with_no_computation_block_refuses_naming_every_key(tmp_path):
    """The state the rule exists to refuse (R5's): no block at all; the refusal names each key, as unstated."""
    from mimarsinan.mapping.platform.target_bundles import ChipKeyUnstatedError
    data = _hand()
    del data["computation"]
    with pytest.raises(ChipKeyUnstatedError) as raised:
        _load(data, tmp_path)
    for path in ORACLE[None] + ORACLE["mvm"]:
        assert path in str(raised.value), path


def test_k04_a_missing_statement_refuses_as_an_unstated_key(tmp_path):
    from mimarsinan.mapping.platform.target_bundles import ChipKeyUnstatedError
    with pytest.raises(ChipKeyUnstatedError, match=r"neither stated nor named unstated .*readout_law"):
        _load(_hand(readout_law=None), tmp_path)


# --- the readings the rule accepts (each the green side of a clause) ----------------------------------------------

@pytest.mark.parametrize("quote, value", [
    ("In total 16 32b outputs (SUM[31:0]) are computed", 32),  # one unit-anchored number among four
    ("256 8-bit digital inputs", 8),
    ("quantized to four-bit weights", 4),  # a number word
    ("Bit-width (W, Q, K, V) (8, 8, 8, 8)", 8),  # one number, repeated
    ("before conversion into INT8", 8),  # a format before its number
    # [U-CHIPKEYS3] m-5: the format clause alone reads these (another number in the span, the format anchoring one)
    ("INT8 operands on 64 lanes", 8),
    ("FP16 operands on 64 lanes", 16),
    ("BF16 operands on 64 lanes", 16),
    ("UINT8 operands on 64 lanes", 8),
    # each spelling of the width's unit after its number (bits, bit, b)
    ("8 bits per input on 64 lanes", 8),
    ("an 8 bit input on 64 lanes", 8),
    ("8b inputs on 64 lanes", 8),
])
def test_the_loader_reads_an_integer_by_one_number_or_its_unit(quote, value, tmp_path):
    bundle = _load(_hand(keys={"activation_bits": value}, activation_bits=_st(value, quote)), tmp_path)
    assert bundle.statements["activation_bits"]["value"] == value


def test_the_loader_reads_each_field_in_its_own_span(tmp_path):
    reads = {"adc_bits": "two 12-bit unsigned results", "mult_bits": "an FP16 multiplier"}
    law = {**_LAW, "quote": "two 12-bit unsigned results ... an FP16 multiplier", "reads": reads,
           "assumed_fields": {"output_law": "t", "range": "t"},
           "derivation": [{"field": "adc_bits", "rule": "sign_and_magnitude", "reads": 12},
                          {"field": "mult_bits", "rule": "ieee_significand", "reads": 16}]}
    value = {**_ADC, "mult_bits": 11}
    bundle = _load(_hand(keys={"readout_law": value}, readout_law={"stated": {**law, "value": value}}), tmp_path)
    assert bundle.statements["readout_law"]["reads"] == reads


def test_a_derived_integer_reads_its_rule_s_input(tmp_path):
    """A 13-bit signed grid read as two 12-bit unsigned magnitudes (the sign-and-magnitude rule) loads."""
    data = _hand(keys={"readout_law": dict(_ADC)}, readout_law={"stated": {**_LAW, "derivation": [
        {"field": "adc_bits", "rule": "sign_and_magnitude", "reads": 12}]}})
    assert _load(data, tmp_path).statements["readout_law"]["value"]["adc_bits"] == 13


def test_a_composed_grid_loads_by_its_law(tmp_path):
    """Two 4-bit cells holding an 8-bit magnitude, its sign held apart: the 9-bit grid (law magnitude_signed_apart);
    the count is read by the cells it counts, the width by its unit, in one span."""
    bundle = _load(_hand(data={"geometry": {**_TWO_TYPES, "name": "hand_nine", "weight_bits": 9, "cores": [
        _TWO_TYPES["cores"][1]]}}, weight_bits={"unstated": {"note": "n", "assumed": 9, "composition": {
            "slices_per_weight": _st(2, "two 4-bit cells"), "cell_bits": _st(4, "two 4-bit cells"),
            "law": "magnitude_signed_apart"}}}), tmp_path)
    assert bundle.statements["weight_bits"]["composition"]["law"] == "magnitude_signed_apart"


def test_an_object_is_compared_on_its_function_bearing_fields(tmp_path):
    """A cost-only field of an object key (the lanes per core) is not part of the stated computation."""
    value = {"classes": ["L"], "lanes_per_core": 4}
    bundle = _load(_hand(keys={"onchip_arithmetic": value}, onchip_arithmetic=_st(
        {"classes": ["L"]}, "a ReLU unit per core")), tmp_path)
    assert bundle.statements["onchip_arithmetic"]["value"] == {"classes": ["L"]}


def test_the_hand_description_loads_with_its_statements(tmp_path):
    bundle = _load(_hand(), tmp_path)
    assert bundle.statements["core_semantics"]["kind"] == "stated"
    assert bundle.statements["readout_law"] == {"kind": "unstated", "note": "the deployment declares it"}


def test_the_rule_s_key_list_is_the_paper_s_chip_description_list():
    from mimarsinan.mapping.platform.descriptions.computation_keys import CHIP_COMPUTATION_KEYS
    table = {sem: tuple(p for p, (applies, _) in CHIP_COMPUTATION_KEYS.items() if applies == sem) for sem in ORACLE}
    assert table == ORACLE


# --- the verify's data mutants, each caught by the reading (v-m1) -------------------------------------------------

#: A misreading of a shipped statement, value and statement changed together: (description, key path, the field of an
#: object value or None, the misread value). The verify's VD01 to VD04.
MISREADINGS = {
    "VD01": ("truenorth", "thresholding_mode", None, "<"),
    "VD02": ("hermes_64", "cores.has_bias", None, False),
    "VD03": ("neurram", "readout_law", "adc_bits", 4),
    "VD04": ("sram_cim_macro", "readout_law", "adc_bits", 16),
}


@pytest.mark.parametrize("mutant", sorted(MISREADINGS))
def test_a_consistent_misreading_of_a_source_is_caught(mutant):
    name, path, field, wrong = MISREADINGS[mutant]
    from mimarsinan.mapping.platform.target_bundles import get_target_bundle
    statement = copy.deepcopy(get_target_bundle(name).statements[path])
    assert statement["kind"] == "stated"
    if field is None:
        statement["value"] = wrong
    else:
        statement["value"][field] = wrong
    from mimarsinan.mapping.platform.descriptions.computation_keys import DERIVATIONS, _options
    from mimarsinan.mapping.platform.descriptions.readings import value_reading_errors
    loader = value_reading_errors(path, statement, lambda f: _options(path, f), DERIVATIONS)
    assert loader or _readings(path, statement), f"{mutant}: {path}={wrong!r} passes every reading of its quote"


# --- the class sweep ------------------------------------------------------------------------------------------------

def _chip_group_keys():
    """The function-bearing keys of the chip groups a description may set (declarable, in the two sections)."""
    from mimarsinan.config_schema.registry import Category
    return sorted(key for key, entry in REGISTRY.items()
                  if entry.group in ("hardware", "spiking") and entry.decision_class == FUNCTION_BEARING
                  and entry.section in ("deployment_parameters", "platform_constraints")
                  and entry.category is not Category.RUNTIME
                  and not (entry.category is Category.DERIVED and not entry.declarable))


@pytest.mark.parametrize("key", _chip_group_keys())
def test_every_function_bearing_key_of_the_chip_groups_is_a_chip_key_or_a_deployment_choice(key):
    """The class sweep, one test per registry key: a key that changes the computation and declares the chip or its
    neuron dynamics is a chip key every description states or names, covered by one, or the deployment's choice."""
    from mimarsinan.mapping.platform.descriptions.computation_keys import (
        CHIP_COMPUTATION_KEYS, COVERED_BY, DEPLOYMENT_CHOICES)
    assert (key in CHIP_COMPUTATION_KEYS) + (key in COVERED_BY) + (key in DEPLOYMENT_CHOICES) == 1, key
    assert (CHIP_COMPUTATION_KEYS.get(key, (None, "x"))[1] and COVERED_BY.get(key, "x")
            and DEPLOYMENT_CHOICES.get(key, "x")), key


@pytest.mark.parametrize("field, sample", [("has_bias", False), ("operand", {"payload_bits": 8}),
                                           ("weight_bits", 4)])
def test_every_function_bearing_core_field_is_required_where_a_description_sets_it(field, sample):
    from mimarsinan.mapping.platform.descriptions.computation_keys import required_paths
    values = {"core_semantics": "mvm", "cores": [{"max_axons": 8, "max_neurons": 8, field: sample}]}
    assert f"cores.{field}" in required_paths(values)


def test_a_function_bearing_key_a_description_sets_beyond_the_list_is_required():
    from mimarsinan.mapping.platform.descriptions.computation_keys import required_paths
    values = {"core_semantics": "spiking", "cores": [{"max_axons": 8, "max_neurons": 8}], "bias_row_splitting": "auto"}
    assert "bias_row_splitting" in required_paths(values)
    assert "activation_bits" not in required_paths(values)


# --- the faithfulness label states what an exact class rests on (v-m4) ---------------------------------------------

def test_a_faithfulness_states_the_configuration_its_class_rests_on(tmp_path):
    data = _hand(data={"faithfulness": {"class": "exact", "given": ["leakage off by the deployment"]}})
    faith = _load(data, tmp_path).faithfulness
    assert faith.label() == 'exact, given "leakage off by the deployment"'  # [U-CHIPKEYS6] a quoted field
    assert faith.with_declared(["cores"]).label() == 'approximate, given "leakage off by the deployment": cores declared'


@pytest.mark.parametrize("given", ["leakage off", ["leakage: off"], [""], ["a; b"]])
def test_a_faithfulness_given_is_a_list_of_named_settings(given, tmp_path):
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError
    with pytest.raises(TargetBundleError, match=r"faithfulness given"):
        _load(_hand(data={"faithfulness": {"class": "exact", "given": given}}), tmp_path)


def test_odin_s_exact_class_states_leakage_and_learning_off_by_the_deployment():
    from mimarsinan.mapping.platform.target_bundles import get_target_bundle
    assert get_target_bundle("odin").faithfulness.label() == ('exact, given "leakage and SDSP learning off by the '
                                                              'deployment"')


# --- [U-CHIPKEYS3] the readings each condition of the rule accepts and refuses (hand-made, direct) -----------------

def _integer_error(span, number, unit="bits"):
    from mimarsinan.mapping.platform.descriptions.readings import integer_reading_error
    return integer_reading_error(span, number, unit)


#: The number words a quote may spell, written here independently of the reading's table (each a case of its own).
SPELLED_NUMBERS = [("one", 1), ("two", 2), ("three", 3), ("four", 4), ("five", 5), ("six", 6), ("seven", 7),
                   ("eight", 8), ("nine", 9), ("ten", 10), ("twelve", 12), ("sixteen", 16), ("thirty-two", 32)]


@pytest.mark.parametrize("word, number", SPELLED_NUMBERS)
def test_each_number_word_is_read_as_its_number(word, number):
    assert _integer_error(f"{word}-bit inputs", number) is None
    assert _integer_error(f"{word}-bit inputs", number + 100) is not None


def test_a_thousands_grouped_numeral_is_one_number_mc06():
    """MC06: "1,024" is the one number 1024, never its first group 1 (nor its last, 24)."""
    assert _integer_error("a 1,024-bit bus", 1024) is None
    assert _integer_error("a 1,024-bit bus", 1) is not None
    assert _integer_error("a 1,024-bit bus", 24) is not None


@pytest.mark.parametrize("span, wrong", [
    ("a 2.5-bit effective precision", 2),  # a decimal's first digits are no integer
    ("a 2.5-bit effective precision", 5),  # nor its last
    ("a .5-bit effective step", 5),  # a leading decimal mark
    ("a ,5-bit grouping", 5),  # a leading group mark
])
def test_a_decimal_or_a_cut_number_is_never_read_as_an_integer(span, wrong):
    assert _integer_error(span, wrong) is not None


@pytest.mark.parametrize("span, number", [
    ("two 4-bit cells", 2),
    ("4 PCM devices and 8-bit inputs", 4),
    ("4 RRAM devices and 8-bit inputs", 4),  # devices alone anchor the count
    ("8 crossbars at 2 bits per device", 8),
    ("4 PCMs per weight and 8-bit inputs", 4),
    ("across 8 2-bit memristor cells", 8),  # a numeral width between the count and its cells
    ("two four-bit memristor RRAM cells", 2),  # a spelled width, then two words
    ("two G+ RRAM cells beside 8-bit inputs", 2),  # a word holding '+'
    ("two G- RRAM cells beside 8-bit inputs", 2),  # a word holding '-'
])
def test_a_count_is_read_by_the_cells_it_counts(span, number):
    assert _integer_error(span, number, "cells") is None


@pytest.mark.parametrize("span, number", [
    ("4 bits and 8 cells", 4),  # a width is never a count
    ("4 b and 8 cells", 4),
    ("4 b cells", 4),  # a width's spaced unit before the cells: a width, never a count
])
def test_a_width_is_never_read_as_a_count(span, number):
    assert _integer_error(span, number, "cells") is not None


def test_a_count_is_never_read_as_a_width_mc10():
    """MC10: a width field's one number is never the number a count anchors."""
    assert _integer_error("4 PCM devices", 4, "bits") is not None
    assert _integer_error("4 PCM devices", 4, "cells") is None


def test_a_field_of_no_unit_reads_one_number():
    assert _integer_error("256 rows", 256, None) is None
    assert _integer_error("256 rows of 8-bit cells", 8, None) is not None


@pytest.mark.parametrize("span, value, options", [
    ("updated per event", "per_cycle", ("per_event", "per_cycle")),  # an option spelled with a space
    ("updated per-event", "per_cycle", ("per_event", "per_cycle")),  # with a hyphen
    ("updated per_event", "per_cycle", ("per_event", "per_cycle")),  # with an underscore
])
def test_a_span_spelling_another_option_refuses_however_the_option_is_spelled(span, value, options):
    from mimarsinan.mapping.platform.descriptions.readings import option_reading_error
    assert option_reading_error(span, value, options) is not None
    assert option_reading_error(span, value.replace("cycle", "event"), options) is None


@pytest.mark.parametrize("quote, span, ok", [
    ("In total 128b outputs", "128b outputs", True),
    ("In total 128b outputs", "8b outputs", False),
    ("a 2.5-bit precision", "2.5-bit", True),
    ("a 2.5-bit precision", "5-bit", False),
    ("a 2.5-bit precision", "a 2", False),
    ("a 1,024-bit bus", "1,024-bit bus", True),
    ("a 1,024-bit bus", "024-bit bus", False),
    ("a 1,024-bit bus", "a 1", False),
    ("x >= y", ">= y", True),
    ("x >= y", "= y", False),
    ("x >= y", "x >", False),
    ("8 bits, 8 cells", "8 bits", True),
    ("8 bits. 8 cells", "8 bits", True),
    ("the cell holds 8b and 8b", "8b", True),  # a later occurrence at its boundaries is the span
    ("a8b and 8b", "8b", True),
])
def test_a_span_is_read_at_its_quote_s_token_boundaries(quote, span, ok):
    from mimarsinan.mapping.platform.descriptions.readings import span_at_token_boundaries
    assert span_at_token_boundaries(span, quote) is ok


# --- [U-CHIPKEYS3] each field of each statement form accepted (each element of the closed field sets) ---------------

def test_a_stated_statement_takes_each_of_its_fields(tmp_path):
    """value, source, location, quote, reads, derivation, assumed_fields and note together."""
    data = _hand(keys={"readout_law": dict(_ADC)}, readout_law={"stated": {**_LAW, "note": "the test's", "derivation": [
        {"field": "adc_bits", "rule": "sign_and_magnitude", "reads": 12}]}})
    assert _load(data, tmp_path).statements["readout_law"]["note"] == "the test's"


def test_a_composition_reading_takes_each_of_its_fields(tmp_path):
    """A stated reading with a derivation and a note: two 5-bit slices (4-bit magnitudes with their sign), a signed word
    of 10 bits."""
    geometry = {**_TWO_TYPES, "name": "hand_ten", "weight_bits": 10, "cores": [_TWO_TYPES["cores"][1]]}
    reading = _st(5, "two 4-bit cells", "two 4-bit cells", note="the test's",
                  derivation=[{"field": None, "rule": "sign_and_magnitude", "reads": 4}])
    bundle = _load(_hand(data={"geometry": geometry}, weight_bits={"unstated": {"note": "n", "assumed": 10, "composition": {
        "slices_per_weight": _st(2, "two 4-bit cells", note="n"), "cell_bits": reading, "law": "signed_word"}}}),
        tmp_path)
    assert bundle.statements["weight_bits"]["composition"]["cell_bits"]["stated"]["note"] == "the test's"


def test_an_unstated_statement_takes_its_note_and_assumption(tmp_path):
    assert _load(_hand(), tmp_path).statements["cores.has_bias"] == {"kind": "unstated", "note": _NOTE,
                                                                       "assumed": False}


def test_a_derivation_by_the_ieee_significand_reads_each_format(tmp_path):
    """Each format of the significand rule (half, single, double precision) gives its significand."""
    from mimarsinan.mapping.platform.descriptions.computation_keys import DERIVATIONS
    rule = DERIVATIONS["ieee_significand"][1]
    assert [rule(16), rule(32), rule(64)] == [11, 24, 53]
    assert DERIVATIONS["sign_and_magnitude"][1](8) == 9


@pytest.mark.parametrize("law, composed", [("signed_word", 8), ("magnitude_signed_apart", 9)])
def test_each_composition_law_composes_its_grid(law, composed, tmp_path):
    geometry = {**_TWO_TYPES, "name": "hand_law", "weight_bits": composed, "cores": [_TWO_TYPES["cores"][1]]}
    bundle = _load(_hand(data={"geometry": geometry}, weight_bits={"unstated": {
        "note": "n", "assumed": composed, "composition": {"slices_per_weight": _st(4, "4 2-bit memristor cells"),
                                                          "cell_bits": _st(2, "4 2-bit memristor cells"), "law": law}}}),
        tmp_path)
    assert bundle.values["weight_bits"] == composed


def test_a_core_count_holding_whole_copies_of_the_slices_loads(tmp_path):
    """m-3: four cells per weight hold two copies (a polarity twin) of two slices."""
    geometry = {**_TWO_TYPES, "name": "hand_pair", "weight_bits": 9, "cores": [_TWO_TYPES["cores"][1]]}
    bundle = _load(_hand(data={"geometry": geometry, "core_fields": {"cells_per_weight": 4, "weight_cell_rows": 1}},
                         weight_bits={"unstated": {"note": "n", "assumed": 9, "composition": {
                             "slices_per_weight": _st(2, "two 4-bit cells"), "cell_bits": _st(4, "two 4-bit cells"),
                             "law": "magnitude_signed_apart"}}}), tmp_path)
    assert bundle.values["cores"][0]["cells_per_weight"] == 4


# --- [U-CHIPKEYS3] the departing form: the source's computation, the value assumed in its place, the label ----------

def test_a_departing_statement_loads_and_its_label_prints_the_departure(tmp_path):
    bundle = _load(_hand(data={"faithfulness": {"class": "approximate"}}, weight_bits={"departs": dict(_DEPARTING)}),
                   tmp_path)
    assert bundle.statements["weight_bits"]["kind"] == "departs"
    assert bundle.faithfulness.label() == ('approximate: weight_bits departs from its source\'s computation "binary '
                                           'synapse times a per-neuron integer of the type of its axon", assumed "8"')


def test_a_departure_is_named_after_the_description_s_own(tmp_path):
    bundle = _load(_hand(data={"faithfulness": {"class": "refused", "named": ["a departure of the description"]}},
                         weight_bits={"departs": dict(_DEPARTING)}), tmp_path)
    assert bundle.faithfulness.named[0] == 'named "a departure of the description"' and len(bundle.faithfulness.named) == 2


def test_a_departure_is_named_after_the_pricing_of_a_description_that_states_none(tmp_path):
    """[U-CHIPKEYS4] 3(f): the pricing the run resolves is named first, then the description's own, then its
    statements' departures."""
    data = _hand(data={"faithfulness": {"class": "refused", "named": ["a departure of the description"]}},
                 weight_bits={"departs": dict(_DEPARTING)})
    del data["keys"]["platform_physics_profile"]
    named = _load(data, tmp_path).faithfulness.named
    assert named[:2] == ("no physics profile", 'named "a departure of the description"') and len(named) == 3


@pytest.mark.parametrize("field", ["computation", "source", "location", "quote", "reads", "reason"])
def test_a_departing_statement_needs_each_text(field, tmp_path):
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError
    with pytest.raises(TargetBundleError, match=rf"weight_bits: departs needs a non-empty {field}"):
        _load(_hand(data={"faithfulness": {"class": "approximate"}},
                    weight_bits={"departs": {**_DEPARTING, field: " "}}), tmp_path)


def test_the_departing_form_takes_its_note(tmp_path):
    bundle = _load(_hand(data={"faithfulness": {"class": "approximate"}}, weight_bits={"departs": dict(_DEPARTING)}),
                   tmp_path)
    assert bundle.statements["weight_bits"]["note"] == "the test's"


@pytest.mark.parametrize("name", ["truenorth", "dynap_se_64x256x4"])
def test_a_chip_whose_synapse_the_grid_cannot_hold_states_its_departure(name):
    """The report's candidate m-4: TrueNorth's and DYNAP-SE's weight grids depart from their sources' computation,
    each with its quote and the grid assumed in its place, and the label prints the departure."""
    from mimarsinan.mapping.platform.target_bundles import get_target_bundle
    bundle = get_target_bundle(name)
    statement = bundle.statements["weight_bits"]
    assert statement["kind"] == "departs" and statement["assumed"] == bundle.values["weight_bits"]
    assert (f"weight_bits departs from its source's computation {json.dumps(statement['computation'])}, assumed "
            f"{json.dumps(json.dumps(statement['assumed']))}") in bundle.faithfulness.label()
    assert not bundle.faithfulness.label().startswith("exact")


# --- [U-CHIPKEYS3] the faithfulness the loader judges, each kind and field --------------------------------------------

@pytest.mark.parametrize("faith, label", [
    ({"class": "exact"}, "exact"),
    ({"class": "approximate", "named": ["a"]}, 'approximate: named "a"'),  # [U-CHIPKEYS6] typed text quoted
    ({"class": "refused", "named": ["b"]}, 'refused: named "b"'),
    ({"class": "exact", "given": ["g"], "note": "a note of the test"}, 'exact, given "g"'),
])
def test_each_faithfulness_kind_and_field_loads(faith, label, tmp_path):
    assert _load(_hand(data={"faithfulness": faith}), tmp_path).faithfulness.label() == label


@pytest.mark.parametrize("faith", [{"class": "exact", "named": ["a"]}, {"class": "approximate"},
                                   {"class": "refused"}, {"class": "guessed", "named": ["a"]}])
def test_a_faithfulness_kind_against_what_it_names_refuses(faith, tmp_path):
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError
    with pytest.raises(TargetBundleError, match=r"faithfulness .*: one of"):
        _load(_hand(data={"faithfulness": faith}), tmp_path)


# --- [U-CHIPKEYS4] 3(f): the pricing in the label is derived from the pricing the run resolves, never stated ---------

def _unpriced(faith, extra=None):
    """The hand-made chip stating no pricing (neither a profile nor its absence)."""
    data = _hand(data={"faithfulness": faith, **(extra or {})})
    del data["keys"]["platform_physics_profile"]
    return data


@pytest.mark.parametrize("faith, label", [
    ({"class": "exact"}, "approximate: no physics profile"),  # the pricing is a departure: exact becomes approximate
    ({"class": "approximate"}, "approximate: no physics profile"),  # the pricing names what departs
    ({"class": "refused"}, "refused: no physics profile"),
    ({"class": "refused", "named": ["b"]}, 'refused: no physics profile; named "b"'),
    ({"class": "exact", "given": ["g"]}, 'approximate, given "g": no physics profile'),
])
def test_a_description_that_states_no_pricing_names_its_unresolved_pricing(faith, label, tmp_path):
    bundle = _load(_unpriced(faith), tmp_path)
    assert bundle.faithfulness.label() == label and bundle.faithfulness.pricing == ("no physics profile",)


@pytest.mark.parametrize("faith", [{"class": "exact", "named": ["a"]}, {"class": "guessed"}])
def test_a_faithfulness_of_a_description_that_states_no_pricing_is_judged_by_what_it_names(faith, tmp_path):
    """An exact class names nothing of its own, whatever its pricing; a kind of no form refuses."""
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError
    with pytest.raises(TargetBundleError, match=r"faithfulness .*: one of"):
        _load(_unpriced(faith), tmp_path)


@pytest.mark.parametrize("data", [{"faithfulness": {"class": "refused", "named": ["no physics profile"]}},
                                  {"faithfulness": {"class": "approximate", "named": ["a", "no physics profile"]}}])
def test_a_description_never_states_the_pricing_its_label_derives(data, tmp_path):
    """The pricing departure is the label's, derived from the run's pricing: a description stating it refuses by name,
    priced or not. [U-CHIPKEYS5] Refused as a text of the pricing clause's form, read back by the label grammar
    (test_chip_label_clauses.py holds every form and every site)."""
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError
    for chip in (_hand(data=data), _unpriced(data["faithfulness"])):
        with pytest.raises(TargetBundleError, match=r"faithfulness named 'no physics profile' holds the label's derived "
                                                    r"clause \(no_pricing\): the label derives it from the run"):
            _load(chip, tmp_path)


def test_a_description_stating_its_pricing_absent_names_no_pricing(tmp_path):
    data = _unpriced({"class": "exact"}, {"pricing_absent": "a hand-made chip no profile prices"})
    assert _load(data, tmp_path).faithfulness.label() == "exact"


# --- [U-CHIPKEYS3] the conditions of the statement rule, each its own case (the harness's first run) -----------------

def test_the_swept_keys_are_the_chip_groups_function_bearing_keys():
    """swept_keys reads the registry by the chip groups (hardware, spiking), function-bearing and settable: the test's
    own sweep, written independently."""
    from mimarsinan.mapping.platform.descriptions.computation_keys import swept_keys
    assert swept_keys() == tuple(_chip_group_keys())
    assert "core_semantics" in swept_keys() and "spiking_family" in swept_keys()


@pytest.mark.parametrize("key, reason", [
    ("no_such_key", "no_such_key is not a registry key"),
    ("device", "device is not declarable"),  # a runtime key
    ("spiking_mode", "spiking_mode is not declarable"),  # a derived key no declaration sets
    ("target", "target cannot be set by a bundle"),  # a composite key
    ("experiment_name", "experiment_name cannot be set by a bundle"),  # a top-level key
])
def test_a_key_no_description_may_set_refuses_by_name(key, reason, tmp_path):
    from mimarsinan.mapping.platform.descriptions.computation_keys import unsettable_reason
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError
    assert unsettable_reason(key).startswith(reason)
    with pytest.raises(TargetBundleError, match=re.escape(reason)):
        _load(_hand(keys={key: "x"}), tmp_path)


@pytest.mark.parametrize("key", ["weight_quantization", "core_semantics", "activation_bits", "allow_coalescing"])
def test_a_declarable_key_of_either_section_may_be_set(key):
    """A derived key a declaration sets, a deployment parameter and a platform constraint: no reason refuses them."""
    from mimarsinan.mapping.platform.descriptions.computation_keys import unsettable_reason
    assert unsettable_reason(key) is None


def test_key_value_reads_a_core_field_over_the_core_types():
    from mimarsinan.mapping.platform.descriptions.computation_keys import MISSING, key_value
    one = {"max_axons": 8, "max_neurons": 8}
    assert key_value({"cores": [dict(one), dict(one)]}, "cores.has_bias") is MISSING
    assert key_value({"cores": [{**one, "has_bias": True}, dict(one)]}, "cores.has_bias") == [True, None]
    assert key_value({"cores": [{**one, "has_bias": True}, {**one, "has_bias": True}]}, "cores.has_bias") is True
    assert key_value({}, "cores.has_bias") is MISSING
    assert key_value({"cores": [{**one, "has_bias": False}]}, "cores") == [{**one, "has_bias": False}]
    assert key_value({"weight_bits": 4}, "activation_bits") is MISSING and key_value({"weight_bits": 4}, "weight_bits") == 4


def test_key_value_reads_an_object_key_s_function_bearing_fields_only():
    from mimarsinan.mapping.platform.descriptions.computation_keys import key_value
    assert key_value({"onchip_arithmetic": {"classes": ["L"], "lanes_per_core": 4}}, "onchip_arithmetic") == {
        "classes": ["L"]}
    assert key_value({"activation_bits": {"x": 1}}, "activation_bits") == {"x": 1}  # a key of no object schema
    assert key_value({"no_such_key": {"x": 1}}, "no_such_key") == {"x": 1}


def test_required_paths_reads_the_semantics_keys_and_every_function_bearing_key_or_core_field_set():
    from mimarsinan.mapping.platform.descriptions.computation_keys import required_paths
    mvm = required_paths({"core_semantics": "mvm"})
    assert mvm == ("core_semantics", "weight_bits", "cores.has_bias", "activation_bits", "readout_law",
                   "onchip_arithmetic", "cores.operand")
    spiking = required_paths({"core_semantics": "spiking", "allow_coalescing": True, "bias_resolution": "weight_grid",
                              "cores": [{"max_axons": 8, "max_neurons": 8, "weight_bits": 4, "count": 1}]})
    assert "allow_coalescing" not in spiking and "cores.max_axons" not in spiking  # cost-only: not required
    assert spiking.count("bias_resolution") == 1 and spiking.count("cores.weight_bits") == 1
    assert "cores" not in spiking and "activation_bits" not in spiking
    assert required_paths({"core_semantics": "mvm", "weight_bits": 8}).count("weight_bits") == 1
    assert required_paths({"core_semantics": "mvm", "cores": [{"has_bias": True}]}).count("cores.has_bias") == 1


@pytest.mark.parametrize("path, field, options", [
    ("core_semantics", None, ("spiking", "mvm")),  # a key's options
    ("activation_bits", None, ()),  # a key of no options
    ("readout_law", "output_law", ("requantize", "identity")),  # an object field's options
    ("readout_law", "adc_bits", ()),  # an object field of no options
    ("onchip_arithmetic", "classes", ("L", "E", "P")),  # a list field's item options
    ("readout_law", "no_such_field", ()),
    ("cores.has_bias", None, ()),  # a core field: no options read
    ("no_such_key", None, ()),
    ("activation_bits", "bits", ()),  # a field of a key with no object schema
])
def test_the_options_a_closed_value_is_read_against(path, field, options):
    from mimarsinan.mapping.platform.descriptions.computation_keys import _options
    assert _options(path, field) == options


def test_function_bearing_differences_name_only_the_function_bearing_paths():
    from mimarsinan.mapping.platform.descriptions.computation_keys import function_bearing_differences
    from mimarsinan.mapping.platform.target_bundles import Faithfulness, TargetBundle
    core = {"max_axons": 8, "max_neurons": 8, "count": 1, "has_bias": False}

    def bundle(values):
        return TargetBundle("x", "g", "c", Faithfulness("exact"), json.dumps(values))
    first = bundle({"core_semantics": "mvm", "weight_bits": 8, "allow_coalescing": True, "cores": [dict(core)]})
    second = bundle({"core_semantics": "mvm", "weight_bits": 4, "allow_coalescing": False,
                     "cores": [{**core, "has_bias": True, "max_axons": 16}]})
    assert function_bearing_differences(first, second) == ("cores.has_bias", "weight_bits")
    assert function_bearing_differences(first, first) == ()


def test_a_path_s_value_is_read_through_lists_and_objects():
    from mimarsinan.mapping.platform.descriptions.computation_keys import _path_value
    assert _path_value({"cores": [{"a": 1}, "x"]}, "cores.a") == (1, None)
    assert _path_value({"o": {"a": 1}}, "o.a") == 1
    assert _path_value({"o": 5}, "o.a") is None
    assert _path_value({}, "o") is None


def test_the_refusal_names_a_required_key_with_what_it_is_or_bare(tmp_path):
    """A chip key unstated is named with what it is; a function-bearing key the description sets beyond the list is
    named bare."""
    from mimarsinan.mapping.platform.target_bundles import ChipKeyUnstatedError
    with pytest.raises(ChipKeyUnstatedError, match=r"activation_bits \(the activation grid at the host-chip "
                                                   r"boundaries\); bias_resolution"):
        _load(_hand(keys={"bias_resolution": "weight_grid"}, activation_bits=None), tmp_path)


def test_a_departing_statement_without_its_assumption_is_refused_as_departing_only(tmp_path):
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError
    with pytest.raises(TargetBundleError) as raised:
        _load(_hand(data={"faithfulness": {"class": "approximate"}},
                    weight_bits={"departs": {k: v for k, v in _DEPARTING.items() if k != "assumed"}}), tmp_path)
    assert "departs needs the value assumed" in str(raised.value) and "named unstated" not in str(raised.value)


def test_the_statement_rule_reads_values_with_no_cores_or_no_grid():
    """statement_errors on values a test builds (no cores, no weight grid): the composition is judged against no grid,
    named as such, and nothing breaks."""
    from mimarsinan.mapping.platform.descriptions.computation_keys import statement_errors
    block = _hand()["computation"]
    block["weight_bits"] = {"stated": {"value": 8, "source": "s", "location": "l", "quote": "8-bit weights",
                                       "reads": "8-bit weights", "composition": {
                                           "slices_per_weight": _st(1, "one 8-bit weight per cell"),
                                           "cell_bits": _st(8, "8-bit cells"), "law": "signed_word"}}}
    unstated, errors, _ = statement_errors(block, {"core_semantics": "mvm"})
    assert "weight_bits: stated 8, but the description sets nothing" in errors
    assert any("signed_word of" in e and "the description sets None" in e for e in errors), errors


def test_every_composed_key_has_its_tie():
    from mimarsinan.mapping.platform.descriptions.readings import COMPOSITION_READINGS, COMPOSITION_TIES
    assert set(COMPOSITION_TIES) == set(COMPOSITION_READINGS)
    assert all(COMPOSITION_TIES[k][0] in COMPOSITION_READINGS[k] for k in COMPOSITION_TIES)


def test_a_skeleton_needs_the_description_s_core_semantics():
    from mimarsinan.mapping.platform.descriptions.computation_keys import skeleton
    with pytest.raises(ValueError, match="core semantics"):
        skeleton({"weight_bits": 8}, "n")
    block = skeleton({"core_semantics": "mvm", "weight_bits": 8}, "n")
    assert block["weight_bits"] == {"unstated": {"note": "n", "assumed": 8, "composition": {
        "slices_per_weight": {"unstated": {"note": "n"}}, "cell_bits": {"unstated": {"note": "n"}}}}}
    assert block["activation_bits"] == {"unstated": {"note": "n"}}


# --- [U-CHIPKEYS3] the readings' conditions, each its own case ------------------------------------------------------

def test_a_decimal_anchored_beside_the_read_number_is_no_number():
    assert _integer_error("a 2.5-bit average of 8-bit weights", 8) is None


def test_the_spelled_option_rule_judges_only_what_it_can_spell():
    from mimarsinan.mapping.platform.descriptions.readings import option_reading_error
    options = ("per_event", "per_cycle")
    assert option_reading_error("updated per event", "<=", ("per_event", "<=")) is None  # an unspellable value
    assert option_reading_error("per event, not per cycle", "per_event", options) is None  # its own spelled too
    assert option_reading_error("updated per event", "per_cycle", options + ("<",)) is not None  # an unspellable option
    assert option_reading_error("updated per event", "per_cycle", (3,) + options) is not None  # a non-text option
    assert option_reading_error("updated per event", "per_cycle", ("per_cycle", "per_event")) is not None


@pytest.mark.parametrize("quote, span", [
    ("8 bits of 8", "8 bits"),  # a span at the quote's start
    ("x 1 8-bit", "8-bit"),  # a digit two back, no mark between
    ("8.Then 4-bit", "Then 4-bit"),  # a mark, then no digit
    (".5 is 8", "5 is 8"),  # a mark at the quote's start
    ("a,5-bit", "5-bit"),  # a mark after a letter
    ("8 9", "8"),  # no mark between two numbers
    ("x.5 bits", "x"),  # a mark after a letter, a digit after it
    ("8,", "8"),  # a mark at the quote's end
    ("8, 9", "8"),  # a mark, then a space
])
def test_a_span_beside_a_mark_or_a_number_that_is_no_cut_is_read(quote, span):
    from mimarsinan.mapping.platform.descriptions.readings import span_at_token_boundaries
    assert span_at_token_boundaries(span, quote)


def test_a_stated_boolean_is_never_read_as_an_integer():
    from mimarsinan.mapping.platform.descriptions.readings import value_reading_errors
    stated = {"value": True, "quote": "a bias per 8 rows", "reads": "a bias per 8 rows"}
    assert value_reading_errors("cores.has_bias", stated, lambda f: (), {}) == []


def test_a_stated_number_that_is_no_integer_is_not_read_as_one():
    from mimarsinan.mapping.platform.descriptions.readings import value_reading_errors
    stated = {"value": 2.5, "quote": "a 2.5 V range", "reads": "a 2.5 V range"}
    assert value_reading_errors("readout_law.range_value", stated, lambda f: (), {}) == []


def test_a_composition_reading_with_a_defect_is_not_read_further():
    from mimarsinan.mapping.platform.descriptions.readings import composition_errors
    composition = {"slices_per_weight": {"stated": {"value": 0, "source": " ", "location": "l", "quote": "0 cells",
                                                    "reads": "0 cells"}},
                   "cell_bits": {"unstated": {"note": "n"}}}
    assert composition_errors("w", composition, 8, ("slices_per_weight", "cell_bits"), {}) == [
        "w.slices_per_weight: stated needs a non-empty source"]


_DERIVED_LAW = {**_LAW, "derivation": [{"field": "adc_bits", "rule": "sign_and_magnitude", "reads": 12}]}
CLAUSES_3B = {
    "K63": ("a statement that is a list", {"readout_law": ["unstated"]}, r"readout_law: a statement is one of"),
    "K64": ("a statement of two forms", {"readout_law": {"unstated": {"note": "n"}, "stated": {}}},
            r"readout_law: a statement is one of"),
    "K65": ("a form whose body is no object", {"readout_law": {"unstated": []}},
            r"readout_law: unstated takes the fields"),
    "K66": ("a per-field span that is blank", {"keys": {"readout_law": dict(_ADC)}, "readout_law": {"stated": {
        **_DERIVED_LAW, "reads": {"adc_bits": " "}}}}, r"readout_law: reads ' ' is not a span of its quote"),
    "K67": ("a per-field span that is no text", {"keys": {"readout_law": dict(_ADC)}, "readout_law": {"stated": {
        **_DERIVED_LAW, "reads": {"adc_bits": 12}}}}, r"readout_law: reads 12 is not a span of its quote"),
    "K68": ("a derivation that is no list", {"keys": {"readout_law": dict(_ADC)}, "readout_law": {"stated": {
        **_LAW, "derivation": {"field": "adc_bits", "rule": "sign_and_magnitude", "reads": 12}}}},
            r"readout_law: derivation is a list of \{field, rule, reads\}"),
    "K69": ("a derivation entry that is no object", {"keys": {"readout_law": dict(_ADC)}, "readout_law": {"stated": {
        **_LAW, "derivation": ["adc_bits"]}}}, r"readout_law: derivation is a list"),
    "K70": ("a derivation of a field the value does not read (an assumed one)",
            {"keys": {"readout_law": dict(_ADC)}, "readout_law": {"stated": {**_LAW, "derivation": [
                {"field": "adc_bits", "rule": "sign_and_magnitude", "reads": 12},
                {"field": "mult_bits", "rule": "ieee_significand", "reads": 32}]}}},
            r"readout_law: derivation is a list .* each a field the value reads \(\['adc_bits'\]\)"),
    "K71": ("two derivations of one field", {"keys": {"readout_law": dict(_ADC)}, "readout_law": {"stated": {
        **_LAW, "derivation": [{"field": "adc_bits", "rule": "sign_and_magnitude", "reads": 12}] * 2}}},
            r"readout_law: derivation is a list .* one per field it derives"),
    "K72": ("a derivation naming no field", {"keys": {"readout_law": dict(_ADC)}, "readout_law": {"stated": {
        **_LAW, "derivation": [{"rule": "sign_and_magnitude", "reads": 12}]}}}, r"readout_law: derivation is a list"),
    "K73": ("a law that is no name", {"weight_bits": {"unstated": {"note": "n", "assumed": 8, "composition": {
        "slices_per_weight": _st(1, "one 8-bit weight per cell"), "cell_bits": _st(8, "8-bit cells"),
        "law": ["signed_word"]}}}}, r"weight_bits.composition: both readings stated, so a law"),
}


@pytest.mark.parametrize("clause", sorted(CLAUSES_3B))
def test_the_loader_refuses_each_further_clause_of_its_rule(clause, tmp_path):
    from mimarsinan.mapping.platform.target_bundles import ChipKeyUnstatedError, TargetBundleError
    _, changes, match = CLAUSES_3B[clause]
    with pytest.raises(TargetBundleError, match=match) as raised:
        _load(_hand(**changes), tmp_path)
    assert not isinstance(raised.value, ChipKeyUnstatedError)


# --- [U-CHIPKEYS3] the bundle loader's own clauses (target_bundles), each its own case -------------------------------

def _hand_file(tmp_path, text, name="hand"):
    from mimarsinan.mapping.platform.target_bundles import load_bundle
    path = tmp_path / f"{name}.json"
    path.write_text(text, encoding="utf-8")
    return load_bundle(path)


@pytest.mark.parametrize("change, match", [
    ({"data": {"geometry": "no_such_geometry"}}, r"no_such_geometry"),
    ({"data": {"geometry": {**_TWO_TYPES, "name": "recom_64x64_256"}}}, r"declares a second geometry under a registered"),
    ({"data": {"geometry": {**_TWO_TYPES, "cores": 5}}}, r"geometry: "),  # a TypeError of the geometry
    ({"data": {"geometry": {**_TWO_TYPES, "cores": [{"max_neurons": 8, "count": 1}]}}}, r"geometry: "),  # a KeyError
    ({"data": {"geometry": {**_TWO_TYPES, "weight_bits": 0}}}, r"geometry: "),  # a ValueError
    ({"data": {"geometry": {k: v for k, v in _TWO_TYPES.items() if k != "provenance"}}}, r"geometry has unknown"),
    ({"data": {"geometry": ["recom_64x64_256"]}}, r"geometry must be an object"),
    ({"keys": {"platform_physics_overrides": {}}}, r"declared through its named override blocks"),
    ({"data": {"faithfulness": {"class": "exact", "given": [5]}}}, r"faithfulness given"),
    ({"data": {"core_fields": []}}, r"core_fields must be an object"),
    ({"data": {"core_fields": {"max_axons": 4}}}, r"core_fields \['max_axons'\] are set by the geometry"),
    ({"data": {"core_fields": {"no_field": 1}}}, r"core_fields \['no_field'\] are no core-type field"),
    ({"data": {"core_fields": {"cells_per_weight": 0, "weight_cell_rows": 1}}}, r"core_fields.cells_per_weight"),
    ({"data": {"core_fields": {"cells_per_weight": 2, "weight_cell_rows": 0}}}, r"core_fields.weight_cell_rows"),
    ({"data": {"core_fields": {"cells_per_weight": 3, "weight_cell_rows": 1}}}, r"cells_per_weight"),
    ({"data": {"keys": ["core_semantics"]}}, r"keys must be an object"),
    ({"keys": {"cores": []}}, r"cores is set by the geometry"),
    ({"data": {"geometry": {**_TWO_TYPES, "weight_bits": 40}}}, r"hand.json: weight_bits=40 is above the maximum 32"),
    ({"data": {"geometry": {**_TWO_TYPES, "cores": [{**_TWO_TYPES["cores"][0], "has_bias": "yes"}]}}},
     r"hand.json: cores\[0\].has_bias must be bool, got str"),
    ({"data": {"faithfulness": {}}}, r"faithfulness has unknown fields \[\] or lacks \['class'\]"),
    ({"data": {"faithfulness": {"class": "exact", "given": "leakage"}}}, r"faithfulness given 'leakage'"),
])
def test_the_bundle_loader_refuses_each_clause_by_name(change, match, tmp_path):
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError
    with pytest.raises(TargetBundleError, match=match):
        _load(_hand(**change), tmp_path)


@pytest.mark.parametrize("field", ["name", "geometry", "keys", "faithfulness"])
def test_a_bundle_without_a_required_field_refuses_by_name(field, tmp_path):
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError
    data = {k: v for k, v in _hand().items() if k != field}
    with pytest.raises(TargetBundleError, match=rf"the bundle has unknown fields \[\] or lacks \['{field}'\]"):
        _hand_file(tmp_path, json.dumps(data))


def test_a_bundle_named_with_the_realization_separator_refuses(tmp_path):
    """A bundle's name is letters, digits and underscores, as a realization's: "<bundle>@<realization>" addresses a
    realization, never a bundle."""
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError
    with pytest.raises(TargetBundleError, match=r"name 'x@y' .*letters, digits and underscores"):
        _hand_file(tmp_path, json.dumps({**_hand(), "name": "x@y"}), name="x@y")


def test_a_bundle_file_that_is_no_json_or_no_object_or_misnamed_refuses(tmp_path):
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError
    with pytest.raises(TargetBundleError, match=r"not JSON"):
        _hand_file(tmp_path, "{")
    with pytest.raises(TargetBundleError, match=r"the bundle must be an object"):
        _hand_file(tmp_path, "[]")
    with pytest.raises(TargetBundleError, match=r"name 'hand' differs from the file name 'other'"):
        _hand_file(tmp_path, json.dumps(_hand()), name="other")


def test_a_geometry_bundled_twice_refuses_by_name(tmp_path):
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError, load_bundles
    for name in ("first", "second"):
        (tmp_path / f"{name}.json").write_text(json.dumps({**_hand(), "name": name}), encoding="utf-8")
    with pytest.raises(TargetBundleError, match=r"recom_64x64_256.*already bundled by 'first'"):
        load_bundles(tmp_path)


def test_the_faithfulness_label_and_its_departures():
    from mimarsinan.mapping.platform.target_bundles import Faithfulness
    refused = Faithfulness("refused", ("no physics profile",))
    assert refused.with_departures(["x declared"]).label() == "refused: no physics profile; x declared"
    assert refused.with_departures([]) is refused
    assert Faithfulness("exact", (), ("g",)).with_declared(["b", "a"]).label() == 'approximate, given "g": a declared; b declared'
    assert Faithfulness("approximate", ("d",)).label() == "approximate: d"


def test_technology_blocks_leave_out_a_profile_whose_descriptions_store_one_weight_per_cell(tmp_path):
    from mimarsinan.mapping.platform.target_bundles import load_bundles, technology_blocks
    (tmp_path / "hand.json").write_text(json.dumps(_hand(keys={"platform_physics_profile": "generic_estimated_22nm"})),
                                        encoding="utf-8")
    known = load_bundles(tmp_path)
    assert technology_blocks(known) == {}
    assert technology_blocks(known) == {}  # the cached read


def test_the_decision_class_of_a_target_is_the_join_of_its_keys():
    from mimarsinan.config_schema.registry.decision_classes import FUNCTION_BEARING
    from mimarsinan.mapping.platform.target_bundles import target_decision_class
    assert target_decision_class("neurram") == FUNCTION_BEARING


@pytest.mark.parametrize("module", ["computation_keys", "departures", "readings", "realizations"])
def test_each_module_of_the_rule_imports_as_a_process_s_first_import(module):
    """A reader that imports one of the rule's modules first (a script, a probe) gets it: the package loads the loader
    first, so the cycle through the registry closes in one order."""
    import subprocess
    import sys
    name = f"mimarsinan.mapping.platform.descriptions.{module}"
    done = subprocess.run([sys.executable, "-c", f"import {name}"], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-600:]


# --- [U-CHIPKEYS3] the harness's first run, further conditions ------------------------------------------------------

CLAUSES_3D = {
    "K74": ("an object's reads that is a list of its fields (the loader reads it as no reads)",
            {"keys": {"readout_law": dict(_ADC)}, "readout_law": {"stated": {**_DERIVED_LAW, "reads": ["adc_bits"]}}},
            r"readout_law: stated needs a non-empty reads"),
    "K75": ("a text field that is an object", {"keys": {"activation_bits": 8}, "activation_bits": {"stated": {
        "value": 8, "source": {"name": "s"}, "location": "l", "quote": "8-bit inputs", "reads": "8-bit"}}},
            r"activation_bits: stated needs a non-empty source"),
    "K76": ("an empty per-field reads", {"keys": {"readout_law": dict(_ADC)}, "readout_law": {"stated": {
        **_DERIVED_LAW, "reads": {}}}}, r"readout_law: stated needs a non-empty reads"),
}


@pytest.mark.parametrize("clause", sorted(CLAUSES_3D))
def test_the_loader_refuses_each_clause_found_by_the_harness(clause, tmp_path):
    from mimarsinan.mapping.platform.target_bundles import ChipKeyUnstatedError, TargetBundleError
    _, changes, match = CLAUSES_3D[clause]
    with pytest.raises(TargetBundleError, match=match) as raised:
        _load(_hand(**changes), tmp_path)
    assert not isinstance(raised.value, ChipKeyUnstatedError)


@pytest.mark.parametrize("where, unit", [
    ("readout_law.adc_bits", "bits"), ("weight_bits", "bits"), ("weight_bits.composition.cell_bits", "bits"),
    ("weight_bits.composition.slices_per_weight", "cells"), ("simulation_steps", None), ("target_tq", None),
])
def test_the_unit_of_a_field_is_read_from_its_name(where, unit):
    from mimarsinan.mapping.platform.descriptions.readings import unit_of
    assert unit_of(where) == unit


def test_target_names_and_technology_blocks_read_the_shipped_bundles_by_default():
    from mimarsinan.mapping.platform.descriptions.realizations import names
    from mimarsinan.mapping.platform.target_bundles import target_bundles, target_names, technology_blocks
    assert target_names() == names(target_bundles())
    assert technology_blocks() == technology_blocks(target_bundles())


def test_read_spans_refuses_a_reads_of_neither_form():
    """read_spans itself (a reads the loader passes to it is a text or an object): a list of fields is no form."""
    from mimarsinan.mapping.platform.descriptions.readings import read_spans
    assert read_spans({"reads": ["adc_bits"], "quote": "adc_bits"}, ["adc_bits"]) == (
        {}, ["reads is a span, or one span per field of an object value, got ['adc_bits']"])


def test_key_value_reads_a_value_that_is_no_object_and_a_schema_of_no_fields_whole():
    """An object key holding no object (the registry refuses it at the bundle's keys; key_value reads it whole), and a
    key whose schema has no fields of its own (the cores' item schema): the value whole."""
    from mimarsinan.mapping.platform.descriptions.computation_keys import key_value
    assert key_value({"readout_law": 5}, "readout_law") == 5
    assert key_value({"cores": {"a": 1}}, "cores") == {"a": 1}


CLAUSES_3G = {
    "K77": ("a derivation that is no collection", {"keys": {"readout_law": dict(_ADC)}, "readout_law": {"stated": {
        **_LAW, "derivation": 5}}}, r"readout_law: derivation is a list"),
    "K78": ("a composition that is a list of its readings", {"weight_bits": {"unstated": {
        "note": "n", "assumed": 8, "composition": ["slices_per_weight", "cell_bits"]}}},
            r"weight_bits.composition: composition takes"),
    "K79": ("a count of slices that is no integer", {"weight_bits": _grid(_st(2.0, "two 4-bit cells"),
                                                                            {"unstated": {"note": "n"}})},
            r"weight_bits.composition.slices_per_weight: a count of cells or bits is a positive integer, got 2.0"),
}


@pytest.mark.parametrize("clause", sorted(CLAUSES_3G))
def test_the_loader_refuses_each_clause_found_by_the_second_run(clause, tmp_path):
    from mimarsinan.mapping.platform.target_bundles import ChipKeyUnstatedError, TargetBundleError
    _, changes, match = CLAUSES_3G[clause]
    with pytest.raises(TargetBundleError, match=match) as raised:
        _load(_hand(**changes), tmp_path)
    assert not isinstance(raised.value, ChipKeyUnstatedError)


def test_a_list_value_is_read_item_by_item():
    from mimarsinan.mapping.platform.descriptions.readings import value_reading_errors
    stated = {"value": [4], "quote": "8-bit inputs", "reads": "8-bit inputs"}
    assert value_reading_errors("x_bits", stated, lambda f: (), {}) != []
    assert value_reading_errors("x_bits", {**stated, "value": [8]}, lambda f: (), {}) == []


# --- [U-CHIPKEYS4] 3(d): a derivation entry's field set is closed -----------------------------------------------------

CLAUSES_4 = {
    "K80": ("a derivation entry with a field of no kind", {"keys": {"readout_law": dict(_ADC)}, "readout_law": {
        "stated": {**_LAW, "derivation": [{"field": "adc_bits", "rule": "sign_and_magnitude", "reads": 12, "why": "t"}]}}},
            r"readout_law: a derivation entry takes the fields \['field', 'rule', 'reads'\], got \['why'\]"),
    "K80b": ("a derivation entry with two fields of no kind", {"keys": {"readout_law": dict(_ADC)}, "readout_law": {
        "stated": {**_LAW, "derivation": [{"field": "adc_bits", "rule": "sign_and_magnitude", "reads": 12, "why": "t",
                                           "note": "n"}]}}},
             r"readout_law: a derivation entry takes the fields \['field', 'rule', 'reads'\], got \['note', 'why'\]"),
    "K80c": ("a composition reading's derivation entry with a field of no kind", {"weight_bits": _grid(
        {"unstated": {"note": "n"}}, _st(5, "two 4-bit cells", derivation=[
            {"field": None, "rule": "sign_and_magnitude", "reads": 4, "source": "s"}]))},
             r"weight_bits.composition.cell_bits: a derivation entry takes the fields .*, got \['source'\]"),
}


@pytest.mark.parametrize("clause", sorted(CLAUSES_4))
def test_a_derivation_entry_s_field_set_is_closed(clause, tmp_path):
    from mimarsinan.mapping.platform.target_bundles import ChipKeyUnstatedError, TargetBundleError
    _, changes, match = CLAUSES_4[clause]
    with pytest.raises(TargetBundleError, match=match) as raised:
        _load(_hand(**changes), tmp_path)
    assert not isinstance(raised.value, ChipKeyUnstatedError)


def test_a_derivation_entry_of_its_three_fields_loads(tmp_path):
    from mimarsinan.mapping.platform.descriptions.readings import DERIVATION_FIELDS
    assert DERIVATION_FIELDS == ("field", "rule", "reads")
    data = _hand(keys={"readout_law": dict(_ADC)}, readout_law={"stated": {**_LAW, "derivation": [
        {"field": "adc_bits", "rule": "sign_and_magnitude", "reads": 12}]}})
    assert _load(data, tmp_path).statements["readout_law"]["derivation"] == [
        {"field": "adc_bits", "rule": "sign_and_magnitude", "reads": 12}]


# --- [U-CHIPKEYS4] 3(f): the faithfulness's pricing, replaced by the pricing a run resolves ---------------------------

def test_a_faithfulness_s_pricing_is_replaced_and_never_changes_what_it_rests_on():
    from mimarsinan.mapping.platform.target_bundles import Faithfulness
    refused = Faithfulness("refused", ("no physics profile", "d"), (), ("no physics profile",))
    declared = refused.with_pricing(("physics profile 'x' declared",))
    assert declared.label() == "refused: physics profile 'x' declared; d"
    assert declared.with_pricing(("no physics profile",)) == refused
    assert Faithfulness("exact", (), ("g",)).with_pricing(("no physics profile",)).label() == (
        'approximate, given "g": no physics profile')
    assert Faithfulness("approximate", ("d",)).with_pricing(()) == Faithfulness("approximate", ("d",))
    assert declared.with_departures(["k declared"]).with_pricing(()).named == ("d", "k declared")


def test_a_description_stating_no_profile_binds_no_technology_s_block(tmp_path):
    """technology_blocks lists a block under the profile a description binds: one stating no profile binds none, whatever
    its cells per weight (the harness's survivor: the profile's test made true lists a block under no technology)."""
    from mimarsinan.mapping.platform.target_bundles import load_bundles, technology_blocks
    data = _hand(data={"core_fields": {"cells_per_weight": 2, "weight_cell_rows": 1}})
    del data["keys"]["platform_physics_profile"]
    (tmp_path / "hand.json").write_text(json.dumps(data), encoding="utf-8")
    known = load_bundles(tmp_path)
    assert known["hand"].values["cores"][0]["cells_per_weight"] == 2 and technology_blocks(known) == {}
