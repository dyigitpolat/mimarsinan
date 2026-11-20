"""[U-CHIPKEYS4] Items 3(a) and 3(b) of the third verifies' rulings: the per-condition sweep as a class. Every member of
every character class of the rule's regular expressions (ranges and escapes included), every element of their structure
(lookarounds, optional parts, quantifiers, word boundaries, anchors, counted repeats, flags), every ordering comparison at
its boundary sibling and every integer constant of a comparison moved by one has a hand-made case here or in the unit's
other three test files, each red on its mutant (the harness, steps/harness.py, lists them).

The verify's named mutants come first, each a case of its own: the three survivors of the ruled element kind (V0563 a
leading-dot decimal, V0633 a span spelling one capitalized option for another, V0746 a capitalized bundle and
realization name) and the real ones of the verify's extra kinds (V0295, V0312, V0313 spans cut at a quote's first or
second character; V0574, V0575 malformed digit groupings; V0576 a number read inside a word; V0586 a format read inside
a word; V0610, V0611 a width's spaced unit read as a count; V0644 an option spelled inside a word; V0750 an empty name;
V0753 a digest of the wrong length). Every case reads the rule's own functions with hand-made text: none reads a shipped
description."""

import copy
import json
from pathlib import Path

import pytest

import mimarsinan

TARGETS_DIR = Path(mimarsinan.__file__).parent / "mapping" / "platform" / "targets"


def _tokens(span):
    from mimarsinan.mapping.platform.descriptions.readings import number_tokens
    return [value for value, _, _ in number_tokens(span)]


def _integer_error(span, number, unit="bits"):
    from mimarsinan.mapping.platform.descriptions.readings import integer_reading_error
    return integer_reading_error(span, number, unit)


def _option_error(span, value, options):
    from mimarsinan.mapping.platform.descriptions.readings import option_reading_error
    return option_reading_error(span, value, options)


def _boundaries(span, quote):
    from mimarsinan.mapping.platform.descriptions.readings import span_at_token_boundaries
    return span_at_token_boundaries(span, quote)


# --- the verify's named mutants, first --------------------------------------------------------------------------------

def test_a_leading_dot_decimal_is_no_number_v0563():
    """V0563 (the numeral's lookbehind class without 0-9): ".55" is no integer, nor its second digit."""
    assert _tokens("a .55 V supply") == []
    assert _integer_error("a .55 V supply and 5-bit cells", 5) is None  # the 5 the span states is read, not ".55"'s
    assert _integer_error("a .55 V supply", 5) is not None


def test_a_span_spelling_one_capitalized_option_for_another_refuses_v0633():
    """V0633 (the spelled pattern's first class without A-Z): a capitalized closed option is spelled, so a span
    spelling Default for a stated Novena refuses."""
    options = ("Default", "Novena", "TTFS")
    assert _option_error("the neuron resets as Default does", "Novena", options) is not None
    assert _option_error("the neuron resets as Novena does", "Novena", options) is None


def _hand_bundle(name, real=None):
    """A hand-made bundle named ``name`` (on a profile of its own), with one realization named ``real``."""
    from unit.mapping.platform.hand_descriptions import described
    data = described({"name": name, "geometry": "recom_64x64_256",
                      "keys": {"platform_physics_profile": "generic_estimated_22nm"}, "faithfulness": {"class": "exact"}})
    if real is not None:
        data["realizations"] = {real: {
            "geometry": {"name": f"{name}_{real}_geometry", "cores": [{"max_axons": 64, "max_neurons": 64, "count": 2,
                                                                       "has_bias": False}],
                         "weight_bits": 8, "provenance": "a hand-made realization", "claim_class": "quarantined"},
            "computation": {}, "pricing_absent": "a hand-made realization no profile prices",
            "record": {"report": "r.json", "sha256": "0" * 64, "result": "a hand-made record"}}}
    return data


def _load_dir(tmp_path, data):
    from mimarsinan.mapping.platform.target_bundles import load_bundles
    (tmp_path / f"{data['name']}.json").write_text(json.dumps(data), encoding="utf-8")
    return load_bundles(tmp_path)


@pytest.mark.parametrize("name, real", [("Hand", "Wide"), ("hand2", "r2"), ("hand_x", "wide_x")])
def test_a_name_of_capitals_digits_or_underscores_is_a_name_v0746(name, real, tmp_path):
    """V0746 (the name class without A-Z), and its digits and underscore: each member names a bundle and a
    realization."""
    from mimarsinan.mapping.platform.target_bundles import get_target_bundle, target_names
    known = _load_dir(tmp_path, _hand_bundle(name, real))
    assert target_names(known) == [name, f"{name}@{real}"]
    assert get_target_bundle(f"{name}@{real}", known).name == f"{name}@{real}"


def test_an_empty_realization_name_refuses_v0750(tmp_path):
    """V0750 (the name's '+' made '*'): a name is one character at least."""
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError
    with pytest.raises(TargetBundleError, match=r"realization '': a realization name is letters, digits and underscores"):
        _load_dir(tmp_path, _hand_bundle("hand", ""))


@pytest.mark.parametrize("digest, ok", [("0" * 64, True), ("a" * 64, True), ("0" * 63, False), ("0" * 65, False),
                                        ("a" * 63, False), ("A" * 64, False)])
def test_a_digest_is_sixty_four_lower_hex_digits_v0753(digest, ok, tmp_path):
    """V0753 (the digest's counted repeat loosened): 63 or 65 digits refuse, as capitals do; digits alone and letters
    alone are hex."""
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError
    data = _hand_bundle("hand", "r")
    data["realizations"]["r"]["record"]["sha256"] = digest
    if ok:
        _load_dir(tmp_path, data)
    else:
        with pytest.raises(TargetBundleError, match=r"record.sha256 is the report's 64-hex sha256"):
            _load_dir(tmp_path, data)


@pytest.mark.parametrize("span, quote", [
    ("6-bit output", "16-bit output"),  # V0295: cut at the quote's first character (a number)
    ("6-bit", "16-bit output"),
    ("atrix", "matrix"),  # a word cut at its first character
    ("5 V supply", "2.5 V supply"),  # V0312, V0313: cut after a decimal mark at the quote's second character
    ("024 rows", "1,024 rows"),  # after a group mark at the second character
    ("1", "1,2"),  # cut before a group mark one digit from the quote's end
])
def test_a_span_cut_at_the_quote_s_first_or_second_character_refuses_v0295_v0312_v0313(span, quote):
    assert not _boundaries(span, quote)


@pytest.mark.parametrize("span", ["1234,567 rows", "1,24 rows", "12,34,567 rows", "1,0245 rows"])
def test_a_malformed_digit_grouping_is_no_integer_v0574_v0575(span):
    """V0574 and V0575 (the grouping's counted repeats loosened): a group of four leading digits, a group of two
    digits, are no thousands grouping."""
    assert _tokens(span)[0] is None


def test_a_three_digit_leading_group_is_one_number():
    """The leading group's upper bound (three digits): "128,000" is 128000."""
    assert _tokens("a 128,000-cell array") == [128000]


@pytest.mark.parametrize("span", ["fourteen rows", "someone's 8-bit word", "nineteen rows"])
def test_a_number_word_is_read_as_a_whole_word_only_v0576(span):
    """V0576 (the word pattern's closing \\b removed) and its opening \\b: "four" is no number in "fourteen", nor
    "one" in "someone"."""
    assert 4 not in _tokens(span) and 1 not in _tokens(span) and 9 not in _tokens(span)


def test_a_format_is_read_as_a_whole_word_only_v0586():
    """V0586 (the format's lookbehind removed): "point8" holds no INT8 format."""
    assert _integer_error("4 cells, point8", 8) is not None


@pytest.mark.parametrize("span", ["8 -bits cells", "4 bit cells", "4 bits cells", "4 - bit cells", "2 -b cells"])
def test_a_width_s_spaced_unit_is_never_a_count_v0610_v0611(span):
    """V0610 and V0611 (the count's negative lookahead's optional parts made required): a width, however its unit is
    spaced or spelled, is never a count of cells."""
    number = int(span.split()[0])
    assert _integer_error(span, number, "cells") is not None


@pytest.mark.parametrize("span", ["nonrequantize at the output", "requantized at the output"])
def test_an_option_spelled_inside_a_word_is_not_spelled_v0644(span):
    """V0644 (the option's opening \\b removed) and its closing \\b: "requantize" is spelled as a whole word only."""
    assert _option_error(span, "identity", ("identity", "requantize")) is None


# --- the numeral ----------------------------------------------------------------------------------------------------

def test_a_number_before_a_sentence_s_full_stop_is_read():
    """The decimal part's digits are one at least: "8." is 8 and its full stop."""
    assert _tokens("a width of 8.") == [8]
    assert _integer_error("a width of 8.", 8) is None


# --- the width's unit and format --------------------------------------------------------------------------------------

@pytest.mark.parametrize("span, number", [
    ("4 cells and 8 bits", 8),  # the unit spaced, with no hyphen
    ("4 cells of 32b words", 32),  # the unit's one-letter form
    ("4 cells and a 6-Bit output", 6),  # the unit in capitals (the flag)
    ("4 cells of int8 values", 8),  # each format: int, uint, fp, bf
    ("4 cells of uint8 values", 8),
    ("4 cells of FP16 values", 16),  # in capitals (the flag)
    ("4 cells of bf16 values", 16),
    ("4 cells of INT8 values", 8),
    ("fp32 words in 8 rows", 32),  # a format right before its number, not anywhere before it
])
def test_a_width_is_read_by_its_unit_or_its_format(span, number):
    assert _integer_error(span, number) is None


@pytest.mark.parametrize("span, number", [
    ("a 4-byte word of 8 cells", 4),  # a unit that only starts as a bit's ("byte")
    ("4 cells, point8", 8),  # a format inside a word
])
def test_a_width_s_unit_or_format_is_read_as_a_whole_word_only(span, number):
    assert _integer_error(span, number) is not None


# --- the count of cells -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("span, number", [
    ("Two 4-bit cells", 2),  # a number word in capitals (the flag)
    ("two big cells of 8 rows", 2),  # a word starting like a width's unit ("big") is no width
    ("two 4-Bit CELLS of 8 rows", 2),  # the count's words in capitals
    ("8 Crossbars and 4 rows", 8),
    ("4 PCM devices and 8 rows", 4),
    ("two cells of 8 rows", 2),  # no word between the count and its cells
    ("two RRAM memory cells of 8 rows", 2),  # two words between
    ("two 4-bit memory cells of 8 rows", 2),  # a width, then one word
    ("two 4 bit cells of 8 rows", 2),  # a width spaced from its unit
    ("two  cells of 8 rows", 2),  # two spaces
    ("8 cell arrays and 4 rows", 8),  # the cells' singular
    ("8 device arrays and 4 rows", 8),
    ("8 crossbar arrays and 4 rows", 8),
    ("8 pcm arrays and 4 rows", 8),
])
def test_a_count_is_read_by_the_cells_it_counts_in_each_form(span, number):
    assert _integer_error(span, number, "cells") is None


@pytest.mark.parametrize("span, number", [
    ("8crossbars of 4 rows", 8),  # no space between the count and its cells
    ("two very big memory cells of 8 rows", 2),  # three words between: no count
    ("two cellsx of 8 rows", 2),  # a word that only starts as the cells'
    ("two  - bits big memory cells of 8 rows", 2),  # a width's unit without its number is a word
    ("two  -bit big memory cells of 8 rows", 2),  # a unit's word without its letters is a word
    ("two 4-bitcells of 8 rows", 2),  # a width joined to the cells
    ("two x-bitcells of 8 rows", 2),  # a spelled width joined to the cells
    ("two memorycells of 8 rows", 2),  # a word joined to the cells
])
def test_a_count_is_not_read_beyond_its_form(span, number):
    assert _integer_error(span, number, "cells") is not None


# --- the spelled options ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("span, value, options, refuses", [
    ("a bit serial array", "parallel", ("Bit_Serial", "parallel"), True),  # a capital after the underscore
    ("the E class", "L", ("L", "E"), False),  # a one-letter code has no spelling
    ("an ab mode", "cd", ("ab", "cd"), True),  # a two-letter option is spelled
    ("per event", "ab_cd", ("per_", "ab_cd"), False),  # an option ending at its underscore has no spelling
    ("updated Per Event", "per_cycle", ("per_event", "per_cycle"), True),  # spelled in capitals (the flag)
])
def test_the_spelled_option_rule_reads_each_form_of_an_option(span, value, options, refuses):
    assert (_option_error(span, value, options) is not None) is refuses


# --- [U-CHIPKEYS5] (4): each whitespace run read as one space, one case per whitespace form ---------------------------
# The reading rule collapses each run of whitespace of a quote and of a span to one space before it reads
# (readings.plain, the first act of read_spans, span_at_token_boundaries, integer_reading_error and
# option_reading_error), so every pattern sees single spaces and a tab, a newline or a doubled space reads as a space.

#: The whitespace forms, each read as one space.
WHITESPACE_FORMS = {"tab": "\t", "newline": "\n", "doubled space": "  ", "space and tab": " \t", "carriage return": "\r\n"}


def _spaced(text, form):
    return text.replace(" ", form)


@pytest.mark.parametrize("form", sorted(WHITESPACE_FORMS))
def test_plain_reads_each_whitespace_run_as_one_space(form):
    from mimarsinan.mapping.platform.descriptions.readings import plain
    assert plain(_spaced("8 2 - bit cells", WHITESPACE_FORMS[form])) == "8 2 - bit cells"
    assert plain("a b") == "a b" and plain("ab") == "ab"


@pytest.mark.parametrize("form", sorted(WHITESPACE_FORMS))
@pytest.mark.parametrize("span, number, unit", [
    ("8 2 - bit cells", 8, "cells"),  # a count with a spaced width (the verify's M0891, M0895)
    ("8 4-bit cells", 8, "cells"),  # M0901, M0898
    ("8 four-bit multi level cells", 8, "cells"),  # M0910, M0907
    ("8 memristor cells of 4 bits", 8, "cells"),  # M0922, M0919
    ("8 12-bit cells", 8, "cells"),  # M0882
    ("4 cells and a 6 - bit output", 6, "bits"),  # a width's spaced unit
])
def test_a_reading_in_any_whitespace_form_reads_as_with_one_space(span, number, unit, form):
    """Read: the span with each space in the form reads as the span with one space."""
    assert _integer_error(_spaced(span, WHITESPACE_FORMS[form]), number, unit) is None
    assert _integer_error(span, number, unit) is None


@pytest.mark.parametrize("form", sorted(WHITESPACE_FORMS))
@pytest.mark.parametrize("span, number", [
    ("4 - bit", 4),  # a width's spaced unit is never a count (the verify's M0850, M0854)
    ("2 bit cells", 2),  # M0874
    ("2 - bit cells", 2),  # M0877
])
def test_a_width_in_any_whitespace_form_is_never_a_count(span, number, form):
    assert _integer_error(_spaced(span, WHITESPACE_FORMS[form]), number, "cells") is not None


@pytest.mark.parametrize("form", sorted(WHITESPACE_FORMS))
def test_an_option_spelled_in_any_whitespace_form_is_spelled(form):
    options = ("bit_serial", "parallel")
    assert _option_error(_spaced("a bit serial array", WHITESPACE_FORMS[form]), "parallel", options) is not None
    assert _option_error(_spaced("a bit serial array", WHITESPACE_FORMS[form]), "bit_serial", options) is None


@pytest.mark.parametrize("form", sorted(WHITESPACE_FORMS))
def test_a_span_in_any_whitespace_form_is_a_span_of_its_quote(form):
    """A span written with one space is a span of a quote that holds the form there, and the other way round."""
    from mimarsinan.mapping.platform.descriptions.readings import read_spans
    quote = "each core holds 8 4-bit cells of 64 rows"
    spaced = _spaced(quote, WHITESPACE_FORMS[form])
    assert _boundaries("8 4-bit cells", spaced) and _boundaries(_spaced("8 4-bit cells", WHITESPACE_FORMS[form]), quote)
    assert read_spans({"reads": "8 4-bit cells", "quote": spaced}, [None]) == ({None: "8 4-bit cells"}, [])
    assert read_spans({"reads": _spaced("8 4-bit", WHITESPACE_FORMS[form]), "quote": quote}, [None])[1] == []
    assert read_spans({"reads": "8 4-bit cells", "quote": _spaced("each core holds 8 4-bitcells", "x")}, [None])[1]


# --- [U-CHIPKEYS5] (4): the verify's witnesses, one case per real survivor the rule does not make equivalent -----------

def test_a_thousands_number_of_two_groups_is_one_number_m0839():
    """M0839 (_INTEGER's group repeat made exactly one)."""
    assert _tokens("1,000,000 synapses") == [1000000]
    assert _integer_error("1,000,000 synapses", 1000000, None) is None


def test_a_two_digit_width_in_a_count_is_read_m0889():
    """M0889 (the count's width digits made one)."""
    assert _integer_error("8 12-bit cells", 8, "cells") is None


def test_a_span_cut_at_an_underscore_cuts_a_token_m0962():
    """M0962 (the token characters' \\w without the underscore): "bits" in "cell_bits" cuts a word."""
    assert not _boundaries("bits", "the cell_bits field")
    assert not _boundaries("cell", "the cell_bits field")


# --- [U-CHIPKEYS5] (4): the verify's 28 range-end witnesses (m-1), one case each ---------------------------------------

@pytest.mark.parametrize("span, tokens, number, unit, read", [
    ("a .09 V supply", [], 9, None, False),  # M0805: the numeral's lookbehind range without 0
    ("a .95 V supply", [], 5, None, False),  # M0806: ... without 9
    ("a 9-bit signed integer", [9], 9, "bits", True),  # M0810, M0841: the numeral's and the integer's digits without 9
    ("0.9 V", [None], 0, None, False),  # M0816: the decimal part's digits without 9
    ("10,000 rows", [10000], 10000, None, True),  # M0822: the integer's group digits without 0
    ("9,024 rows", [9024], 9024, None, True),  # M0823: the leading group's digits without 9
    ("1,999 rows", [1999], 1999, None, True),  # M0831: the group's digits without 9
])
def test_each_digit_range_end_reads_its_numbers(span, tokens, number, unit, read):
    assert _tokens(span) == tokens
    assert (_integer_error(span, number, unit) is None) is read


@pytest.mark.parametrize("span", ["8 10-bit cells", "8 19-bit cells",  # M0886, M0887: the count's width digits
                                  "8 a-bit multi level cells of 4 bits",  # M0902: the spelled width's letters without a
                                  "8 z-bit multi level cells of 4 bits",  # M0903: ... without z
                                  "8 zinc cells of 4 bits"])  # M0914: the count's words' letters without z
def test_each_range_end_of_the_count_reads_its_count(span):
    assert _integer_error(span, 8, "cells") is None


@pytest.mark.parametrize("span", ["an aint8 x 4 cells", "a zint8 x 4 cells"])  # M0864, M0865: the lookbehind's a, z
def test_a_format_after_a_letter_at_either_range_end_is_no_format(span):
    assert _integer_error(span, 8, "bits") is not None


@pytest.mark.parametrize("span, options", [
    ("ttfs quantized cores", ("lif", "ttfs_quantized")),  # M0956: a later segment's letters without z
    ("the core is AZ_ZA here", ("lif", "AZ_ZA")),  # the first segment's capitals without A (the verify's M0943)
    ("the core is ZETA here", ("lif", "ZETA")),  # ... without Z (M0944)
])
def test_an_option_of_letters_at_each_range_end_is_spelled(span, options):
    assert _option_error(span, "lif", options) is not None


@pytest.mark.parametrize("name", ["A_wide", "Z_wide", "wide_z", "wide_9"])  # M1150, M1151, M1154, M1157
def test_a_name_of_each_range_end_is_a_name(name):
    from mimarsinan.mapping.platform.descriptions.realizations import NAME
    assert NAME.fullmatch(name) is not None


@pytest.mark.parametrize("report", ["A/report.json", "Z/report.json", "z/report.json", "run_9/report.json",  # M1163-70
                                    "a/A.json", "a/Z.json", "a/z.json", "a/9.json"])  # M1177, M1178, M1181, M1184
def test_a_report_path_of_each_range_end_is_a_path(report):
    from mimarsinan.mapping.platform.descriptions.realizations import RECORD_FORMS
    assert RECORD_FORMS["report"][1](report)


# --- [U-CHIPKEYS5] m-5: a byte's 'B' anchors no width in bits ----------------------------------------------------------

@pytest.mark.parametrize("span, number, read", [
    ("1 B and 8 cells", 1, False),  # the verify's probe H: a byte unit is no width in bits
    ("a 2-B word of 8 cells", 2, False),
    ("1 b and 8 cells", 1, True),  # the bit's one-letter unit
    ("32b outputs of 8 cells", 32, True),
    ("8 cells of 16 BITS", 16, True),  # the bit's spelled unit in any case
    ("8 cells of 16 Bit words", 16, True),
])
def test_a_width_s_one_letter_unit_is_the_bit_s_b(span, number, read):
    assert (_integer_error(span, number, "bits") is None) is read


# --- [U-CHIPKEYS5] (4): each quantifier at every other form, its case (the harness's quantifier kind) -----------------

@pytest.mark.parametrize("span, number, unit, read", [
    ("a 6--bit output of 4 cells", 6, "bits", False),  # a width's hyphen at most once
    ("4 cells of 6 bitss", 6, "bits", False),  # a width's unit's plural s at most once
    ("a uuint8 word of 4 cells", 8, "bits", False),  # a format's u at most once
    ("2 --bit cells of 4 rows", 2, "cells", True),  # the count's lookahead: a hyphen at most once before the unit
    ("2 bitss cells of 4 rows", 2, "cells", True),  # ... a unit's s at most once: "bitss" is a word
    ("8 4--bit cells", 8, "cells", False),  # the count's width: a hyphen at most once
    ("8 4-bitss cells", 8, "cells", False),  # ... the unit's s at most once
    ("8 four-bitss big memory cells", 8, "cells", False),  # the spelled width's s at most once: then three words
    ("8 4-bit 2-bit cells", 8, "cells", False),  # one width at most
    ("8 cellss of 4 rows", 8, "cells", False),  # each counted noun's s at most once
    ("8 devicess of 4 rows", 8, "cells", False),
    ("8 crossbarss of 4 rows", 8, "cells", False),
    ("8 pcmss of 4 rows", 8, "cells", False),
])
def test_each_optional_element_is_read_at_most_once(span, number, unit, read):
    assert (_integer_error(span, number, unit) is None) is read


def test_an_option_of_several_underscored_parts_is_spelled():
    """The spelled option's underscored parts repeat: "bit_serial_x" is spelled "bit serial x"."""
    assert _option_error("the bit serial x mode", "parallel", ("bit_serial_x", "parallel")) is not None
    assert _option_error("the bit serial x mode", "bit_serial_x", ("bit_serial_x", "parallel")) is None


# --- [U-CHIPKEYS5] (4): IGNORECASE added where a class holds both cases is no equivalent ------------------------------
# Python's IGNORECASE folds some letters outside A-Z onto A-Z (U+0130 and U+0131 onto i, U+212A KELVIN SIGN onto k,
# U+017F LONG S onto s), so a class of ASCII letters takes them under the flag: each pattern's case.

NON_ASCII_FOLDING = ["\u0130", "\u0131", "\u212a", "\u017f"]


@pytest.mark.parametrize("letter", NON_ASCII_FOLDING)
def test_a_name_and_a_report_path_are_ascii_whatever_folds_onto_ascii(letter):
    from mimarsinan.mapping.platform.descriptions.realizations import NAME, RECORD_FORMS
    assert NAME.fullmatch(f"wide{letter}") is None
    assert not RECORD_FORMS["report"][1](f"results/{letter}.json")


@pytest.mark.parametrize("letter", NON_ASCII_FOLDING)
def test_an_option_of_a_letter_folding_onto_ascii_is_not_spelled(letter):
    assert _option_error(f"the {letter}x mode", "parallel", (f"{letter}x", "parallel")) is None
