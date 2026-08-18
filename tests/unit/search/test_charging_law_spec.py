"""[TS1] The charging law's SPEC states what the code does — countably.

The TS1 section of ``docs/thesis_support_source_plan.md`` is what a TS2/TS3
implementer reads before extending the accountant, so its countable claims are
load-bearing: a section that says "three sites" above a table of four sends an
implementer looking for one charging point fewer than the code has, and the one
they drop is whichever the prose named last. These tests hold the section's
numerals, its site table and the shipped ``charge_evaluation`` calls to each
other, so a stage that adds a charging site must say so in the spec.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src" / "mimarsinan"
PLAN = REPO / "docs" / "thesis_support_source_plan.md"
CHANNEL_DECLARATIONS = SRC / "search" / "problems" / "joint" / "types.py"

NUMERALS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
CLAIM = re.compile(r"charged at (\w+) sites?, across (\w+) channels?", re.IGNORECASE)
CHARGE_CALL = re.compile(r"(?<![\w.])charge_evaluation\(")
CHANNEL_DECLARATION = re.compile(r"^([A-Z][A-Z_]*_CHANNEL) = \"([a-z_]+)\"$", re.MULTILINE)
SITE_CELL = re.compile(r"`([\w/]+\.py)(?:::(\w+))?`")
BACKTICKED = re.compile(r"`([^`]+)`")


def ts1_section() -> str:
    text = PLAN.read_text(encoding="utf-8")
    start = text.index("## TS1")
    return text[start:text.index("\n## ", start)]


def claimed_counts() -> Tuple[int, int]:
    """The section's own (sites, channels), as an implementer reads them."""
    match = CLAIM.search(ts1_section())
    assert match is not None, (
        "the TS1 section must state how many sites charge and how many "
        "channels they answer for, in a sentence a reader can check"
    )
    words = [match.group(1).lower(), match.group(2).lower()]
    assert all(word in NUMERALS for word in words), f"unreadable numerals: {words}"
    return NUMERALS[words[0]], NUMERALS[words[1]]


def site_rows() -> List[List[str]]:
    """The section's Site/When/Channel table, one list of cells per row."""
    lines = [line.strip() for line in ts1_section().splitlines()]
    header = next(
        (i for i, line in enumerate(lines)
         if line.startswith("|") and "Site" in line and "Channel" in line),
        None,
    )
    assert header is not None, "the TS1 section must table its charging sites"
    rows: List[List[str]] = []
    for line in lines[header + 2:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def charge_call_sites() -> List[str]:
    """Every place the shipped tree charges the accountant, as ``path:line``."""
    hits: List[str] = []
    for path in sorted(SRC.rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if CHARGE_CALL.search(line) and not line.lstrip().startswith("def "):
                hits.append(f"{path.relative_to(SRC).as_posix()}:{number}")
    return hits


def declared_channels() -> Dict[str, str]:
    """The channel constants the joint problem declares: name -> wire value."""
    return dict(CHANNEL_DECLARATION.findall(
        CHANNEL_DECLARATIONS.read_text(encoding="utf-8")
    ))


def files_under(name: str) -> List[Path]:
    return [p for p in SRC.rglob("*.py") if p.as_posix().endswith("/" + name)]


def tabled_file(row: List[str]) -> str:
    """The source file a table row charges in."""
    match = SITE_CELL.search(row[0])
    assert match is not None, f"row {row[0]!r} names no source file"
    return match.group(1)


class TestTheSpecCountsTheSitesTheCodeCharges:
    def test_the_stated_site_count_is_the_number_of_charging_calls(self):
        claimed, _ = claimed_counts()
        sites = charge_call_sites()

        assert claimed == len(sites), (
            f"the TS1 section claims {claimed} charging sites; the tree has "
            f"{len(sites)}: {sites}"
        )

    def test_the_stated_site_count_is_the_number_of_rows_below_it(self):
        claimed, _ = claimed_counts()
        rows = site_rows()

        assert claimed == len(rows), (
            f"the TS1 section claims {claimed} charging sites and then tables "
            f"{len(rows)}: {[row[0] for row in rows]}"
        )

    def test_every_charging_call_is_tabled_where_it_lives(self):
        # Per FILE, not just in total: three rows against validate.py's three
        # calls is the check that catches a merged or a dropped row.
        tabled = Counter(tabled_file(row) for row in site_rows())
        shipped: Counter = Counter()
        for site in charge_call_sites():
            path = site.rsplit(":", 1)[0]
            named = [name for name in tabled if path.endswith(name)]
            shipped[named[0] if named else path] += 1

        assert tabled == shipped, (
            f"the TS1 table charges {dict(tabled)}; the tree charges "
            f"{dict(shipped)}"
        )

    def test_every_row_names_a_file_that_charges_and_a_symbol_that_exists(self):
        for row in site_rows():
            match = SITE_CELL.search(row[0])
            assert match is not None, f"row {row[0]!r} names no source file"
            name, symbol = match.group(1), match.group(2)
            candidates = files_under(name)
            assert len(candidates) == 1, f"{name} resolves to {candidates}"
            source = candidates[0].read_text(encoding="utf-8")
            assert CHARGE_CALL.search(source), f"{name} charges nothing"
            if symbol is not None:
                assert f"def {symbol}(" in source, f"{name} has no {symbol}"


class TestTheSpecCountsTheChannelsTheProblemDeclares:
    def test_the_stated_channel_count_is_the_number_declared(self):
        _, claimed = claimed_counts()
        channels = declared_channels()

        assert claimed == len(channels), (
            f"the TS1 section claims {claimed} channels; the problem declares "
            f"{sorted(channels.values())}"
        )

    def test_every_declared_channel_is_one_a_seam_asks_through(self):
        # A constant nobody passes would inflate the claim without carrying an
        # ask, so the count means "channels a driver can arrive on".
        used = " ".join(
            path.read_text(encoding="utf-8")
            for path in CHANNEL_DECLARATIONS.parent.glob("*.py")
            if path != CHANNEL_DECLARATIONS
        )
        unused = [name for name in declared_channels() if name not in used]

        assert not unused, f"declared but never asked through: {unused}"

    def test_every_channel_the_table_names_is_a_real_one(self):
        values = set(declared_channels().values())
        for row in site_rows():
            named = [token for token in BACKTICKED.findall(row[-1])]
            assert all(token in values for token in named), (
                f"row {row[0]!r} routes to {named}, not one of {sorted(values)}"
            )
