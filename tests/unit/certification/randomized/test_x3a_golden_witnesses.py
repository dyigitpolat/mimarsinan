"""Every model-level hypothesis has a witness: a difference with it off, equality or a named refusal with it on.

The witnesses are hand-built minimal programs and packings (``scripts/x3a_golden/witnesses.py``); the
expected values below were derived by hand from the box's kernels (D17's witness table: H1-theta 1 vs 0
spikes, H7 theta 5, w = [+3, -3], 1 spike adjacent and 0 interleaved, H8 float32 at 2**24 + 2).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))

from x3a_golden.hypotheses import LABELS  # noqa: E402
from x3a_golden.witnesses import run_witness  # noqa: E402

#: label -> (expected, got) of the off arm's first divergence, and the on arm's outcome.
EXPECTED = {
    "H1-theta": ((0, 1), "refused:H1-theta"),
    "H1-latency": ((2, 1), "refused:H1-latency"),
    "H2": ((2, 1), "refused:H2"),
    "H3": ((0, 1), "refused:H3"),
    "H4": ((2, 1), "refused:H4"),
    "H5-collapse": ((2, 4), "equal"),
    "H5-skew": ((1, 2), "equal"),
    "H6": ((6, 2), "equal"),
    "H7": ((1, 0), "equal"),
    "H8": (((1 << 24) + 2, 1 << 24), "refused:H8"),
    "H9": ((2, 0), "refused:H9"),
    "H10": ((2, 0), "refused:H10"),
    "R1": ((1, 2), "equal"),
    "R2": ((1, 2), "equal"),
    "R3": ((4, 2), "equal"),
    "R6": ((1, 0), "equal"),
}


def test_every_label_has_a_witness():
    assert set(EXPECTED) == set(LABELS)


@pytest.mark.parametrize("label", sorted(EXPECTED))
def test_the_witness_differs_off_and_holds_on(label):
    (expected, got), on = EXPECTED[label]
    result = run_witness(label)
    assert result["off"] is not None and result["off"] != "not admitted", result
    assert (result["off"]["expected"], result["off"]["got"]) == (expected, got), result
    assert result["on"] == on, result


def test_a_core_wide_event_loop_makes_coalescing_function_bearing():
    """Finding: H7 must scope a per-event core's fire checks to the neuron's own synaptic events."""
    from x3a_golden.witnesses import foreign_event_witness

    assert foreign_event_witness() == {"identity": 1, "coalesced_own": 1, "coalesced_core_wide": 2}
