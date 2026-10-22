"""The model-level screen: deterministic by seed, clean with every hypothesis on, every arm drawn by name."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))

from x3a_golden.generator import CONTROLS, DOMAINS, generate  # noqa: E402
from x3a_golden.hypotheses import D17_ARMS, LABELS  # noqa: E402
from x3a_golden.program import canonical_json  # noqa: E402
from x3a_golden.screen import clopper_pearson, run_instance, summarize  # noqa: E402


def test_the_generator_is_a_pure_function_of_its_seed():
    first = generate(random.Random("x3a-golden/all_on/7"), "all_on")
    again = generate(random.Random("x3a-golden/all_on/7"), "all_on")
    assert first[0].digest() == again[0].digest()
    assert canonical_json({k: v.tolist() for k, v in first[1].items()}) == \
        canonical_json({k: v.tolist() for k, v in again[1].items()})


@pytest.mark.parametrize("arm", ["all_on", "H5-skew", "H7", "H10"])
def test_an_instance_record_is_byte_identical_twice(arm):
    for seed in range(3):
        assert canonical_json(run_instance(arm, seed)) == canonical_json(run_instance(arm, seed))


def test_all_hypotheses_on_give_no_counterexample_on_a_sample():
    recs = [run_instance("all_on", seed) for seed in range(60)]
    row = summarize("all_on", recs)
    assert row["counterexamples"] == 0, row["first_counterexamples"]
    assert row["packed_compared"] >= 40 and row["network_compared"] >= 20, row
    assert set(row["semantics"]) == {"streamed_lif", "sync_lif", "sync_ttfs", "value"}


def test_every_arm_has_a_domain_and_twelve_d17_arms_exist():
    assert len(D17_ARMS) == 12 and set(LABELS) - set(D17_ARMS) == {"R1", "R2", "R3", "R6"}
    assert set(DOMAINS) == {"all_on", *LABELS, *CONTROLS}


def test_clopper_pearson_matches_closed_forms():
    assert clopper_pearson(0, 1000)[1] == pytest.approx(1 - 0.05 ** (1 / 1000))
    lo, hi = clopper_pearson(5, 10)
    assert lo == pytest.approx(0.18709, abs=1e-4) and hi == pytest.approx(0.81291, abs=1e-4)
    assert clopper_pearson(10, 10) == (pytest.approx((0.025) ** (1 / 10)), 1.0)


def test_a_packed_program_round_trips_through_its_exchange_json():
    import json

    from x3a_golden.machine import run_packed
    from x3a_golden.schedule import Packed
    from x3a_golden.screen import pack

    for seed in range(6):
        rng = random.Random(f"x3a-golden/all_on/{seed}")
        prog, inputs, knobs, _ = generate(rng, "all_on")
        try:
            packed = pack(prog, rng, knobs, frozenset(), next(iter(inputs.values())).shape[0])
        except Exception as err:  # a refused declaration has no artifact to exchange
            assert type(err).__name__ == "GoldenRefusal"
            continue
        again = Packed.from_json(prog, json.loads(json.dumps(packed.to_json())))
        first, second = run_packed(prog, packed, inputs)[0], run_packed(prog, again, inputs)[0]
        assert all((first[k] == second[k]).all() for k in first)
