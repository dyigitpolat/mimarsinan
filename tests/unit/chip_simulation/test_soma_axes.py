"""The two soma-law axes: vocabulary, legality, bits-driven derivation, SomaLaw.

``firing_granularity`` and ``membrane_arithmetic`` are DECLARED here and
executed nowhere yet — the whole point of this suite is that the default point
answers exactly as it did before the axes existed, and that the ODIN point is
declarable only where it is meaningful.
"""

import ast
from pathlib import Path

import pytest

from mimarsinan.chip_simulation.soma_axes import (
    FIRING_GRANULARITIES,
    MEMBRANE_ARITHMETICS,
    PER_CYCLE_FIRING,
    PER_EVENT_FIRING,
    SATURATING_UNSIGNED_MEMBRANE,
    UNBOUNDED_MEMBRANE,
    WEIGHT_SIGN_GRANULARITIES,
    derived_firing_granularity,
    derived_membrane_arithmetic,
    legal_firing_granularities,
    legal_membrane_arithmetics,
    resolved_membrane_bits,
)
from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW, SomaLaw

_SRC = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"

_STREAMED_LIF = {"spiking_family": "lif", "spiking_variant": "streamed"}
_WINDOWED_LIF = {"spiking_family": "lif", "spiking_variant": "synchronized"}
_TTFS = {"spiking_family": "ttfs", "spiking_variant": "analytical"}
_TTFS_CYCLE = {"spiking_family": "ttfs", "spiking_variant": "cascaded"}
_MVM = {"core_semantics": "mvm"}


class TestFiringGranularityLegality:
    """per_event is declarable ONLY at the (lif, streamed) point."""

    def test_streamed_lif_admits_both(self):
        assert legal_firing_granularities(_STREAMED_LIF) == FIRING_GRANULARITIES

    @pytest.mark.parametrize(
        "cfg", [_WINDOWED_LIF, _TTFS, _TTFS_CYCLE, _MVM],
        ids=["synchronized", "ttfs", "ttfs_cycle", "mvm"],
    )
    def test_every_other_point_locks_per_cycle(self, cfg):
        assert legal_firing_granularities(cfg) == (PER_CYCLE_FIRING,)

    def test_a_bare_lif_config_derives_streamed_and_admits_both(self):
        assert legal_firing_granularities({"spiking_family": "lif"}) == (
            FIRING_GRANULARITIES
        )

    def test_the_predicate_is_total_over_partial_and_invalid_configs(self):
        for cfg in ({}, {"spiking_family": "banana"}, {"spiking_variant": "banana"},
                    {"spiking_mode": "lif"}, {"core_semantics": "mvm",
                                              "spiking_variant": "streamed"}):
            legal = legal_firing_granularities(cfg)
            assert set(legal) <= set(FIRING_GRANULARITIES) and legal

    def test_the_derived_default_is_never_per_event(self):
        for cfg in (_STREAMED_LIF, _WINDOWED_LIF, _TTFS, _MVM, {}):
            assert derived_firing_granularity(cfg) == PER_CYCLE_FIRING


class TestMembraneArithmeticLegality:
    def test_lif_admits_both(self):
        assert legal_membrane_arithmetics(_STREAMED_LIF) == MEMBRANE_ARITHMETICS
        assert legal_membrane_arithmetics(_WINDOWED_LIF) == MEMBRANE_ARITHMETICS

    @pytest.mark.parametrize("cfg", [_TTFS, _TTFS_CYCLE, _MVM],
                             ids=["ttfs", "ttfs_cycle", "mvm"])
    def test_non_lif_locks_unbounded(self, cfg):
        assert legal_membrane_arithmetics(cfg) == (UNBOUNDED_MEMBRANE,)

    def test_an_unknown_family_rules_nothing_out(self):
        assert legal_membrane_arithmetics({"spiking_family": "banana"}) == (
            MEMBRANE_ARITHMETICS
        )


class TestBitsDrivenDerivation:
    def test_no_width_derives_unbounded(self):
        assert derived_membrane_arithmetic(_STREAMED_LIF) == UNBOUNDED_MEMBRANE
        assert derived_membrane_arithmetic({**_STREAMED_LIF, "membrane_bits": 0}) == (
            UNBOUNDED_MEMBRANE
        )

    def test_a_declared_width_derives_saturating(self):
        assert derived_membrane_arithmetic({**_STREAMED_LIF, "membrane_bits": 8}) == (
            SATURATING_UNSIGNED_MEMBRANE
        )

    def test_the_derived_value_never_leaves_the_legal_set(self):
        """A width declared against TTFS must not silently derive an illegal
        arithmetic; the contract check is what reports the contradiction."""
        derived = derived_membrane_arithmetic({**_TTFS, "membrane_bits": 8})
        assert derived in legal_membrane_arithmetics({**_TTFS, "membrane_bits": 8})

    @pytest.mark.parametrize("raw,expected", [
        (None, 0), (0, 0), (8, 8), ("12", 12), (-4, 0), ("banana", 0), (True, 0),
    ])
    def test_the_width_reader_is_total(self, raw, expected):
        assert resolved_membrane_bits({"membrane_bits": raw}) == expected


class TestSomaLaw:
    def test_the_default_point_is_todays_law(self):
        law = SomaLaw.resolve({})
        assert law.firing_granularity == PER_CYCLE_FIRING
        assert law.membrane_arithmetic == UNBOUNDED_MEMBRANE
        assert law.membrane_bits == 0
        assert law.bias_slot == "tail"
        assert law.is_default_point
        assert law.point_tag() is None
        assert law == DEFAULT_SOMA_LAW

    def test_the_odin_point_resolves_from_a_config(self):
        law = SomaLaw.resolve({
            **_STREAMED_LIF, "firing_mode": "Novena", "thresholding_mode": "<=",
            "firing_granularity": PER_EVENT_FIRING, "membrane_bits": 8,
        })
        assert law.firing_granularity == PER_EVENT_FIRING
        assert law.membrane_arithmetic == SATURATING_UNSIGNED_MEMBRANE
        assert law.membrane_bits == 8
        assert law.is_per_event and law.saturates
        assert not law.is_default_point
        assert law.point_tag() == "per_event-sat8"

    def test_the_tag_discriminates_widths(self):
        eight = SomaLaw.resolve({**_STREAMED_LIF, "membrane_bits": 8})
        sixteen = SomaLaw.resolve({**_STREAMED_LIF, "membrane_bits": 16})
        assert eight.point_tag() == "sat8"
        assert sixteen.point_tag() == "sat16"
        assert eight != sixteen

    def test_resolving_an_attribute_carrier_agrees_with_the_mapping(self):
        cfg = {**_STREAMED_LIF, "firing_granularity": PER_EVENT_FIRING,
               "membrane_bits": 8, "firing_mode": "Novena"}
        from types import SimpleNamespace

        carrier = SimpleNamespace(
            firing_mode="Novena", thresholding_mode="<=",
            firing_granularity=PER_EVENT_FIRING,
            membrane_arithmetic=SATURATING_UNSIGNED_MEMBRANE, membrane_bits=8,
        )
        assert SomaLaw.resolve(carrier) == SomaLaw.resolve(cfg)

    def test_the_law_is_frozen(self):
        with pytest.raises(Exception):
            DEFAULT_SOMA_LAW.membrane_bits = 8  # type: ignore[misc]

    def test_nothing_constructs_the_law_outside_its_one_constructor(self):
        """The law has ONE constructor; a second call site is a second SSOT."""
        direct = []
        for path in _SRC.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "SomaLaw"):
                    direct.append(f"{path.relative_to(_SRC).as_posix()}:{node.lineno}")
        assert direct == [], direct

    def test_the_constructor_itself_has_exactly_one_body(self):
        home = _SRC / "chip_simulation" / "soma_law.py"
        tree = ast.parse(home.read_text(encoding="utf-8"))
        constructions = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "cls"
        ]
        assert len(constructions) == 1


class TestVocabularyIsClosed:
    def test_no_axis_reuses_a_taken_word(self):
        """'windowed' already means variant == synchronized (§14 J2-4)."""
        assert "windowed" not in FIRING_GRANULARITIES
        assert FIRING_GRANULARITIES == ("per_cycle", "per_event")
        assert MEMBRANE_ARITHMETICS == ("unbounded", "saturating_unsigned")
        assert WEIGHT_SIGN_GRANULARITIES == ("per_synapse", "per_axon")
