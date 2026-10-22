"""E192 ratchet: every time or energy constant is read by a pricing formula or declared unmultiplied."""

from pathlib import Path

from mimarsinan.deployment_record.platform_physics import vocabulary
from mimarsinan.deployment_record.units import ENERGY, TIME

_PRICERS = Path(vocabulary.__file__).resolve().parents[1] / "cost" / "absolute"
_DECLARED = "declared, not multiplied"
#: Shrink-only: time rows no formula reads until a compute model per core class (E187, E188) lands.
UNREAD_ALLOWLIST = frozenset({"t_array_read", "t_adc_conversion", "t_hop"})


def _pricing_source() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(_PRICERS.glob("*.py")))


def unread_rows(rows, source: str):
    """Time and energy rows that no pricing formula names and that are not declared unmultiplied."""
    return sorted(
        key for key, _group, dimension, _unit, multiplicand, _doc in rows
        if dimension in (ENERGY, TIME) and multiplicand != _DECLARED and f'"{key}"' not in source
    )


def test_every_time_and_energy_row_is_read_or_allowlisted():
    assert unread_rows(vocabulary.ROWS, _pricing_source()) == sorted(UNREAD_ALLOWLIST)


def test_a_seeded_unread_time_row_fails_the_ratchet():
    seeded = vocabulary.ROWS + (("t_seeded_unread", vocabulary.GLOBAL, TIME, "ns", "latency_steps",
                                 "A row no formula reads, seeded to prove the ratchet bites."),)
    assert "t_seeded_unread" in unread_rows(seeded, _pricing_source())


def test_the_allowlist_only_shrinks():
    assert len(UNREAD_ALLOWLIST) == 3


def test_the_write_law_lane_and_value_noc_rows_are_read():
    read = {"e_write_per_cell", "e_verify_read_per_cell", "t_write_per_unit", "t_write_setup_per_event",
            "e_noc_value_per_byte_hop", "e_lane_op", "t_lane_op"}
    assert read <= {row[0] for row in vocabulary.ROWS}
    assert not read & set(unread_rows(vocabulary.ROWS, _pricing_source()))
