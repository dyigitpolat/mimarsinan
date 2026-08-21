"""The serial fold kernel (plan §2.2): ascending slots, adjacent occurrences,
saturating updates, per-event compare/reset, the tail bias, and the loud
emission bound. Every number here is hand-derived from the fold's definition —
a kernel that merely reproduces itself would pass nothing."""

from __future__ import annotations

import pytest
import torch

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.models.nn.lif_kernels import measurement_plane
from mimarsinan.models.spiking.serial import (
    EMISSION_COUNT_CEILING,
    EmissionBoundExceededError,
    SerialFoldUnsupportedError,
    SerialResetLawError,
    lif_serial_fold,
)

_UNBOUNDED = SomaLaw(
    firing_mode="Novena", thresholding_mode="<=",
    firing_granularity="per_event", membrane_arithmetic="unbounded",
    membrane_bits=0,
)


def _law(**kwargs) -> SomaLaw:
    fields = {
        "firing_mode": "Novena", "thresholding_mode": "<=",
        "firing_granularity": "per_event",
        "membrane_arithmetic": "unbounded", "membrane_bits": 0,
    }
    fields.update(kwargs)
    return SomaLaw(**fields)


def _saturating(bits: int, **kwargs) -> SomaLaw:
    return _law(membrane_arithmetic="saturating_unsigned", membrane_bits=bits,
                **kwargs)


def _fold(weight, events, theta, *, law=_UNBOUNDED, memb=None, hw_bias=None):
    w = torch.tensor(weight, dtype=torch.float64)
    e = torch.tensor(events, dtype=torch.float64)
    m = (torch.zeros(e.shape[0], w.shape[0], dtype=torch.float64)
         if memb is None else torch.tensor(memb, dtype=torch.float64))
    b = None if hw_bias is None else torch.tensor(hw_bias, dtype=torch.float64)
    counts = lif_serial_fold(
        m, w, e, torch.tensor(float(theta), dtype=torch.float64),
        soma_law=law, hw_bias=b,
    )
    return counts, m


def test_adjacent_occurrences_of_one_slot_change_the_count():
    """The §2.3 adjacency witness: theta=5, w=[+3,-3], e=[2,1] fires ONCE
    adjacent and ZERO times interleaved. The kernel must fold adjacent."""
    counts, memb = _fold([[3.0, -3.0]], [[2.0, 1.0]], 5.0)
    assert counts.tolist() == [[1.0]]
    # +3 -> 3 (no fire), +3 -> 6 >= 5 fires, Novena zero, -3 -> -3.
    assert memb.tolist() == [[-3.0]]

    # The same events delivered interleaved (+3, -3, +3) never reach theta.
    interleaved, _ = _fold([[3.0, -3.0, 3.0]], [[1.0, 1.0, 1.0]], 5.0)
    assert interleaved.tolist() == [[0.0]]


def test_slots_are_folded_in_ascending_canonical_order():
    """Descending delivery of the same multiplicities is a different count —
    so the ascending order is observable, not decorative."""
    ascending, _ = _fold([[4.0, -1.0]], [[1.0, 1.0]], 4.0)
    descending, _ = _fold([[-1.0, 4.0]], [[1.0, 1.0]], 4.0)
    assert ascending.tolist() == [[1.0]]
    assert descending.tolist() == [[0.0]]


def test_multi_spike_in_one_cycle_is_the_new_degree_of_freedom():
    counts, memb = _fold([[2.0]], [[7.0]], 2.0)
    assert counts.tolist() == [[7.0]]
    assert memb.tolist() == [[0.0]]


def test_novena_resets_to_hard_zero():
    _, novena = _fold([[5.0]], [[1.0]], 3.0, law=_law(firing_mode="Novena"))
    assert novena.tolist() == [[0.0]]


def test_the_subtractive_reset_is_refused_because_the_lemma_fails():
    """A subtractive reset leaves ``m >= theta`` whenever one event carries
    ``>= 2*theta``, so the zero-magnitude row of a pair FIRES and the masked
    pass stops being the serial fold. The minimal witness: theta=3, w=[10],
    e=[[2],[1]] — the true serial fold gives [[2],[1]], the masked pass gave
    [[2],[2]], and the second lane's count depended on the FIRST lane being in
    the batch. It is refused at the kernel, not approximated."""
    with pytest.raises(SerialResetLawError, match="Novena"):
        _fold([[10.0]], [[2.0], [1.0]], 3.0, law=_law(firing_mode="Default"))
    # And the law the point must declare instead runs, batch-independently.
    counts, _ = _fold([[10.0]], [[2.0], [1.0]], 3.0)
    assert counts.tolist() == [[2.0], [1.0]]
    single, _ = _fold([[10.0]], [[1.0]], 3.0)
    assert single.tolist() == [[1.0]]


def test_thresholding_mode_decides_the_exact_tie():
    inclusive, _ = _fold([[3.0]], [[1.0]], 3.0, law=_law(thresholding_mode="<="))
    strict, _ = _fold([[3.0]], [[1.0]], 3.0, law=_law(thresholding_mode="<"))
    assert inclusive.tolist() == [[1.0]]
    assert strict.tolist() == [[0.0]]


def test_saturating_unsigned_clamps_on_every_update():
    """membrane_bits=4 -> [0, 15]: the ceiling bites BETWEEN two events, and
    the floor rectifies a negative membrane (ODIN's unsigned register)."""
    law = _saturating(4)
    _, memb = _fold([[10.0, -40.0]], [[2.0, 0.0]], 100.0, law=law)
    assert memb.tolist() == [[15.0]]
    _, floored = _fold([[10.0, -40.0]], [[2.0, 1.0]], 100.0, law=law)
    assert floored.tolist() == [[0.0]]


def test_saturation_destroys_charge_the_unbounded_law_would_keep():
    events = [[3.0]]
    weight = [[7.0]]
    bounded, _ = _fold(weight, events, 20.0, law=_saturating(4))
    unbounded, _ = _fold(weight, events, 20.0)
    assert bounded.tolist() == [[0.0]]
    assert unbounded.tolist() == [[1.0]]


def test_hw_bias_folds_at_the_declared_tail_position():
    """A bias event at the TAIL sees the membrane the axons left; a head
    placement would fire on a different cycle history. theta=4, w=[-3],
    e=[1], bias=+4: tail gives 1 -> 4 >= 4 fires. Head would give 4 (fire,
    reset) then -3, i.e. the same count here — so the discriminating case
    uses a bias that only crosses AFTER the axon charge."""
    counts, memb = _fold([[3.0]], [[1.0]], 4.0, hw_bias=[2.0])
    assert counts.tolist() == [[1.0]]      # 3 then 3+2=5 >= 4
    assert memb.tolist() == [[0.0]]
    # Bias alone never crosses; axon alone never crosses; only the tail sum does.
    bias_only, _ = _fold([[0.0]], [[1.0]], 4.0, hw_bias=[2.0])
    assert bias_only.tolist() == [[0.0]]


def test_bias_slot_other_than_tail_is_refused():
    law = SomaLaw(
        firing_mode="Novena", thresholding_mode="<=",
        firing_granularity="per_event", membrane_arithmetic="unbounded",
        membrane_bits=0, bias_slot="head",
    )
    with pytest.raises(SerialFoldUnsupportedError, match="bias_slot"):
        _fold([[1.0]], [[1.0]], 1.0, law=law, hw_bias=[1.0])


def test_per_cycle_law_may_not_drive_the_serial_fold():
    with pytest.raises(SerialFoldUnsupportedError, match="firing_granularity"):
        _fold([[1.0]], [[1.0]], 1.0, law=_law(firing_granularity="per_cycle"))


def test_emission_bound_raises_and_never_clamps():
    over = float(EMISSION_COUNT_CEILING + 1)
    with pytest.raises(EmissionBoundExceededError, match="127"):
        _fold([[1.0]], [[over]], 1.0)
    at_bound, _ = _fold([[1.0]], [[float(EMISSION_COUNT_CEILING)]], 1.0)
    assert at_bound.tolist() == [[float(EMISSION_COUNT_CEILING)]]


def test_non_integer_and_negative_multiplicities_are_refused():
    with pytest.raises(SerialFoldUnsupportedError, match="multiplicit"):
        _fold([[1.0]], [[1.5]], 1.0)
    with pytest.raises(SerialFoldUnsupportedError, match="multiplicit"):
        _fold([[1.0]], [[-1.0]], 1.0)


def test_row_pair_lemma_zero_magnitude_events_are_no_ops():
    """§2.3(4): a zero-magnitude physical row is a no-op GIVEN m < theta on
    entry — which holds after every event and, at window start, under the
    V0*theta < theta precondition. Interleave a zero row at every position."""
    torch.manual_seed(0)
    weight = torch.tensor([[2.0, -1.0, 3.0], [1.0, 4.0, -2.0]], dtype=torch.float64)
    events = torch.tensor([[3.0, 1.0, 2.0], [0.0, 2.0, 5.0]], dtype=torch.float64)
    theta = torch.tensor(4.0, dtype=torch.float64)
    for init in (0.0, 3.999):
        base_m = torch.full((2, 2), init, dtype=torch.float64)
        base = lif_serial_fold(base_m, weight, events, theta, soma_law=_UNBOUNDED)
        # Expand every logical slot into (excitatory, inhibitory) rows.
        pair_w = torch.zeros(2, 6, dtype=torch.float64)
        pair_e = torch.zeros(2, 6, dtype=torch.float64)
        for slot in range(3):
            sign_row = 0 if float(weight[0, slot]) >= 0 else 1
            pair_w[:, 2 * slot + sign_row] = weight[:, slot]
            pair_e[:, 2 * slot + sign_row] = events[:, slot]
            pair_e[:, 2 * slot + (1 - sign_row)] = events[:, slot]
        pair_m = torch.full((2, 2), init, dtype=torch.float64)
        paired = lif_serial_fold(pair_m, pair_w, pair_e, theta, soma_law=_UNBOUNDED)
        assert torch.equal(base, paired), f"row-pair lemma broke at V0*theta={init}"


def test_row_pair_lemma_precondition_is_load_bearing():
    """At ``m0 == theta`` the zero-magnitude row FIRES, so inserting it
    CHANGES the count — the exact reason §1.2 constrains ``V0*theta <
    theta``. Below theta the same insertion is invisible."""
    theta = torch.tensor(4.0, dtype=torch.float64)
    paired_w = torch.tensor([[0.0, 4.0]], dtype=torch.float64)
    paired_e = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    bare_w = torch.tensor([[4.0]], dtype=torch.float64)
    bare_e = torch.tensor([[1.0]], dtype=torch.float64)

    def counts(weight, events, init):
        return lif_serial_fold(
            torch.full((1, 1), init, dtype=torch.float64), weight, events,
            theta, soma_law=_UNBOUNDED).tolist()

    # Below theta the zero row is a no-op (the lemma).
    assert counts(paired_w, paired_e, 3.0) == counts(bare_w, bare_e, 3.0) == [[1.0]]
    # At theta it fires on its own and the count doubles.
    assert counts(paired_w, paired_e, 4.0) == [[2.0]]
    assert counts(bare_w, bare_e, 4.0) == [[1.0]]


def test_lattice_snap_is_the_integer_lsb_and_only_inside_the_plane():
    law = _saturating(8)
    assert law.membrane_lattice_quantum == 1.0
    assert law.membrane_bounds == (0.0, 255.0)
    weight = torch.tensor([[1.6]], dtype=torch.float64)
    events = torch.tensor([[1.0]], dtype=torch.float64)
    theta = torch.tensor(2.0, dtype=torch.float64)
    outside = lif_serial_fold(
        torch.zeros(1, 1, dtype=torch.float64), weight, events, theta,
        soma_law=law)
    with measurement_plane():
        inside = lif_serial_fold(
            torch.zeros(1, 1, dtype=torch.float64), weight, events, theta,
            soma_law=law)
    assert outside.tolist() == [[0.0]]   # 1.6 < 2
    assert inside.tolist() == [[1.0]]    # snaps to 2 >= 2


def test_unbounded_law_declares_no_lattice_and_no_bounds():
    assert _UNBOUNDED.membrane_lattice_quantum is None
    assert _UNBOUNDED.membrane_bounds is None


def test_grouped_lead_dimension_equals_the_per_core_fold():
    """The packed layout (G, N, A) must fold exactly as G separate cores."""
    torch.manual_seed(5)
    weight = torch.randint(-3, 4, (3, 4, 5)).to(torch.float64)
    events = torch.randint(0, 3, (2, 3, 5)).to(torch.float64)
    theta = torch.tensor([[3.0] * 4] * 3, dtype=torch.float64)
    grouped_m = torch.zeros(2, 3, 4, dtype=torch.float64)
    grouped = lif_serial_fold(grouped_m, weight, events, theta, soma_law=_UNBOUNDED)
    for g in range(3):
        flat_m = torch.zeros(2, 4, dtype=torch.float64)
        flat = lif_serial_fold(
            flat_m, weight[g], events[:, g, :], theta[g], soma_law=_UNBOUNDED)
        assert torch.equal(grouped[:, g, :], flat)
        assert torch.equal(grouped_m[:, g, :], flat_m)
