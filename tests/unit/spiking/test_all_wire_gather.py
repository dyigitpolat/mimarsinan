"""[U-SC3 A-2] A gather made all-wire by an armed absolute entry, read by a core under streamed LIF: count-exact.

``nested_gather_core_read`` (``host_gather_forms``) concatenates a host-side gather (the absolute stem beside a host op
over a core output) with a core output, inside a neural segment, and a core reads the outer concatenation. At input
data scale 1 the stem stays absolute, the outer gather is MIXED, and the NF walk gathered each source's train per cycle.
At any other input data scale U-SC2's seam arms the stem at its boundary gauge (``arm_absolute_core_entries``), so both
gathers are wire; the walk's per-cycle gather rule then needed every source to carry a train already, the host-side
inner gather carried none, and the outer gather was re-encoded whole from its rate (``uniform_spike_train`` of the
18-wide rate): the core output's spike timing was lost. Under streamed LIF (verbatim transfer) the deployed program
delivers that core output's train as emitted, so the twins differed (47 of 48 streamed cases at input data scale 2,
684 counts, ``phase_c_r/g1_tlif/fixtures/summary.json``); windowed LIF re-times every hop, so it agreed.

Now a layout gather inside a neural segment moves every source's train per cycle, each source delivered by its own
rule (a core's train as emitted, a host value through its boundary re-encode at its own currency), MIXED or wire, as
the deployed program feeds each source to the reading core. Every count is compared at atol 0 (no tolerance).

[U-SC3 targeted pass] The layout gather carries its event train by the same per-source rule wherever a fold reads events:
always under the per-event soma law (the event-serial fold), and inside the measurement plane under the per-cycle law
(the cycle fold), MIXED or wire alike. The frozen unit kept a wire gather's events only under the per-cycle law, so under
the per-event law the core reading an all-wire gather refused by name (``SerialFoldUnsupportedError``) where R had read
it count-exact (``absolute_concat`` at input data scale 2: review.md, finding 1).
"""

from __future__ import annotations

import pytest
import torch

import unit.spiking.host_gather_forms as hgf
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.mapping.support.schedule.pass_cut import VERBATIM
from mimarsinan.models.nn.lif_kernels import measurement_plane
from mimarsinan.spiking.scale_aware_boundaries import propagate_boundary_input_scales
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver
from mimarsinan.spiking.wire_gauge_seam import establish_deployed_wire_gauge

from unit.spiking.host_gather_forms import MIXED_FORMS, SHAKEDOWN_FORMS, build, compare, samples

SEEDS = range(8)
#: The survey's encoding thetas (phase B's): the first is the diagnosis's; the other two run in the slow profile.
THETAS = (2.185, 0.731, 1.0)


def _seam(input_data_scale: float):
    """The deployed seam first, then the per-source and boundary scales at the same input scale (the survey's
    ``ids1_seam`` and ``ids2_seam`` variants)."""

    def establish(flow):
        establish_deployed_wire_gauge(flow, input_data_scale=input_data_scale)
        compute_per_source_scales(flow.get_mapper_repr(), arm_wire_value_ops=True)
        propagate_boundary_input_scales(flow, input_data_scale=input_data_scale)

    return establish


def _differing_counts(form: str, input_data_scale: float, semantics: str, wq: bool, theta: float) -> dict:
    """``{seed: differing counts}`` for every seed with any (32 samples each, the survey's)."""
    differing = {}
    for seed in SEEDS:
        program = build(form, seed, theta, semantics, wq, establish=_seam(input_data_scale))
        mism, total, _ = compare(program, semantics, n=32, seed=seed)
        assert total > 0
        if sum(mism.values()):
            differing[seed] = mism
    return differing


@pytest.mark.parametrize("wq", (False, True), ids=("float", "wq4"))
@pytest.mark.parametrize("semantics", ("streamed_lif", "windowed_lif"))
@pytest.mark.parametrize("input_data_scale", (1.0, 2.0))
def test_a_nested_gather_read_by_a_core_is_count_exact_on_eight_seeds(input_data_scale, semantics, wq):
    assert _differing_counts("nested_gather_core_read", input_data_scale, semantics, wq, THETAS[0]) == {}


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("theta", THETAS[1:])
@pytest.mark.parametrize("wq", (False, True), ids=("float", "wq4"))
@pytest.mark.parametrize("semantics", ("streamed_lif", "windowed_lif"))
@pytest.mark.parametrize("input_data_scale", (1.0, 2.0))
def test_a_nested_gather_read_by_a_core_is_count_exact_at_the_other_survey_thetas(input_data_scale, semantics, wq,
                                                                                   theta):
    assert _differing_counts("nested_gather_core_read", input_data_scale, semantics, wq, theta) == {}


@pytest.mark.parametrize("semantics", ("streamed_lif", "windowed_lif"))
@pytest.mark.parametrize("form", ("nested_gather_host_read", "absolute_concat", "absolute_concat_host"))
def test_the_other_mixed_forms_stay_count_exact_at_both_input_scales(form, semantics):
    """The forms the layout rule also reaches (a gather made all-wire read by a host op; the mixed concatenation read
    by a core and by a host op) stay exact at both input scales."""
    for input_data_scale in (1.0, 2.0):
        for wq in (False, True):
            assert _differing_counts(form, input_data_scale, semantics, wq, THETAS[0]) == {}, (input_data_scale, wq)


#: The per-event soma law (ODIN's granularity) at its two arithmetic points, each with the weights it reads count-exact
#: on every fixture form R admits: the unbounded accumulator with float weights, and ODIN's 8-bit saturating unsigned
#: register with 4-bit weights (the integer membrane lattice). The crossed pairs (the unbounded accumulator with 4-bit
#: weights, the 8-bit register with float weights) differ on R at forms without any gather (``pool_join_max``): outside
#: this unit (``final_units/usc3/tp/out/per_event_r``, ``per_event_r_odin``).
PER_EVENT_POINTS = {
    "unbounded_float": ({"firing_granularity": "per_event"}, False),
    "sat8_wq4": ({"firing_granularity": "per_event", "membrane_arithmetic": "saturating_unsigned", "membrane_bits": 8},
                 True),
}


@pytest.fixture
def per_event(monkeypatch):
    """``point -> (semantics name, wq)``: each per-event point as a streamed semantics of ``host_gather_forms`` (verbatim
    transfer, the only discipline the law admits), the deployed flow and the NF policy carrying the same law."""
    names = {}
    for point, (axes, wq) in PER_EVENT_POINTS.items():
        law = SomaLaw.resolve(axes)
        name = f"streamed_lif_per_event_{point}"
        monkeypatch.setitem(hgf.SEMANTICS, name, (
            False,
            dict(spiking_mode="lif", cycle_accurate_lif_forward=True, pass_transfer=VERBATIM, soma_law=law,
                 firing_mode=law.firing_mode),
            lambda law=law: LifSegmentPolicy(soma_law=law),
        ))
        names[point] = (name, wq)
    return names


@pytest.mark.parametrize("point", tuple(PER_EVENT_POINTS))
@pytest.mark.parametrize("form", tuple(MIXED_FORMS))
def test_a_layout_gather_read_under_the_per_event_law_is_count_exact(per_event, form, point):
    """Under the per-event law a layout gather, MIXED or made all-wire by the armed absolute stem, carries each source's
    own events per cycle, and every core folds them count-exact at both input data scales on eight seeds."""
    semantics, wq = per_event[point]
    for input_data_scale in (1.0, 2.0):
        assert _differing_counts(form, input_data_scale, semantics, wq, THETAS[0]) == {}, input_data_scale


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("point", tuple(PER_EVENT_POINTS))
@pytest.mark.parametrize("form", SHAKEDOWN_FORMS + ("densenet_bc",))
def test_every_fixture_form_is_count_exact_under_the_per_event_law(per_event, form, point):
    """The shakedown forms and DenseNet-BC under the per-event law: a core read through a layout node (a concatenation, a
    pad, a reshape) folds that node's events; every count equal at both input data scales on eight seeds."""
    semantics, wq = per_event[point]
    for input_data_scale in (1.0, 2.0):
        assert _differing_counts(form, input_data_scale, semantics, wq, THETAS[0]) == {}, input_data_scale


def test_the_per_event_walk_outside_the_measurement_plane_reads_an_all_wire_gather(per_event):
    """The walk outside the measurement plane (the tuning plane: training telemetry, the AQ D-hat) folds the all-wire
    gather's events as the measurement plane does: with float weights its output is bit-equal to the plane's."""
    semantics, wq = per_event["unbounded_float"]
    policy = hgf.SEMANTICS[semantics][2]
    for seed in SEEDS:
        program = build("absolute_concat", seed, THETAS[0], semantics, wq, establish=_seam(2.0))
        x = samples(program, 32, seed)
        with torch.no_grad():
            tuning = SegmentForwardDriver(program.repr_, hgf.T, policy())(x)
            with measurement_plane():
                plane = SegmentForwardDriver(program.repr_, hgf.T, policy())(x)
        assert torch.equal(tuning, plane), seed
