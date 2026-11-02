"""[U-DOOR] The exactness door: every spiking pairing the declaration door admits reads NF/SCM count-exact.

A pairing is a soma point (firing granularity x membrane arithmetic) beside the weight representation (float, or the
Weight Quantization step's integer grid) and a LIF semantics (streamed, or windowed: the synchronized variant's re-timed
hops), with or without the phase-dithered encode. Each pairing is declared through the run's derivation
(``build_flat_pipeline_config``, the door both the CLI and the wizard take). Either the door refuses it by name (a
``ToolchainRefusal`` whose message names the pairing, the reason and the remedy), or the twins, built at the door's
resolved point (its ``SomaLaw``, its ``weight_quantization``, its dither), read every core perceptron's window counts
equal at atol 0 on the fixture forms of ``host_gather_forms`` (a gather-free form and the gather forms, 8 seeds, input
data scales 1 and 2), the deployed flow read both outside and inside the measurement plane (a deployed arithmetic does
not depend on where it is read). Never a differing count.

Classes this closes (probes and denominators: ``final_units/udoor/report.md``):

- the per-event fold with quantized weights on the unbounded accumulator decided threshold ties by float dust: the NF
  slot folds ``W * theta_int`` while the deployed core folds the integer weights, and the fold snapped only onto a
  saturating register's LSB. It now snaps onto the weight-quantized program's integer chip lattice too, the one the
  per-cycle policy and the NF's lattice node snap onto (made exact);
- an unsigned saturating register with float weights: the register counts whole LSBs, which float weights do not have;
  the per-event fold snapped the membrane to whole thresholds only inside the measurement plane, so the deployed count
  depended on where it was read (refused by name at the door);
- an unsigned saturating register under the per-cycle law: the NF twin integrates the unbounded accumulator, the
  deployed register clamps (the floor at zero is crossed by any negative net charge), every fixture case differs
  (refused by name at the door);
- the phase-dithered encode keyed by the last-dim index of the value as the NF walk holds it, while the deployed program
  keys each stage-input position by the channel of its IR node's output at unbounded geometry (``phase_channels``: a
  core node's output channel, a host node's output index): a 4-D host value, a re-timed convolution hop, a host-side
  gather of several producers, a host add over a whole token value and t2_04's ViT rotated differently. A rule by the
  value's shape (the last-dim index, or the flat index of a host value) cannot be the deployed one: a token-wise
  encoding is one host node per token (keyed 0..5 each), a host add over the same token value one node (keyed 0..23).
  Now every element is keyed from the IR the program is built from (its IR node and output index through the program's
  channel table), so both twins read one table (made exact; ``TOKEN_FORMS``).
"""

from __future__ import annotations

import pytest

import unit.spiking.host_gather_forms as hgf
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.common.refusal.toolchain_refusal import is_toolchain_refusal, refusal_in_chain
from mimarsinan.config_schema.runtime import build_flat_pipeline_config
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE, VERBATIM
from mimarsinan.models.nn.lif_kernels import MembraneRailTouchedError, measurement_plane
from mimarsinan.models.spiking.serial import EmissionBoundExceededError, SerialFoldUnsupportedError
from mimarsinan.spiking.scale_aware_boundaries import propagate_boundary_input_scales
from mimarsinan.spiking.segment_forward import LifSegmentPolicy
from mimarsinan.spiking.segment_currency import MixedGatherDitherError
from mimarsinan.spiking.wire_gauge_seam import establish_deployed_wire_gauge

SEEDS = range(8)
INPUT_DATA_SCALES = (1.0, 2.0)
THETA_ENC = 2.185
_BIASLESS_CORES = [{"max_axons": 128, "max_neurons": 256, "count": 4, "has_bias": False}]

#: ``name -> (spiking_variant, firing_granularity or None, membrane_bits, membrane_signed, weight_bits or None)``.
PAIRINGS = {
    # The native pairings the templates and the targets declare.
    "per_cycle.unbounded.float": ("streamed", None, 0, False, None),
    "per_cycle.unbounded.wq4": ("streamed", None, 0, False, 4),
    "per_event.unbounded.float": ("streamed", "per_event", 0, False, None),
    "per_event.sat8.wq4": ("streamed", "per_event", 8, False, 4),
    "windowed.unbounded.float": ("synchronized", None, 0, False, None),
    "windowed.unbounded.wq4": ("synchronized", None, 0, False, 4),
    # The crossed pairings.
    "per_event.unbounded.wq4": ("streamed", "per_event", 0, False, 4),
    "per_event.sat8.float": ("streamed", "per_event", 8, False, None),
    "per_cycle.sat8.float": ("streamed", None, 8, False, None),
    "windowed.sat8.float": ("synchronized", None, 8, False, None),
    # The register under the per-cycle law, with its weight quantization.
    "per_cycle.sat8.wq4": ("streamed", None, 8, False, 4),
    "windowed.sat8.wq4": ("synchronized", None, 8, False, 4),
    # The two's-complement register (asserts its rails unreachable) under the per-cycle law.
    "per_cycle.ssat8.float": ("streamed", None, 8, True, None),
    "per_cycle.ssat8.wq4": ("streamed", None, 8, True, 4),
}
#: What the door does with each pairing (the classes of the module docstring).
REFUSED = {"per_event.sat8.float", "per_cycle.sat8.float", "windowed.sat8.float", "per_cycle.sat8.wq4",
           "windowed.sat8.wq4"}
ADMITTED = tuple(name for name in PAIRINGS if name not in REFUSED)

GATHER_FREE_FORM = "projection"
GATHER_FORMS = ("dense_concat", "absolute_concat", "nested_gather_core_read")
OTHER_FORMS = tuple(f for f in hgf.SHAKEDOWN_FORMS + tuple(hgf.MIXED_FORMS) + ("densenet_bc",)
                    if f not in (GATHER_FREE_FORM,) + GATHER_FORMS)


def declare(name: str, *, dither: bool = False) -> dict:
    """The pairing through the run's derivation: the resolved flat config, or the door's refusal raised."""
    variant, granularity, bits, signed, weight_bits = PAIRINGS[name]
    dp = {"spiking_family": "lif", "spiking_variant": variant}
    if granularity is not None:
        dp["firing_granularity"] = granularity
    if dither:
        dp["spike_phase_dither"] = True
    pc: dict = {"cores": _BIASLESS_CORES}
    if bits:
        pc["membrane_bits"] = bits
    if signed:
        pc["membrane_signed"] = True
    if weight_bits:
        pc["weight_bits"] = weight_bits
    return build_flat_pipeline_config(dp, pc, pipeline_mode="phased" if weight_bits else "vanilla")


def _seam(input_data_scale: float):
    def establish(flow):
        establish_deployed_wire_gauge(flow, input_data_scale=input_data_scale)
        compute_per_source_scales(flow.get_mapper_repr(), arm_wire_value_ops=True)
        propagate_boundary_input_scales(flow, input_data_scale=input_data_scale)

    return establish


@pytest.fixture
def twins(monkeypatch):
    """``resolved config -> (semantics name, wq)``: the door's resolved point as a ``host_gather_forms`` semantics, the
    deployed flow and the NF policy carrying the same soma law, the same re-timing and the same dither."""

    def register(resolved: dict):
        law = SomaLaw.resolve(resolved)
        windowed = bool(resolved.get("lif_per_hop_retiming"))
        dither = bool(resolved.get("spike_phase_dither"))
        name = f"door_{law.point_tag() or 'default'}_{'windowed' if windowed else 'streamed'}_{int(dither)}"
        monkeypatch.setitem(hgf.SEMANTICS, name, (
            windowed,
            dict(spiking_mode="lif", cycle_accurate_lif_forward=True, pass_transfer=COLLAPSE if windowed else VERBATIM,
                 soma_law=law, firing_mode=law.firing_mode, phase_dither=dither),
            lambda: LifSegmentPolicy(retime=windowed, soma_law=law, phase_dither=dither),
        ))
        return name, bool(resolved["weight_quantization"])

    return register


def _case_differing(form: str, semantics: str, wq: bool, input_data_scale: float, seed: int,
                    wq_bits: int = hgf.WQ_BITS) -> dict:
    """``{plane: differing counts}`` of one case, the deployed flow read outside and inside the measurement plane."""
    program = hgf.build(form, seed, THETA_ENC, semantics, wq, establish=_seam(input_data_scale), wq_bits=wq_bits)
    x = hgf.samples(program, 32, seed)
    _out, nf = hgf.nf_window_counts(program, x, semantics)
    differing = {}
    for plane in ("outside", "inside"):
        if plane == "inside":
            with measurement_plane():
                _o, scm = hgf.scm_window_counts(program, x, semantics)
        else:
            _o, scm = hgf.scm_window_counts(program, x, semantics)
        mism = sum(hgf.count_mismatches(nf, scm).values())
        if mism:
            differing[plane] = mism
    return differing


def _exact_or_refused_by_name(form: str, semantics: str, wq: bool, expected: tuple = (),
                              wq_bits: int = hgf.WQ_BITS) -> None:
    """Every case (input data scale x seed) reads 0 differing counts in both planes, or refuses by name with a refusal
    the pairing declares (``expected``: a two's-complement register's rail, ``MembraneRailTouchedError``; a MIXED gather
    under the dithered encode, ``MixedGatherDitherError``). Any other exception fails the test: an anonymous one is a
    loud failure, and an undeclared refusal would let a test pass on refusals alone."""
    differing = {}
    for input_data_scale in INPUT_DATA_SCALES:
        for seed in SEEDS:
            try:
                case = _case_differing(form, semantics, wq, input_data_scale, seed, wq_bits)
            except Exception as exc:  # noqa: BLE001 - classified below
                named = refusal_in_chain(exc)
                assert named is not None, f"{type(exc).__name__} is not a refusal by name: {exc}"
                assert isinstance(named, expected), f"{type(named).__name__}: undeclared for this pairing: {named}"
                continue
            if case:
                differing[(input_data_scale, seed)] = case
    assert differing == {}


def expected_refusals(name: str, *, dither: bool = False) -> tuple:
    """The refusals by name a pairing's programs may meet at run time: the two's-complement register refuses a membrane
    that reaches its rail (its law asserts the rails unreachable); the event-serial law refuses a cycle count above
    its chip's count currency (``EmissionBoundExceededError``); the dithered encode refuses a MIXED gather."""
    _variant, granularity, _bits, signed, _wb = PAIRINGS[name]
    return (((MembraneRailTouchedError,) if signed else ())
            + ((EmissionBoundExceededError,) if granularity == "per_event" else ())
            + ((MixedGatherDitherError,) if dither else ()))


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_the_door_refuses_a_pairing_the_twins_cannot_hold_exactly_by_name(name):
    with pytest.raises(Exception) as caught:
        declare(name)
    assert is_toolchain_refusal(caught.value), type(caught.value).__name__
    message = str(caught.value)
    assert "membrane_bits=8" in message and "saturating_unsigned" in message, message
    assert "Declare" in message or "declare" in message, message


@pytest.mark.parametrize("name", sorted(ADMITTED))
def test_the_door_admits_the_native_and_the_exact_pairings(name):
    resolved = declare(name)
    variant, granularity, bits, signed, weight_bits = PAIRINGS[name]
    law = SomaLaw.resolve(resolved)
    assert law.membrane_bits == bits
    assert law.is_per_event == (granularity == "per_event")
    assert bool(resolved["weight_quantization"]) == bool(weight_bits)


#: Each fast case builds and reads 16 programs (8 seeds x 2 input data scales) on both twins, the deployed flow twice:
#: about 30 s alone, so its budget holds under a loaded host.
_FAST = pytest.mark.timeout(240)


@_FAST
@pytest.mark.parametrize("form", (GATHER_FREE_FORM,) + GATHER_FORMS)
@pytest.mark.parametrize("name", sorted(ADMITTED))
def test_an_admitted_pairing_is_count_exact_on_eight_seeds_at_both_input_scales(twins, name, form):
    semantics, wq = twins(declare(name))
    _exact_or_refused_by_name(form, semantics, wq, expected_refusals(name))


@pytest.mark.slow
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("form", OTHER_FORMS)
@pytest.mark.parametrize("name", sorted(ADMITTED))
def test_an_admitted_pairing_is_count_exact_on_every_fixture_form(twins, name, form):
    semantics, wq = twins(declare(name))
    _exact_or_refused_by_name(form, semantics, wq, expected_refusals(name))


#: The pairings the dithered encode is declared beside (a LIF declaration): the per-cycle law on both semantics and
#: weight representations, and the per-event law's two native pairings.
DITHER_PAIRINGS = ("per_cycle.unbounded.float", "per_cycle.unbounded.wq4", "windowed.unbounded.float",
                   "windowed.unbounded.wq4", "per_event.unbounded.float", "per_event.sat8.wq4")


#: The dithered forms of the fast profile: an all-wire gather read by a core (``nested_gather_core_read`` at input data
#: scale 2), the mixed concatenation (refused by name at input data scale 1: ``MixedGatherDitherError``), a 4-D
#: encoding read by a convolution (``pool_join_max``) and convolution cores reading convolution cores (``chain_2``: a
#: re-timed hop keyed by output channel); under the event-serial law the convolution forms run in the slow profile.
DITHER_FORMS = ("nested_gather_core_read", "absolute_concat", "pool_join_max", "chain_2")
CONVOLUTION_FORMS = ("pool_join_max", "chain_2")
_SLOW = (pytest.mark.slow, pytest.mark.timeout(3600))


def _dither_cases():
    for name in DITHER_PAIRINGS:
        for form in DITHER_FORMS:
            slow = name.startswith("per_event") and form in CONVOLUTION_FORMS
            yield pytest.param(name, form, marks=_SLOW if slow else (_FAST,), id=f"{name}-{form}")


@pytest.mark.parametrize(("name", "form"), tuple(_dither_cases()))
def test_a_dithered_encode_is_count_exact_or_refused_by_name(twins, name, form):
    semantics, wq = twins(declare(name, dither=True))
    _exact_or_refused_by_name(form, semantics, wq, expected_refusals(name, dither=True))


def _slow_dither_cases():
    """Every other fixture form under each dithered pairing; U-SC3's memory forms (wide encodings, chains of
    convolution cores) under the per-cycle law, where a re-timed convolution hop re-encodes a core's output (under the
    event-serial law a core reads its producer's events as emitted, and ``chain_2`` covers the chain)."""
    forms = tuple(f for f in hgf.SHAKEDOWN_FORMS + tuple(hgf.MIXED_FORMS) + ("densenet_bc",) if f not in DITHER_FORMS)
    for name in DITHER_PAIRINGS:
        memory = () if name.startswith("per_event") else tuple(f for f in hgf.MEMORY_FORMS if f not in DITHER_FORMS)
        for form in forms + memory:
            yield pytest.param(name, form, id=f"{name}-{form}")


@pytest.mark.slow
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("name", "form"), tuple(_slow_dither_cases()))
def test_a_dithered_encode_is_count_exact_or_refused_by_name_on_every_fixture_form(twins, name, form):
    semantics, wq = twins(declare(name, dither=True))
    _exact_or_refused_by_name(form, semantics, wq, expected_refusals(name, dither=True))


#: [U-DOOR targeted pass] The token forms (``host_gather_forms.TOKEN_FORMS``): one mapper mapped to several IR nodes per
#: sample. A host op over per-token nodes (the token-wise encoding, keyed 0..5 per token) and a host op over the whole
#: token value (an add, keyed 0..23), each read by token-wise cores, and t2_04's ViT (its MLP's first Linear on chip per
#: token reading a host LayerNorm of the whole token value): under the per-cycle law, on both semantics and weight
#: representations. The event-serial law's NF twin folds one core of one position only, so a token-wise hop refuses by
#: name there (``SerialFoldUnsupportedError``, a boundary: never a count).
TOKEN_DITHER_FORMS = tuple(hgf.TOKEN_FORMS)


def _token_dither_cases():
    for name in DITHER_PAIRINGS:
        for form in TOKEN_DITHER_FORMS:
            yield pytest.param(name, form, marks=(_FAST,), id=f"{name}-{form}")


@pytest.mark.parametrize(("name", "form"), tuple(_token_dither_cases()))
def test_a_dithered_encode_of_a_token_form_is_count_exact_or_refused_by_name(twins, name, form):
    semantics, wq = twins(declare(name, dither=True))
    serial = (SerialFoldUnsupportedError,) if name.startswith("per_event") else ()
    _exact_or_refused_by_name(form, semantics, wq, expected_refusals(name, dither=True) + serial)


def t2_04_declaration() -> dict:
    """t2_04's spiking declaration (``templates/tier_2/t2_04_lifsync_vit_wq_s32_sched_offload_pruned.json``): the
    synchronized LIF variant (re-timed hops), 8-bit weight quantization, the phase-dithered encode."""
    dp = {"spiking_family": "lif", "spiking_variant": "synchronized", "spike_phase_dither": True}
    return build_flat_pipeline_config(dp, {"cores": _BIASLESS_CORES, "weight_bits": 8}, pipeline_mode="phased")


@_FAST
def test_a_t2_04_shaped_program_is_count_exact_under_its_dithered_encode(twins):
    """Guard: the one template that declares the dithered encode, t2_04, at its declared point on its network family
    (``vit_tiny``), 8 seeds at both input data scales: exact (no refusal is declared for it)."""
    resolved = t2_04_declaration()
    assert resolved["spike_phase_dither"] and resolved["weight_quantization"] and resolved["lif_per_hop_retiming"]
    semantics, wq = twins(resolved)
    _exact_or_refused_by_name("vit_tiny", semantics, wq, (), wq_bits=int(resolved["weight_bits"]))


def test_the_mixed_gather_under_dither_still_refuses_by_name(twins):
    """Guard: the MIXED gather (the absolute stem beside a core output at input data scale 1) keeps its refusal."""
    semantics, wq = twins(declare("per_cycle.unbounded.float", dither=True))
    program = hgf.build("absolute_concat", 0, THETA_ENC, semantics, wq, establish=_seam(1.0))
    with pytest.raises(MixedGatherDitherError):
        hgf.nf_window_counts(program, hgf.samples(program, 32, 0), semantics)
