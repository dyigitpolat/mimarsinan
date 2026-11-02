"""[U-IN A-1, F-IN] The input root is an absolute producer at the boundary gauge in both twins.

A core on chip that reads the raw input (``encoding_layer_placement: offload``) at an input data scale ``s`` other than
one was fed in two currencies: the NF encoded ``clamp(x / s)`` and multiplied ``s`` back (the boundary table's gauge,
``normalize_boundary_value``), while the deployed program (SCM and HCM alike) encoded ``clamp(x)`` at the unit gauge of
the raw buffer and folded one (``InputMapper.propagate_source_scale``; the divisor table listed compute stages only), so
every input above one saturated on chip (d_tok: token_skip 6,083 of 13,824 counts, pool_join_max 36,659 of 83,456, the
pipeline's streamed gate ``NfScmParityError`` 8 of 8). U-SC2 F-2 made this one transcode for an input-rooted host chain;
the input root itself is now the same rule: the slice is divided by the producer's boundary-table gauge in the divisor
table every executor reads (``boundary_normalization_scales``), clamped and encoded, and the same gauge is folded into
the consuming core (the source walk's fold of the input root, stamped by the wire-gauge seam, carried by the IR and the
program), while a host op that reads the raw buffer keeps decoding it at one.

Denominators: the forms whose first core reads the raw input under ``offload`` (every form of ``FORMS``, the four of
``MEMORY_FORMS`` and the two of ``TOKEN_FORMS`` whose first perceptron reads the input, plus ``input_core_host``: a core
and an armed host op on the input at once, and ``input_concat``: the raw input concatenated with a core's output, read by
a core), input data scales 1, 2 and 2.024, both LIF semantics, float and 4-bit, 8
seeds. The forms whose input only a host op reads under ``offload`` (``densenet_bc``, ``vit_tiny``,
``wide_concat_host_read``) are the controls: no neural stage reads the input, so the divisor table carries no input entry.
``dense_concat`` under ``offload`` reads its stem core again two hops on inside one streamed segment (V6's unequal-depth
fan-in): it is built with the declared exact remedy, depth-balancing relays, as every form here is (a no-op elsewhere);
without them it differs at every input scale, 1 included, by a mechanism this unit does not touch (the unit record's
``probes/dense_concat_*``).
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

import unit.spiking.host_gather_forms as hgf
from mimarsinan.config_schema.defaults import get_default_deployment_parameters, get_default_platform_constraints
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.pipelining.core.nf_scm_parity import assert_streamed_nf_scm_exact_or_raise
from mimarsinan.spiking.scale_aware_boundaries import propagate_boundary_input_scales
from mimarsinan.spiking.segment_boundary import boundary_normalization_scales
from mimarsinan.spiking.segment_forward import SegmentForwardDriver
from mimarsinan.spiking.wire_gauge_seam import establish_deployed_wire_gauge

THETA_ENC = 2.185
SEEDS = range(8)
INPUT_DATA_SCALES = (1.0, 2.0, 2.024)
SEMANTICS = ("streamed_lif", "windowed_lif")
#: Under ``offload``, the forms whose first core reads the raw input (``probes/which_forms.out`` of the unit record).
CORE_READ_FORMS = (tuple(hgf.FORMS) + ("wide_encoding_read", "wide_core_output", "chain_2", "chain_6")
                   + ("token_skip", "token_residual") + tuple(hgf.INPUT_FORMS))
#: Under ``offload``, the forms whose raw input only a host op reads (the controls).
HOST_READ_FORMS = ("densenet_bc", "vit_tiny", "wide_concat_host_read")
RAW_INPUT = -2
#: The grids below run every case (minutes per form); the default profile reads one case per form and semantics.
#: The default profile's forms: a fully connected stem, a convolution stem, a token-wise stem, a core and a host op.
FAST_FORMS = ("encoding_skip", "pool_join_max", "token_skip", "input_core_host")


def _seam(input_data_scale: float):
    """The run's establishment: the wire-gauge seam, then the two Soft Core Mapping calls at the run's input scale."""

    def establish(flow):
        establish_deployed_wire_gauge(flow, input_data_scale=input_data_scale)
        compute_per_source_scales(flow.get_mapper_repr(), arm_wire_value_ops=True)
        propagate_boundary_input_scales(flow, input_data_scale=input_data_scale)

    return establish


def _offload(form: str, seed: int, semantics: str, wq: bool, input_data_scale: float) -> hgf.Program:
    """The form with its encoding layer on chip; depth-balancing relays where a streamed segment has an unequal-depth
    fan-in (``dense_concat``: the stem core read by a core two hops on; the declared exact remedy, a no-op elsewhere)."""
    return hgf.build(form, seed, THETA_ENC, semantics, wq, establish=_seam(input_data_scale), placement="offload",
                     relays=True)


def _cores_reading_input(program: hgf.Program) -> int:
    return sum(1 for n in program.ir.nodes if isinstance(n, NeuralCore)
               and any(int(s.node_id) == RAW_INPUT for s in n.input_sources.flatten()))


def _neural_stage_reads_input(hybrid) -> bool:
    return any(st.kind == "neural" and any(int(sl.node_id) == RAW_INPUT for sl in st.input_map)
               for st in hybrid.stages)


class _NF(nn.Module):
    """The NF as the pipeline holds it after LIF Adaptation: its forward is the segment walk."""

    def __init__(self, program: hgf.Program, semantics: str):
        super().__init__()
        self.repr_ = program.repr_
        self._perceptrons = nn.ModuleList(list(program.perceptrons))
        self._policy = hgf.SEMANTICS[semantics][2]

    def get_perceptrons(self):
        return list(self._perceptrons)

    def forward(self, x):
        return SegmentForwardDriver(self.repr_, hgf.T, self._policy())(x)


def _streamed_stub(program: hgf.Program):
    cfg = get_default_deployment_parameters()
    cfg.update(get_default_platform_constraints())
    cfg.update({"spiking_family": "lif", "spiking_variant": "streamed", "simulation_steps": hgf.T,
                "input_shape": program.shape, "device": "cpu", "weight_quantization": bool(program.wq)})
    return SimpleNamespace(config=cfg)


def _case(form: str, semantics: str, wq: bool, input_data_scale: float, seed: int) -> dict:
    """One case's readings: differing window counts, the output on the count grid, and the non-vacuity facts."""
    program = _offload(form, seed, semantics, wq, input_data_scale)
    x = hgf.samples(program, 32, seed)
    nf_out, nf = hgf.nf_window_counts(program, x, semantics)
    scm_out, scm = hgf.scm_window_counts(program, x, semantics)
    ref, dep = hgf.output_counts(program, nf_out, scm_out, 32)
    return {"differ": int(sum(hgf.count_mismatches(nf, scm).values())),
            "out_off_grid": float((torch.round(ref) - dep).abs().max()), "out_diff": float((ref - dep).abs().max()),
            "readers": _cores_reading_input(program), "above_one": int((x > 1.0).sum())}


def _differing(form: str, semantics: str, wq: bool, scales, seeds) -> dict:
    differing = {}
    for input_data_scale in scales:
        for seed in seeds:
            case = _case(form, semantics, wq, input_data_scale, seed)
            assert case["readers"] > 0, f"{form}: no core reads the raw input under offload (vacuous case)"
            assert case["above_one"] > 0, "no input above one: the unit gauge would not saturate (vacuous case)"
            if case["differ"] or case["out_off_grid"] or case["out_diff"] >= 1e-3:
                differing[(input_data_scale, seed)] = case
    return differing


@pytest.mark.parametrize("wq", (False, True), ids=("float", "wq4"))
@pytest.mark.parametrize("semantics", SEMANTICS)
@pytest.mark.parametrize("form", FAST_FORMS)
def test_a_core_reading_the_raw_input_reads_it_at_the_boundary_gauge_at_one_case(form, semantics, wq):
    assert _differing(form, semantics, wq, (2.024,), (0,)) == {}


@pytest.mark.parametrize("wq", (False, True), ids=("float", "wq4"))
@pytest.mark.parametrize("semantics", SEMANTICS)
@pytest.mark.parametrize("form", CORE_READ_FORMS)
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_a_core_reading_the_raw_input_reads_it_at_the_boundary_gauge_in_both_twins(form, semantics, wq):
    assert _differing(form, semantics, wq, INPUT_DATA_SCALES, SEEDS) == {}


def _gate_refusals(form: str, wq: bool, scales, seeds) -> dict:
    refused = {}
    for input_data_scale in scales:
        for seed in seeds:
            program = _offload(form, seed, "streamed_lif", wq, input_data_scale)
            x = hgf.samples(program, 32, seed)
            try:
                assert_streamed_nf_scm_exact_or_raise(_streamed_stub(program), _NF(program, "streamed_lif"),
                                                      program.ir, x)
            except AssertionError as exc:
                refused[(input_data_scale, seed)] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return refused


@pytest.mark.parametrize("wq", (False, True), ids=("float", "wq4"))
@pytest.mark.parametrize("form", FAST_FORMS)
def test_the_streamed_exactness_gate_passes_on_a_core_reading_the_raw_input_at_one_case(form, wq):
    assert _gate_refusals(form, wq, (2.0,), (1,)) == {}


@pytest.mark.parametrize("wq", (False, True), ids=("float", "wq4"))
@pytest.mark.parametrize("form", CORE_READ_FORMS)
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_the_streamed_exactness_gate_passes_on_a_core_reading_the_raw_input(form, wq):
    assert _gate_refusals(form, wq, INPUT_DATA_SCALES, SEEDS) == {}


def _cores_for(program: hgf.Program) -> list:
    """A pool every form packs into (each IR core its own hard core at most)."""
    return [{"max_axons": 256, "max_neurons": 256, "count": len(program.ir.get_neural_cores()) + 4}]


@pytest.mark.parametrize("semantics", SEMANTICS)
@pytest.mark.parametrize("form", CORE_READ_FORMS)
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_the_packed_program_equals_the_identity_program_and_the_nf(form, semantics):
    differing = {}
    for wq in (False, True):
        for input_data_scale in INPUT_DATA_SCALES:
            for seed in SEEDS:
                program = _offload(form, seed, semantics, wq, input_data_scale)
                packed = build_hybrid_hard_core_mapping(ir_graph=program.ir, cores_config=_cores_for(program),
                                                        retimed_level_stages=hgf.SEMANTICS[semantics][0])
                hcm = SpikingHybridCoreFlow(program.shape, packed, simulation_length=hgf.T, thresholding_mode="<=",
                                            membrane_integer_lattice=bool(wq), **hgf.SEMANTICS[semantics][1]).eval()
                x = hgf.samples(program, 32, seed)
                with torch.no_grad():
                    hcm_out = hcm(x)
                    scm_out = hgf.deployed_flow(program, semantics)(x)
                nf_out, _nf = hgf.nf_window_counts(program, x, semantics)
                ref, scm = hgf.output_counts(program, nf_out, scm_out, 32)
                _ref, packed_out = hgf.output_counts(program, nf_out, hcm_out, 32)
                readings = (float((packed_out - scm).abs().max()), float((torch.round(ref) - packed_out).abs().max()))
                if any(readings):
                    differing[(wq, input_data_scale, seed)] = readings
    assert differing == {}


@pytest.mark.parametrize("form", CORE_READ_FORMS + HOST_READ_FORMS)
@pytest.mark.timeout(600)
def test_the_divisor_table_carries_the_input_root_whenever_a_neural_stage_reads_it(form):
    """Present at the stamped gauge where a neural stage reads the raw input at a gauge other than one, absent at gauge
    one and wherever only host ops read it; the program keeps it through a pickle, and the packed program carries the
    same table (one rule for SCM, HCM and every cross-simulator runner, which all read this table)."""
    for placement in ("offload", "subsume"):
        for input_data_scale in INPUT_DATA_SCALES:
            program = hgf.build(form, 0, THETA_ENC, "streamed_lif", False, establish=_seam(input_data_scale),
                                placement=placement)
            table = boundary_normalization_scales(program.hybrid)
            reads = _neural_stage_reads_input(program.hybrid)
            if reads and input_data_scale != 1.0:
                assert table.get(RAW_INPUT) == float(input_data_scale), (placement, input_data_scale, table)
            else:
                assert RAW_INPUT not in table, (placement, input_data_scale, table)
            if placement == "offload" or form in HOST_READ_FORMS:  # the denominators: who reads the raw input
                assert reads == (form in CORE_READ_FORMS), (placement, reads)
            elif form not in hgf.INPUT_FORMS:  # subsumed, the first perceptron is a host op ...
                assert not reads, (placement, reads)  # ... while input_concat's last core still reads the raw input
            kept = pickle.loads(pickle.dumps(program.hybrid))
            assert boundary_normalization_scales(kept) == table
            packed = build_hybrid_hard_core_mapping(ir_graph=program.ir, cores_config=_cores_for(program))
            assert boundary_normalization_scales(packed) == table


@pytest.mark.parametrize("form", HOST_READ_FORMS)
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_a_host_op_reading_the_raw_input_keeps_decoding_it_at_one(form):
    """The controls stay exact under ``offload`` at every input scale: the host op that reads the raw input decodes the
    raw buffer at one (F-2's armed absolute entry), as before."""
    differing = {}
    for semantics in SEMANTICS:
        for wq in (False, True):
            for input_data_scale in INPUT_DATA_SCALES:
                for seed in range(4):
                    case = _case(form, semantics, wq, input_data_scale, seed)
                    assert case["readers"] == 0
                    if case["differ"] or case["out_diff"] >= 1e-3:  # a host op's output is not on the count grid
                        differing[(semantics, wq, input_data_scale, seed)] = case
    assert differing == {}


def test_the_input_root_fold_belongs_to_the_wire_currency_alone():
    """The fold belongs to the rate/LIF wire currency the wire-gauge seam establishes: a source walk the seam never
    stamped (a value or TTFS graph, or a graph walked before any establishment) folds the raw input at one, and a
    TTFS-family walk (``arm_wire_value_ops=False``) folds it at one even on a stamped graph."""
    _shape, flow, perceptrons, _thetas = hgf.lif_net("encoding_skip", 0, THETA_ENC, placement="offload")
    repr_ = flow.get_mapper_repr()
    first = perceptrons[0]
    compute_per_source_scales(repr_, arm_wire_value_ops=True)
    propagate_boundary_input_scales(flow, input_data_scale=2.0)
    assert torch.equal(first.per_input_scales, torch.ones_like(first.per_input_scales))
    establish_deployed_wire_gauge(flow, input_data_scale=2.0)
    assert torch.equal(first.per_input_scales, torch.full_like(first.per_input_scales, 2.0))
    compute_per_source_scales(repr_, arm_wire_value_ops=True)
    assert torch.equal(first.per_input_scales, torch.full_like(first.per_input_scales, 2.0))
    compute_per_source_scales(repr_, arm_wire_value_ops=False)
    assert torch.equal(first.per_input_scales, torch.ones_like(first.per_input_scales))
    establish_deployed_wire_gauge(flow, input_data_scale=1.0)
    assert torch.equal(first.per_input_scales, torch.ones_like(first.per_input_scales))


def test_the_ir_carries_the_input_gauge_only_where_a_core_reads_the_raw_input():
    for form, placement, expected in (("encoding_skip", "offload", 2.0), ("encoding_skip", "subsume", None),
                                      ("vit_tiny", "offload", None)):
        program = hgf.build(form, 0, THETA_ENC, "streamed_lif", False, establish=_seam(2.0), placement=placement)
        assert program.ir.__dict__.get("input_wire_gauge") == expected, (form, placement)
    program = hgf.build("encoding_skip", 0, THETA_ENC, "streamed_lif", False, establish=_seam(1.0),
                        placement="offload")
    assert "input_wire_gauge" not in program.ir.__dict__
    assert np.isclose(boundary_normalization_scales(program.hybrid).get(RAW_INPUT, 1.0), 1.0)
