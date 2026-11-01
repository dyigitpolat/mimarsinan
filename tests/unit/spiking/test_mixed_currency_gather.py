"""[U-SC S-3, SD-10c D-10c-2] A host gather whose inputs mix wire-currency and absolute chains.

DenseNet-BC's published order concatenates the stem (a convolution without activation: an ABSOLUTE host chain rooted at
the input) with the outputs of the dense units (WIRE currency: rates of the cores, or host ops passing their gauge
through). The deployed program has no node for the concatenation: every consumer reads each producer's buffer slice at
that producer's own gauge (the per-slice divisor table at a neural input, the per-source decode of a domain-owning
wrapper). The chip-aligned NF walk refused the graph at prepare time (``mixed wire/absolute fan-in at a plain host op is
not supported (ConcatMapper)``) and LIF Adaptation's entry repair crashed on it (``'ConcatMapper' object has no attribute
'_sources_list'``), which stopped every S1 T-LIF row at the AQ D-hat.

A layout gather (a concatenation, a stack, a reshape: nodes that move elements and compute nothing) now carries each
element in its producer's currency, as the deployed buffers do, and every reader reads each element by its producer's
rule; it is not a join, so the repair never arms it. A join the repair cannot arm (a mixed fan-in at anything but a host
ComputeOp) is refused by name.
"""

from __future__ import annotations

import pytest
import torch

from mimarsinan.mapping.mappers.structural import ConcatMapper
from mimarsinan.mapping.support.value_domain import heterogeneous_domain_joins, value_domain_map
from mimarsinan.spiking.scale_aware_boundaries import establish_gauge_for_mixed_domain_seams

from unit.spiking.host_gather_forms import SEMANTICS, build, compare, lif_net

SEMANTICS_NAMES = tuple(SEMANTICS)


def _assert_count_exact(program, semantics, n=32, seed=0):
    mism, total, out_diff = compare(program, semantics, n, seed)
    assert total > 0
    assert mism == {pi: 0 for pi in mism}, (program.form, mism, total)
    assert out_diff < 1e-3, out_diff


def _concats(flow):
    return [n for n in flow.get_mapper_repr().execution_order() if isinstance(n, ConcatMapper)]


def test_a_mixed_concatenation_is_a_gather_not_a_join():
    _, flow, _, _ = lif_net("absolute_concat", 0, 1.3)
    repr_ = flow.get_mapper_repr()
    repr_._ensure_exec_graph()
    flags, joins = value_domain_map(repr_._exec_order, repr_._deps)
    (cat,) = _concats(flow)
    assert len({flags[d] for d in repr_._deps[cat]}) == 2, "fixture: the concatenation mixes the two currencies"
    assert joins == [] and heterogeneous_domain_joins(repr_) == []


def test_the_lif_entry_repair_counts_a_mixed_concatenation_as_nothing_to_arm():
    """RC4: ``AttributeError: 'ConcatMapper' object has no attribute '_sources_list'``."""
    _, flow, _, _ = lif_net("absolute_concat", 0, 1.3)
    assert establish_gauge_for_mixed_domain_seams(flow, input_data_scale=1.0) == 0


@pytest.mark.parametrize("wq", (False, True), ids=("float", "wq4"))
@pytest.mark.parametrize("semantics", SEMANTICS_NAMES)
@pytest.mark.parametrize("form", ("absolute_concat", "absolute_concat_host"))
def test_a_mixed_concatenation_is_walked_count_exact(form, semantics, wq):
    """Read by a core (each element re-encoded by its producer's rule) and by a host affine op (each element decoded by
    the wrapper at its producer's gauge)."""
    for seed in range(3):
        _assert_count_exact(build(form, seed, 1.3, semantics, wq), semantics, seed=seed)


@pytest.mark.parametrize("semantics", SEMANTICS_NAMES)
def test_a_densenet_bc_program_is_walked_count_exact(semantics):
    """DenseNet-BC at the smallest depth (published order: the stem concatenated into every unit of the first block)."""
    program = build("densenet_bc", 0, 1.3, semantics, False)
    assert _concats(program.flow), "fixture: the dense blocks concatenate"
    _assert_count_exact(program, semantics, n=16)


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("wq", (False, True), ids=("float", "wq4"))
@pytest.mark.parametrize("semantics", SEMANTICS_NAMES)
def test_densenet_bc_programs_are_walked_count_exact_across_seeds(semantics, wq):
    for seed in range(4):
        _assert_count_exact(build("densenet_bc", seed, 1.3 + 0.2 * seed, semantics, wq), semantics, n=16, seed=seed)


def test_a_join_only_a_host_compute_op_could_own_is_refused_by_name():
    """The repair arms a mixed fan-in through the ComputeOp's wrap slots; any other node type has none to arm."""
    from mimarsinan.common.refusal.toolchain_refusal import ToolchainRefusal
    from mimarsinan.spiking.scale_aware_boundaries import MixedDomainJoinError, _arm_domain_join

    _, flow, _, _ = lif_net("absolute_concat", 0, 1.3)
    (cat,) = _concats(flow)
    assert issubclass(MixedDomainJoinError, ToolchainRefusal)
    with pytest.raises(MixedDomainJoinError, match="ConcatMapper"):
        _arm_domain_join(cat)


def test_a_phase_dithered_walk_refuses_a_mixed_gather_by_name():
    """The dithered comb of a mixed gather is keyed per producer on the deployed side; the twin does not guess it."""
    from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW
    from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver
    from mimarsinan.spiking.segment_policies import MixedGatherDitherError

    program = build("absolute_concat", 0, 1.3, "streamed_lif", False)
    driver = SegmentForwardDriver(program.repr_, 15, LifSegmentPolicy(phase_dither=True, soma_law=DEFAULT_SOMA_LAW))
    with pytest.raises(MixedGatherDitherError, match="dither"):
        with torch.no_grad():
            driver(torch.rand(2, 8))


@pytest.mark.parametrize("form, recorded", (
    ("absolute_concat", 2), ("absolute_concat_host", 0), ("densenet_bc", 0),
    ("nested_gather_host_read", 1), ("nested_gather_core_read", 3),
))
def test_a_mixed_gather_records_its_currency_only_where_a_reader_reads_it(form, recorded, monkeypatch):
    """A host wrapper reads the gather's buffers as they are; only a neural reader (directly, or through the layout
    nodes between: absolute_concat's concatenation and the reshape after it) needs the per-element currency, so a walk
    keeps none for the others (DenseNet-BC's concatenations feed their bn1 host op). A gather inside a neural segment
    encodes each of its sources by that source's own rule, so it reads a mixed source's currency whoever reads the
    gather (the nested forms: the inner concatenation is kept when only a host op reads the outer one; the inner, the
    outer and the reshape after it when a core does)."""
    import mimarsinan.spiking.segment_currency as currency_mod

    calls = []
    real = currency_mod.gather_currency
    monkeypatch.setattr(currency_mod, "gather_currency", lambda *a, **k: calls.append(1) or real(*a, **k))
    program = build(form, 0, 1.3, "streamed_lif", False)
    _assert_count_exact(program, "streamed_lif", n=4)
    n_walks = 4  # compare() walks the NF once per sample: the program evaluates its host kernels per sample (U-SC2)
    assert len(calls) == recorded * n_walks


@pytest.mark.parametrize("wq", (False, True), ids=("float", "wq4"))
@pytest.mark.parametrize("semantics", SEMANTICS_NAMES)
@pytest.mark.parametrize("form", ("nested_gather_host_read", "nested_gather_core_read"))
def test_a_mixed_gather_nested_in_a_spiking_gather_is_walked_count_exact(form, semantics, wq):
    """[U-SC review M1] A host-side mixed concatenation (no source carries spikes) inside a concatenation that also
    takes core outputs: the outer one runs inside a neural segment and encodes each source by its own rule, the inner one
    by each of its elements' producers. Code state 3 kept the inner currency only where a neural reader reads the outer
    gather, so the form read by a host op raised ``KeyError: ConcatMapper()`` under both semantics."""
    for seed in range(2):
        _assert_count_exact(build(form, seed, 1.3, semantics, wq), semantics, n=8, seed=seed)
