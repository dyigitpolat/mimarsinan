"""[U-BRANGE2, m-7] (d)'s key schema: every trained grid of a checkpoint is declared at its site, never recalibrated.

The map (``mimarsinan.quantizer_state.v1``, sha-pinned) gains three site sections and an inventory:

* ``perceptrons[<fx>].groups``: a perceptron's consumer groups, each ``{"readers": [[<bank perceptron>, <role>], ...],
  "range": r}``, keyed by its readers (the names ``consumer_group_readers`` gives each group);
* ``banks[<bank perceptron>]``: an operand bank's host-fed ``entry`` grid and its ``readout_range``;
* ``trained``: ``{<grid>: "<kind>:<site key>" or a list of them}``, kind one of entry, readout, group, bank_entry,
  bank_readout, lane (a group's key is its perceptron's FX name).

A grid the inventory places at no site, or at a site the map does not declare, is refused by name: the toolchain would
recalibrate it. A map without the new sections reads as before.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from mimarsinan.mapping.support.quantizer_state import (
    QUANTIZER_STATE_FORMAT,
    QuantizerStateError,
    load_quantizer_state,
)


def _write(tmp_path, document, name="state.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"format": QUANTIZER_STATE_FORMAT, **document}, sort_keys=True))
    return {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _field(state, name):
    if not hasattr(state, name):
        pytest.fail(f"the declared state has no {name!r}: (d)'s sites are not declared (they would be recalibrated)")
    return getattr(state, name)


def _load(tmp_path, document):
    try:
        return load_quantizer_state(_write(tmp_path, document))
    except QuantizerStateError as refused:
        pytest.fail(f"a map with (d)'s sections is refused as a whole: {refused}")


#: [U-BRANGE2 stage 1e] A grouped perceptron declares its groups alone (a readout range beside them is refused, the
#: schema clause below); the readout range sits on another perceptron.
GROUPS = {"proj": {"groups": [
    {"readers": [["bank_k", "rhs"], ["bank_q", "lhs"]], "range": 2.25},
    {"readers": [["bank_v", "rhs"]], "range": 4.5},
]}, "head": {"readout_range": 3.0}}
BANKS = {"bank_q": {"entry": {"scale": 6.0, "offset": 3.0}, "readout_range": 1.75}, "bank_v": {"readout_range": 0.5}}


def test_a_group_range_is_keyed_by_its_readers_in_any_order(tmp_path):
    state = _load(tmp_path, {"perceptrons": GROUPS})
    groups = _field(state, "group_ranges")
    assert dict(groups["proj"]) == {(("bank_k", "rhs"), ("bank_q", "lhs")): 2.25, (("bank_v", "rhs"),): 4.5}
    assert state.readout_ranges == {"head": 3.0}


def test_a_readout_range_beside_its_groups_is_refused_by_name(tmp_path):
    """[U-BRANGE2 stage 1e, the schema question (stage 1d's i-2)] A grouped readout's range is the largest of its groups
    (``ReadoutQuantizer.calibrate_groups``) and ``calibrate`` fills every group, so a map declaring a perceptron's
    readout_range beside its groups gives one site two declared sources, the groups silently winning: refused by name
    at load (the release's loader refuses any groups field by name). The groups alone, or the range alone, load."""
    groups = GROUPS["proj"]["groups"]
    with pytest.raises(QuantizerStateError, match=r"'proj' declares a readout_range beside its groups"):
        load_quantizer_state(_write(tmp_path, {"perceptrons": {"proj": {"readout_range": 3.0, "groups": groups}}}))
    assert set(_field(_load(tmp_path, {"perceptrons": {"proj": {"groups": groups}}}), "group_ranges")) == {"proj"}
    assert _load(tmp_path, {"perceptrons": {"proj": {"readout_range": 3.0}}}).readout_ranges == {"proj": 3.0}


def test_a_bank_s_entry_grid_and_readout_range_are_keyed_by_the_bank(tmp_path):
    state = _load(tmp_path, {"perceptrons": {}, "banks": BANKS})
    entries, ranges = _field(state, "bank_entry_grids"), _field(state, "bank_readout_ranges")
    assert set(entries) == {"bank_q"} and entries["bank_q"].scale == 6.0 and entries["bank_q"].offset == 3.0
    assert dict(ranges) == {"bank_q": 1.75, "bank_v": 0.5}


def test_every_trained_grid_at_a_declared_site_loads(tmp_path):
    trained = {"g_entry": "entry:stem", "g_readout": "readout:head", "g_group": "group:proj",
               "g_bank": ["bank_entry:bank_q", "bank_readout:bank_q"], "g_bank_v": "bank_readout:bank_v"}
    perceptrons = {**GROUPS, "stem": {"entry": {"scale": 2.0}}}
    state = _load(tmp_path, {"perceptrons": perceptrons, "banks": BANKS, "trained": trained})
    assert dict(_field(state, "trained"))["g_bank"] == ("bank_entry:bank_q", "bank_readout:bank_q")


@pytest.mark.parametrize("site", [None, []])
def test_a_trained_grid_placed_at_no_site_is_refused_by_name(tmp_path, site):
    with pytest.raises(QuantizerStateError, match=r"trained-grid inventory places \['g_lost'\] at no site"):
        load_quantizer_state(_write(tmp_path, {"perceptrons": GROUPS, "trained": {"g_lost": site}}))


@pytest.mark.parametrize("site", ["readout:absent", "group:stem", "bank_entry:bank_v", "lane:pool_0"])
def test_a_trained_grid_at_a_site_the_map_does_not_declare_is_refused(tmp_path, site):
    document = {"perceptrons": {**GROUPS, "stem": {"readout_range": 1.0}}, "banks": BANKS, "trained": {"g": site}}
    with pytest.raises(QuantizerStateError, match=r"sites the map does not declare .*g -> " + site):
        load_quantizer_state(_write(tmp_path, document))


# [U-BRANGE2 stage 1e, the sweep's F-A15] "entry" names a kind with no key: refused by name, never split blind
@pytest.mark.parametrize("site", ["stem", "weights:stem", 3, "entry"])
def test_a_site_not_of_a_declared_kind_is_refused(tmp_path, site):
    with pytest.raises(QuantizerStateError, match="not '<kind>:<key>'"):
        load_quantizer_state(_write(tmp_path, {"perceptrons": GROUPS, "trained": {"g": site}}))


@pytest.mark.parametrize("groups, refusal", [
    ([], "groups must be a non-empty list"),
    ([{"readers": [], "range": 1.0}], "groups entries hold readers"),
    ([{"readers": [["b", "lhs"]]}], "groups entries hold readers"),
    ([{"readers": [["b", "lhs"]], "range": 1.0}, {"readers": [["b", "lhs"]], "range": 2.0}], "twice"),
    # [U-BRANGE2 stage 1d, the check's M-5] a reader is a [bank, role] pair: one item or three are refused by name
    ([{"readers": [["b"]], "range": 1.0}], "groups entries hold readers"),
    ([{"readers": [["b", "lhs", "x"]], "range": 1.0}], "groups entries hold readers"),
    # [U-BRANGE2 stage 1e, the sweep's F-A08] a group holds its readers and its range only: a third field is refused
    ([{"readers": [["b", "lhs"]], "range": 1.0, "rounding": "floor"}], "groups entries hold readers"),
])
def test_a_malformed_group_is_refused_for_what_it_lacks(tmp_path, groups, refusal):
    """A group holds its readers ([[bank, role], ...], each reader a pair) and its range, nothing else; a reader of one
    or three items is refused by name at load (red on K-C2: a three-item reader would load with its third item dropped,
    a one-item reader fail by IndexError)."""
    with pytest.raises(QuantizerStateError, match=refusal):
        load_quantizer_state(_write(tmp_path, {"perceptrons": {"proj": {"groups": groups}}}))


def test_a_map_without_the_new_sections_reads_as_before(tmp_path):
    state = load_quantizer_state(_write(tmp_path, {"perceptrons": {"stem": {"readout_range": 2.5}}}))
    assert state.readout_ranges == {"stem": 2.5}
    assert not dict(getattr(state, "group_ranges", {})) and not dict(getattr(state, "trained", {}))


# --- [U-BRANGE2 stage 1b] the declared ranges reach their sites (the sweep's SW69 and SW70) --------------------------


def _attention(grouped: bool):
    import contextlib
    import io

    import torch

    from mimarsinan.mapping.mappers.operand import operand_perceptrons
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.mapping.support.operand_lowering import lower_contractions
    from mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step import (
        calibrate_readout_quantizers,
        install_readout_quantizers,
    )
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from unit.chip_simulation.value_run.operand_programs import attention_block, operand_platform

    platform = operand_platform(payload_bits=4, grouped=grouped)
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(attention_block(), (3, 8, 8), 5, device="cpu", packaging=MVM_PACKAGING).eval()
    lower_contractions(flow.get_mapper_repr(), platform)
    banks = list(operand_perceptrons(flow.get_mapper_repr()))
    perceptrons = list(flow.get_perceptrons()) + banks
    quantizers = install_readout_quantizers(perceptrons, 8)
    batches = [torch.randn(4, 3, 8, 8, generator=torch.Generator().manual_seed(s)) for s in (5, 6)]
    calibrate_readout_quantizers(flow, perceptrons, quantizers, batches, platform=platform)
    return flow, banks, perceptrons, quantizers


def test_a_declared_bank_readout_range_is_installed_on_its_bank(tmp_path):
    """Sweep SW69: a bank's declared readout range replaces its calibrated range on that bank's readout quantizer,
    tagged with the map's digest; the other banks keep theirs."""
    from mimarsinan.mapping.support.quantizer_state import DECLARED_STATE_ATTRIBUTE
    from mimarsinan.mapping.support.quantizer_state_banks import declare_bank_readout_ranges

    flow, banks, perceptrons, quantizers = _attention(grouped=False)
    by_perceptron = {id(p): q for p, q in zip(perceptrons, quantizers)}
    target, other = banks[0], banks[1]
    kept = float(by_perceptron[id(other)].range)
    declared = 0.625  # exact in the quantizer's float32 range
    assert float(by_perceptron[id(target)].range) != declared, "fixture: the declaration moves the range"
    state = load_quantizer_state(_write(tmp_path, {"perceptrons": {}, "banks": {str(target.name): {
        "readout_range": declared}}}))
    assert declare_bank_readout_ranges(flow, perceptrons, quantizers, state) == 1
    assert float(by_perceptron[id(target)].range) == declared
    assert getattr(by_perceptron[id(target)], DECLARED_STATE_ATTRIBUTE) == state.sha256
    assert float(by_perceptron[id(other)].range) == kept


def test_a_declared_group_range_reaches_the_group_its_readers_name(tmp_path):
    """Sweep SW70: a consumer group's declared range lands on the group its readers name (``consumer_group_readers``),
    here the qkv perceptron's last group, and every other group keeps its calibrated range."""
    from mimarsinan.mapping.mappers.operand.consumer_groups import consumer_group_readers
    from mimarsinan.mapping.support.quantizer_state import perceptron_fx_names
    from mimarsinan.mapping.support.quantizer_state_banks import declare_group_ranges

    flow, _banks, perceptrons, quantizers = _attention(grouped=True)
    repr_ = flow.get_mapper_repr()
    fx = {id(p): name for name, p in perceptron_fx_names(repr_).items()}
    by_perceptron = {id(p): q for p, q in zip(perceptrons, quantizers)}
    readers = consumer_group_readers(repr_)
    pid = next(i for i in readers if by_perceptron.get(i) is not None and len(readers[i]) > 1)
    quantizer = by_perceptron[pid]
    before = [float(r) for r in quantizer.group_ranges.reshape(-1)]
    last = len(readers[pid]) - 1
    declared = 0.6875  # exact in the quantizer's float32 ranges
    assert before[last] != declared, "fixture: the declaration moves the group's range"
    name = next(fx[id(p)] for p in perceptrons if id(p) == pid)
    state = load_quantizer_state(_write(tmp_path, {"perceptrons": {name: {"groups": [
        {"readers": [list(r) for r in readers[pid][last]], "range": declared}]}}}))
    assert declare_group_ranges(flow, perceptrons, quantizers, state) == 1
    after = [float(r) for r in quantizer.group_ranges.reshape(-1)]
    assert after[last] == declared and after[:last] == before[:last], (before, after)


# --- [U-BRANGE2 stage 1c] the declared state reaches the model through the steps' own hooks (the delta verify's M-3
# to M-9) and every installed field, refusal and reader name is the declared one (this stage's sweep, C-ids) ----


def _lowered(grouped: bool):
    """The attention block converted and lowered on the operand platform: the flow, the platform, its operand bank
    perceptrons and two calibration batches (no readout installed yet)."""
    import contextlib
    import io

    import torch

    from mimarsinan.mapping.mappers.operand import operand_perceptrons
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.mapping.support.operand_lowering import lower_contractions
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from unit.chip_simulation.value_run.operand_programs import attention_block, operand_platform

    platform = operand_platform(payload_bits=4, grouped=grouped)
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(attention_block(), (3, 8, 8), 5, device="cpu", packaging=MVM_PACKAGING).eval()
    lower_contractions(flow.get_mapper_repr(), platform)
    banks = list(operand_perceptrons(flow.get_mapper_repr()))
    batches = [torch.randn(4, 3, 8, 8, generator=torch.Generator().manual_seed(s)) for s in (5, 6)]
    return flow, platform, banks, batches


def _calibrated_readouts(flow, platform, banks, batches):
    """Readout Law's install and calibration on every perceptron and bank, as the step runs them."""
    import contextlib
    import io

    from mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step import (
        calibrate_readout_quantizers,
        install_readout_quantizers,
    )

    perceptrons = list(flow.get_perceptrons()) + banks
    quantizers = install_readout_quantizers(perceptrons, 8)
    with contextlib.redirect_stdout(io.StringIO()):
        calibrate_readout_quantizers(flow, perceptrons, quantizers, batches, platform=platform)
    return perceptrons, quantizers


def _calibrated_entries(flow, batches, bits):
    """Boundary Quantization's install and calibration on every segment entry and every host-fed bank entry."""
    from mimarsinan.mapping.mappers.operand import operand_entry_perceptrons
    from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import (
        calibrate_boundary_scales,
        install_boundary_quantizers,
    )
    from mimarsinan.torch_mapping.encoding_layers import segment_entry_perceptrons

    fed = list(operand_entry_perceptrons(flow.get_mapper_repr()))
    entries = list(segment_entry_perceptrons(flow.get_mapper_repr())) + fed
    quantizers = install_boundary_quantizers(entries, bits)
    calibrate_boundary_scales(flow, entries, quantizers, batches)
    return fed, entries, quantizers


def test_boundary_quantization_s_hook_installs_a_declared_bank_entry_grid(tmp_path):
    """[U-BRANGE2 stage 1c, the delta verify's M-3] A bank whose streamed operand is host-fed takes its declared entry
    grid at Boundary Quantization's own hook (``declare_entry_state``), after its calibration. Red on X47."""
    from mimarsinan.models.nn.activations.value_quantizer import DeclaredBoundaryGrid
    from mimarsinan.pipelining.pipeline_steps.quantization.declared_state import declare_entry_state

    flow, platform, _banks, batches = _lowered(grouped=False)
    fed, entries, quantizers = _calibrated_entries(flow, batches, 8)
    assert fed, "fixture: a bank with a host-fed streamed operand"
    quantizer = quantizers[entries.index(fed[0])]
    declared = 6.0
    assert float(quantizer.grid.scale) != declared, "fixture: the declaration moves the grid"
    ref = _write(tmp_path, {"perceptrons": {}, "banks": {str(fed[0].name): {"entry": {"scale": declared}}}})
    assert declare_entry_state(flow, entries, quantizers, {**platform, "quantizer_state": ref}) >= 1
    assert isinstance(quantizer.grid, DeclaredBoundaryGrid) and float(quantizer.grid.scale) == declared


def test_a_declared_bank_entry_grid_carries_every_declared_field(tmp_path):
    """[U-BRANGE2 stage 1c, sweeps C-D2, C-D4 and C-B2 to C-B8] Through the hook, the bank's grid is the declared one in
    every field: the declared scale, offset, rounding and codes, the bank quantizer's own width (here 6 bits, not the
    8 any default reads), the map's digest, and the signed DAC of a chip without executed lanes. The hook returns the
    sites it declared."""
    from mimarsinan.models.nn.activations.value_quantizer import DeclaredBoundaryGrid
    from mimarsinan.pipelining.pipeline_steps.quantization.declared_state import declare_entry_state

    flow, platform, _banks, batches = _lowered(grouped=False)
    fed, entries, quantizers = _calibrated_entries(flow, batches, 6)
    quantizer = quantizers[entries.index(fed[0])]
    body = {"scale": 5.0, "offset": 0.25, "rounding": "floor", "codes": [-7, 29]}
    ref = _write(tmp_path, {"perceptrons": {}, "banks": {str(fed[0].name): {"entry": body}}})
    assert declare_entry_state(flow, entries, quantizers, {**platform, "quantizer_state": ref}) == 1
    grid = quantizer.grid
    assert isinstance(grid, DeclaredBoundaryGrid)
    assert (float(grid.scale), int(grid.bits), float(grid.offset), grid.rounding, grid.code_min, grid.code_max) == (
        5.0, 6, 0.25, "floor", -7, 29), grid
    assert grid.state_sha256 == ref["sha256"] and grid.unsigned_dac is False, grid


def test_readout_law_s_hook_installs_a_declared_bank_readout_range(tmp_path):
    """[U-BRANGE2 stage 1c, the delta verify's M-4] Readout Law's own hook (``declare_readout_state``) carries a
    declared bank readout range onto that bank's readout quantizer and counts it in its report. Red on X48."""
    from mimarsinan.pipelining.pipeline_steps.quantization.declared_state import declare_readout_state

    flow, platform, banks, batches = _lowered(grouped=False)
    perceptrons, quantizers = _calibrated_readouts(flow, platform, banks, batches)
    by_perceptron = {id(p): q for p, q in zip(perceptrons, quantizers)}
    target = banks[0]
    declared = 0.625  # exact in the quantizer's float32 range
    assert float(by_perceptron[id(target)].range) != declared, "fixture: the declaration moves the range"
    ref = _write(tmp_path, {"perceptrons": {}, "banks": {str(target.name): {"readout_range": declared}}})
    report = declare_readout_state(flow, perceptrons, quantizers, {**platform, "quantizer_state": ref})
    assert report.get("declared_bank_ranges") == 1, report
    assert float(by_perceptron[id(target)].range) == declared


def test_readout_law_s_hook_installs_a_declared_group_range(tmp_path):
    """[U-BRANGE2 stage 1c, the delta verify's M-5] Readout Law's own hook carries a declared consumer-group range onto
    the group its readers name; the perceptron's other groups keep their calibrated ranges. Red on X49."""
    from mimarsinan.mapping.mappers.operand.consumer_groups import consumer_group_readers
    from mimarsinan.mapping.support.quantizer_state import perceptron_fx_names
    from mimarsinan.pipelining.pipeline_steps.quantization.declared_state import declare_readout_state

    flow, platform, banks, batches = _lowered(grouped=True)
    perceptrons, quantizers = _calibrated_readouts(flow, platform, banks, batches)
    repr_ = flow.get_mapper_repr()
    fx = {id(p): name for name, p in perceptron_fx_names(repr_).items()}
    by_perceptron = {id(p): q for p, q in zip(perceptrons, quantizers)}
    readers = consumer_group_readers(repr_)
    pid = next(i for i in readers if by_perceptron.get(i) is not None and len(readers[i]) > 1)
    quantizer = by_perceptron[pid]
    before = [float(r) for r in quantizer.group_ranges.reshape(-1)]
    declared = 0.6875  # exact in the quantizer's float32 ranges
    assert before[0] != declared, "fixture: the declaration moves the group's range"
    ref = _write(tmp_path, {"perceptrons": {fx[pid]: {"groups": [
        {"readers": [list(r) for r in readers[pid][0]], "range": declared}]}}})
    report = declare_readout_state(flow, perceptrons, quantizers, {**platform, "quantizer_state": ref})
    assert report.get("declared_groups") == 1, report
    after = [float(r) for r in quantizer.group_ranges.reshape(-1)]
    assert after[0] == declared and after[1:] == before[1:], (before, after)


def test_readout_law_s_hook_reports_only_the_sections_its_map_declares(tmp_path):
    """[U-BRANGE2 stage 1c, sweeps C-D7 and C-D8] A map that declares one perceptron's readout range and no bank or
    group section reports that range alone: the bank and group counts are the sections' own, never zero counts of
    sections the map does not hold."""
    from mimarsinan.mapping.support.quantizer_state import perceptron_fx_names
    from mimarsinan.pipelining.pipeline_steps.quantization.declared_state import declare_readout_state

    flow, platform, banks, batches = _lowered(grouped=True)
    perceptrons, quantizers = _calibrated_readouts(flow, platform, banks, batches)
    name = next(iter(perceptron_fx_names(flow.get_mapper_repr())))
    ref = _write(tmp_path, {"perceptrons": {name: {"readout_range": 1.5}}})
    report = declare_readout_state(flow, perceptrons, quantizers, {**platform, "quantizer_state": ref})
    assert report == {"declared_ranges": 1}, report


def test_an_entry_grid_for_a_bank_without_a_host_fed_operand_is_refused_by_name(tmp_path):
    """[U-BRANGE2 stage 1c, the delta verify's M-9] An entry grid declared for a bank whose streamed operand is not
    host-fed (it carries no boundary quantizer) is refused by name, never an ``AttributeError``. Red on X66."""
    from mimarsinan.mapping.mappers.operand import operand_entry_perceptrons
    from mimarsinan.mapping.support.quantizer_state_banks import declare_bank_entry_grids

    flow, _platform, banks, _batches = _lowered(grouped=False)
    fed = {id(p) for p in operand_entry_perceptrons(flow.get_mapper_repr())}
    inner = next(b for b in banks if id(b) not in fed)
    state = load_quantizer_state(_write(tmp_path, {"perceptrons": {}, "banks": {str(inner.name): {"entry": {
        "scale": 2.0}}}}))
    with pytest.raises(QuantizerStateError, match=str(inner.name)):
        declare_bank_entry_grids(flow, [], [], state)


def test_a_bank_readout_range_for_a_bank_without_a_readout_quantizer_is_refused_by_name(tmp_path):
    """[U-BRANGE2 stage 1c, sweep C-B10] A declared bank readout range whose bank carries no readout quantizer (the
    bank absent from the quantizers handed in) is refused by name, never an ``AttributeError``."""
    from mimarsinan.mapping.support.quantizer_state_banks import declare_bank_readout_ranges

    flow, platform, banks, batches = _lowered(grouped=False)
    perceptrons, quantizers = _calibrated_readouts(flow, platform, banks, batches)
    statics = [(p, q) for p, q in zip(perceptrons, quantizers) if all(p is not b for b in banks)]
    state = load_quantizer_state(_write(tmp_path, {"perceptrons": {}, "banks": {str(banks[0].name): {
        "readout_range": 1.0}}}))
    with pytest.raises(QuantizerStateError, match=str(banks[0].name)):
        declare_bank_readout_ranges(flow, [p for p, _ in statics], [q for _, q in statics], state)


@pytest.mark.parametrize("case", ["unknown perceptron", "ungrouped perceptron", "readers of no group"])
def test_a_group_declaration_its_perceptron_cannot_carry_is_refused_by_name(tmp_path, case):
    """[U-BRANGE2 stage 1c, sweeps C-B16 and C-B17] A declared consumer group is refused by name where its perceptron is
    no perceptron of the model, carries no grouped readout quantizer (the per-perceptron law), or has no group its
    readers name; never a ``KeyError`` or ``ValueError`` from the lookup."""
    from mimarsinan.mapping.mappers.operand.consumer_groups import consumer_group_readers
    from mimarsinan.mapping.support.quantizer_state import perceptron_fx_names
    from mimarsinan.mapping.support.quantizer_state_banks import declare_group_ranges

    flow, platform, banks, batches = _lowered(grouped=case != "ungrouped perceptron")
    perceptrons, quantizers = _calibrated_readouts(flow, platform, banks, batches)
    repr_ = flow.get_mapper_repr()
    fx = {id(p): name for name, p in perceptron_fx_names(repr_).items()}
    readers = consumer_group_readers(repr_)
    pid = next(i for i in readers if i in fx)
    name = {"unknown perceptron": "no_such_perceptron"}.get(case, fx[pid])
    group = [["no_such_bank", "stream"]] if case == "readers of no group" else [list(r) for r in readers[pid][0]]
    state = load_quantizer_state(_write(tmp_path, {"perceptrons": {name: {"groups": [
        {"readers": group, "range": 1.0}]}}}))
    with pytest.raises(QuantizerStateError, match=name):
        declare_group_ranges(flow, perceptrons, quantizers, state)


def test_every_declared_group_of_a_perceptron_is_installed_counted_and_tagged(tmp_path):
    """[U-BRANGE2 stage 1c, sweeps C-B18, C-B19 and C-B21] Two declared groups of one perceptron each land on the group
    their readers name, both are counted, and the grouped quantizer carries the map's digest (so a later stale map is
    told apart by ``verify_declared_state``)."""
    from mimarsinan.mapping.mappers.operand.consumer_groups import consumer_group_readers
    from mimarsinan.mapping.support.quantizer_state import DECLARED_STATE_ATTRIBUTE, perceptron_fx_names
    from mimarsinan.mapping.support.quantizer_state_banks import declare_group_ranges

    flow, platform, banks, batches = _lowered(grouped=True)
    perceptrons, quantizers = _calibrated_readouts(flow, platform, banks, batches)
    repr_ = flow.get_mapper_repr()
    fx = {id(p): name for name, p in perceptron_fx_names(repr_).items()}
    by_perceptron = {id(p): q for p, q in zip(perceptrons, quantizers)}
    readers = consumer_group_readers(repr_)
    pid = next(i for i in readers if by_perceptron.get(i) is not None and len(readers[i]) > 2)
    quantizer = by_perceptron[pid]
    before = [float(r) for r in quantizer.group_ranges.reshape(-1)]
    picked = {1: 0.6875, len(before) - 1: 0.8125}  # exact in float32
    state = load_quantizer_state(_write(tmp_path, {"perceptrons": {fx[pid]: {"groups": [
        {"readers": [list(r) for r in readers[pid][g]], "range": v} for g, v in picked.items()]}}}))
    assert declare_group_ranges(flow, perceptrons, quantizers, state) == 2
    after = [float(r) for r in quantizer.group_ranges.reshape(-1)]
    assert after == [picked.get(g, r) for g, r in enumerate(before)], (before, after)
    assert getattr(quantizer, DECLARED_STATE_ATTRIBUTE, None) == state.sha256


#: The forms a group's readers take, each built by its fixture: one bank reader per group (the attention block), two
#: bank readers (the self-product), two bank readers and a whole reader, one bank reader and a whole reader.
_READER_FORMS = ("one bank reader", "two bank readers", "two bank readers and a whole reader",
                 "one bank reader and a whole reader")


@pytest.mark.parametrize("form", _READER_FORMS)
def test_a_consumer_group_s_readers_carry_their_role(form):
    """[U-BRANGE2 stage 1c, the delta verify's M-6; stage 1d, the sweep's FORM kind] Each group's readers are named with
    their role, and a group is a payload group (``ConsumerGroups.payload``) exactly where one of its readers reads it
    as payload, so a map keyed by a payload reader finds its group: in every form a group's readers take (``form``,
    each fixture asserting it builds its form). Red on X43 (every reader named "stream"), on a whole reader read as a
    payload reader and on a group read as payload only where every reader reads it so."""
    from mimarsinan.mapping.mappers.operand.consumer_groups import consumer_group_partition, consumer_group_readers

    if form == "one bank reader":
        flow = _lowered(grouped=True)[0]
    else:
        import torch

        torch.manual_seed(0)
        model = {"two bank readers": lambda: _self_product(shared=False),
                 "two bank readers and a whole reader": lambda: _self_product(shared=True),
                 "one bank reader and a whole reader": _product_beside}[form]()
        flow = _readers_of(model)[0]
    repr_ = flow.get_mapper_repr()
    partition, readers = consumer_group_partition(repr_), consumer_group_readers(repr_)
    names = [readers[pid][g] for pid in partition for g in range(partition[pid].count)]
    banks = form.split(" and ")[0]
    assert any(sum(r != ("whole", "") for r in n) == (1 if banks == "one bank reader" else 2)
               and (("whole", "") in n) == form.endswith("a whole reader") for n in names), ("fixture: the form", names)
    seen_payload = False
    for pid, groups in partition.items():
        for g, payload in enumerate(groups.payload):
            roles = {role for _name, role in readers[pid][g]}
            assert payload == ("payload" in roles), (pid, g, payload, readers[pid][g])
            seen_payload = seen_payload or payload
    assert seen_payload, "fixture: a group read as payload"


def _self_product(shared: bool):
    """One projection feeding both operands of one product (``y @ y^T``); with ``shared`` its output is also read
    whole by the head, beside the product."""
    import torch

    class SelfProduct(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed, self.proj = torch.nn.Linear(6, 8), torch.nn.Linear(8, 8)
            self.head = torch.nn.Linear(16 + (32 if shared else 0), 5)

        def forward(self, x):
            y = self.proj(torch.relu(self.embed(x)))
            z = torch.matmul(y, y.transpose(-1, -2)).flatten(1)
            return self.head(torch.cat((z, y.flatten(1)), dim=1) if shared else z)

    return SelfProduct()


def _product_beside():
    """Two projections, one per operand of one product (``a @ b^T``), each also read whole by the head beside the
    product: a group read by one bank (in its role) and a whole reader."""
    import torch

    class ProductBeside(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed, self.a, self.b = torch.nn.Linear(6, 8), torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)
            self.head = torch.nn.Linear(16 + 32 + 32, 5)

        def forward(self, x):
            h = torch.relu(self.embed(x))
            ya, yb = self.a(h), self.b(h)
            z = torch.matmul(ya, yb.transpose(-1, -2)).flatten(1)
            return self.head(torch.cat((z, ya.flatten(1), yb.flatten(1)), dim=1))

    return ProductBeside()


def _readers_of(model):
    import contextlib
    import io

    from mimarsinan.mapping.mappers.operand import operand_perceptrons
    from mimarsinan.mapping.mappers.operand.consumer_groups import consumer_group_readers
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.mapping.support.operand_lowering import lower_contractions
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from unit.chip_simulation.value_run.operand_programs import operand_platform

    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(model.eval(), (4, 6), 5, device="cpu", packaging=MVM_PACKAGING).eval()
    lower_contractions(flow.get_mapper_repr(), operand_platform(payload_bits=4, grouped=True))
    banks = {str(p.name) for p in operand_perceptrons(flow.get_mapper_repr())}
    return flow, banks, consumer_group_readers(flow.get_mapper_repr())


@pytest.mark.parametrize("shared", [False, True])
def test_a_group_read_by_two_readers_names_them_sorted_in_every_build(shared):
    """[U-BRANGE2 stage 1c, the delta verify's M-7; stage 1d, the check's M-3] A group's readers come from a set of
    reader keys whose iteration order moves with the mapper's address and the process's hash seed; the names come
    sorted, as the loader keys a declared group (``_groups``), in each of twelve builds at new addresses, in both forms
    of the self-product: two bank readers, and (``shared``) the same two with a whole reader ``("whole", "")`` beside
    them. Red on X63 (the names in the set's order) and on K-B2 (sorted by role alone, which puts the whole reader's
    empty role first)."""
    import torch

    kept, multi = [], []
    for build in range(12):
        torch.manual_seed(build)
        flow, _banks, readers = _readers_of(_self_product(shared=shared))
        kept.append(flow)  # every build stays alive, so each one sits at new addresses
        multi += [names for groups in readers.values() for names in groups if len(names) > 1]
    assert multi, "fixture: a group read by two readers"
    assert any(("whole", "") in names for names in multi) == shared, ("fixture: the whole reader's form", multi)
    assert all(list(names) == sorted(names) for names in multi), multi


def test_every_group_reader_is_named_by_its_bank_and_role_or_as_a_whole_reader():
    """[U-BRANGE2 stage 1c, sweeps C-G1 and C-G2] A reader's name is the toolchain's own: an operand bank perceptron's
    name with the role it reads the group in ("stream" or "payload"), or ``("whole", "")`` for a reader of every
    channel. The projection read by both operands of one bank and whole by the head names all three readers."""
    import torch

    for shared in (False, True):
        torch.manual_seed(0)
        _flow, banks, readers = _readers_of(_self_product(shared))
        names = [r for groups in readers.values() for group in groups for r in group]
        assert names and all(r == ("whole", "") or (r[0] in banks and r[1] in ("stream", "payload")) for r in names), (
            banks, names)
        assert any(r == ("whole", "") for r in names) == shared, names
        if shared:
            assert any(len(group) == 3 and ("whole", "") in group for groups in readers.values() for group in groups), (
                readers)


def test_a_lane_chip_holds_no_operand_bank_entry_grid():
    """[U-BRANGE2 stage 1c, the delta verify's n-4 for X44] A bank's declared entry grid reads the signed DAC: executed
    lanes (a lane chip's unsigned DAC) and an operand core type are refused together by name, at the run's door and
    at the lowering, so no bank entry grid is ever declared on a lane chip."""
    from mimarsinan.mapping.lanes.declaration import LANE_KEY, lane_operand_conflict
    from mimarsinan.mapping.support.operand_lowering import lower_contractions
    from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
    from unit.chip_simulation.value_run.operand_programs import operand_platform
    from unit.mapping.lanes.lane_programs import lane_platform

    both = {**operand_platform(payload_bits=4, grouped=False), LANE_KEY: lane_platform(("L",))[LANE_KEY]}
    assert lane_operand_conflict(both), "an operand core type with executed lanes is admitted"
    with pytest.raises(Exception, match="do not share a chip") as at_door:
        build_platform_constraints_resolved({"core_semantics": "mvm", "device": "cpu", **both})
    from mimarsinan.common.refusal import ToolchainRefusal

    assert isinstance(at_door.value, ToolchainRefusal), type(at_door.value)
    import contextlib
    import io

    import torch

    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from unit.chip_simulation.value_run.operand_programs import attention_block

    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(attention_block(), (3, 8, 8), 5, device="cpu", packaging=MVM_PACKAGING).eval()
    with pytest.raises(ToolchainRefusal, match="do not share a chip"):
        lower_contractions(flow.get_mapper_repr(), both)


def test_on_a_lane_chip_a_declared_entry_grid_reads_the_unsigned_dac(tmp_path):
    """[U-BRANGE2 stage 1c, sweeps C-D2 and C-D3] On a lane chip (executed lanes, WI-5e) Boundary Quantization's hook
    installs a declared entry grid on the DAC that also reads the 2^b unsigned codes: a declared code range 0 to 15 at 4
    bits is the unsigned lane grid there (``unsigned``), where the signed DAC's codes end at 7."""
    import contextlib
    import io

    import torch

    from mimarsinan.mapping.lanes.model_lanes import lane_absorbed
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.mapping.support.quantizer_state import perceptron_fx_names
    from mimarsinan.models.nn.activations.value_quantizer import DeclaredBoundaryGrid
    from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import (
        calibrate_boundary_scales,
        install_boundary_quantizers,
    )
    from mimarsinan.pipelining.pipeline_steps.quantization.declared_state import declare_entry_state
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from mimarsinan.torch_mapping.encoding_layers import segment_entry_perceptrons
    from unit.chip_simulation.value_run.integer_programs import residual_block_net
    from unit.mapping.lanes.lane_programs import lane_platform

    platform = lane_platform(("L", "E", "P"), activation_bits=4)
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(residual_block_net().eval(), (3, 8, 8), 10, device="cpu",
                                   packaging=MVM_PACKAGING).eval()
    entries = list(segment_entry_perceptrons(flow.get_mapper_repr(), lane_absorbed(flow, platform)))
    quantizers = install_boundary_quantizers(entries, 4)
    batches = [torch.randn(8, 3, 8, 8, generator=torch.Generator().manual_seed(1))]
    calibrate_boundary_scales(flow, entries, quantizers, batches)
    names = {id(p): name for name, p in perceptron_fx_names(flow.get_mapper_repr()).items()}
    ref = _write(tmp_path, {"perceptrons": {names[id(entries[0])]: {"entry": {"scale": 3.0, "codes": [0, 15]}}}})
    assert declare_entry_state(flow, entries, quantizers, {**platform, "quantizer_state": ref}) == 1
    grid = quantizers[0].grid
    assert isinstance(grid, DeclaredBoundaryGrid) and grid.unsigned_dac is True and grid.unsigned, grid
    assert (grid.code_min, grid.code_max) == (0, 15)


def test_a_map_without_a_group_section_declares_no_group_and_reads_no_model(tmp_path):
    """[U-BRANGE2 stage 1c, sweep C-B14] A map without consumer groups declares none: the group site returns 0 before
    it reads the model's names or its consumer partition (a model without a mapper graph is never asked for one)."""
    from mimarsinan.mapping.support.quantizer_state_banks import declare_group_ranges

    state = load_quantizer_state(_write(tmp_path, {"perceptrons": {"stem": {"readout_range": 1.0}}}))
    assert declare_group_ranges(object(), [], [], state) == 0


def test_a_group_s_readers_declared_out_of_order_key_one_group(tmp_path):
    """[U-BRANGE2 stage 1c, sweep C-S1] The loader keys a declared group by its readers sorted, whatever order the map
    lists them in (``consumer_group_readers`` names them sorted): readers listed out of order load as the sorted key.
    The fixture of ``test_a_group_range_is_keyed_by_its_readers_in_any_order`` lists them sorted already."""
    unsorted = {"proj": {"groups": [{"readers": [["bank_q", "lhs"], ["bank_k", "rhs"]], "range": 2.25}]}}
    state = _load(tmp_path, {"perceptrons": unsorted})
    assert dict(_field(state, "group_ranges")["proj"]) == {(("bank_k", "rhs"), ("bank_q", "lhs")): 2.25}


@pytest.mark.parametrize("body", [{"readout_range": 1.0, "range": 2.0}, {"entry": {"scale": 2.0}, "groups": []}])
def test_a_bank_declaring_a_field_of_no_bank_site_is_refused_by_name(tmp_path, body):
    """[U-BRANGE2 stage 1c, sweep C-S3] A bank declares its entry grid, its readout range or both, nothing else: a
    field of no bank site is refused by name at load (never silently left to calibration)."""
    with pytest.raises(QuantizerStateError, match="bank 'bank_q' must declare entry, readout_range or both"):
        load_quantizer_state(_write(tmp_path, {"perceptrons": {}, "banks": {"bank_q": body}}))


# --- [U-BRANGE2 stage 1d] every declared site of each hook, in one map built from the fixture's own site inventory;
# one refusal over every site kind (the check's M-1 to M-6, in the generic form of its 'The generic form') ------------


def _off_half(count: int, start: float, step: float) -> list:
    """``count`` distinct values, each exact in float32 and not in float16 or bfloat16 (a trained float32 grid written
    to the map as its float): a value narrowed to a dtype below float32 on its way to its site reads as another one."""
    import torch

    values = [float(torch.tensor(start + step * k, dtype=torch.float32)) for k in range(count)]
    for value in values:
        narrowed = {float(torch.tensor(value, dtype=d)) for d in (torch.float16, torch.bfloat16)}
        assert value not in narrowed, f"fixture: {value!r} is exact below float32"
    assert len(set(values)) == count, "fixture: one value per site"
    return values


def _bits(value) -> str:
    """A value's float64 bits (a float32 site's value widens exactly): equal bits, the same value bit for bit."""
    import struct

    return struct.pack(">d", float(value)).hex()


def test_boundary_quantization_s_hook_installs_every_declared_entry_site_bit_for_bit(tmp_path):
    """[U-BRANGE2 stage 1d, the check's M-6 (S-2) and its generic form] A map that declares every entry site of the
    fixture (every segment entry by its FX name, every host-fed operand bank entry by its bank's name), each site with
    its own scale and offset, exact in float32 and not in float16, goes through Boundary Quantization's own hook
    (``declare_entry_state``): every site's grid is the declared one bit for bit, carries the map's digest and the
    quantizer's own width, and the hook counts the map's sites. Red on a loop over a declared section that stops early
    and on a value narrowed on its way in."""
    from mimarsinan.mapping.support.quantizer_state import perceptron_fx_names
    from mimarsinan.models.nn.activations.value_quantizer import DeclaredBoundaryGrid
    from mimarsinan.pipelining.pipeline_steps.quantization.declared_state import declare_entry_state

    flow, platform, _banks, batches = _lowered(grouped=True)
    fed, entries, quantizers = _calibrated_entries(flow, batches, 8)
    fx = {id(p): name for name, p in perceptron_fx_names(flow.get_mapper_repr()).items()}
    fed_ids = {id(p) for p in fed}
    sites = [("banks", str(p.name)) if id(p) in fed_ids else ("perceptrons", fx[id(p)]) for p in entries]
    assert len({s for s, _ in sites}) == 2 and len(fed) >= 2, "fixture: segment entries and two host-fed bank entries"
    scales, offsets = _off_half(len(sites), 0.3, 0.0371), _off_half(len(sites), 0.01, 0.0173)
    assert all(float(q.grid.scale) != s for q, s in zip(quantizers, scales)), "fixture: the map moves every site"
    document = {"perceptrons": {}, "banks": {}}
    for (section, name), scale, offset in zip(sites, scales, offsets):
        document[section][name] = {"entry": {"scale": scale, "offset": offset}}
    ref = _write(tmp_path, document)
    assert declare_entry_state(flow, entries, quantizers, {**platform, "quantizer_state": ref}) == len(sites)
    grids = [q.grid for q in quantizers]
    assert all(isinstance(g, DeclaredBoundaryGrid) for g in grids), grids
    got = {site: (_bits(g.scale), _bits(g.offset), int(g.bits), g.state_sha256) for site, g in zip(sites, grids)}
    want = {site: (_bits(s), _bits(o), 8, ref["sha256"]) for site, s, o in zip(sites, scales, offsets)}
    assert got == want


def test_readout_law_s_hook_installs_every_declared_readout_site_bit_for_bit(tmp_path):
    """[U-BRANGE2 stage 1d, the check's M-1, M-2 and M-6 (K-A1, K-A2, K-D1, S-1) in their generic form] A map that
    declares every readout site of the grouped fixture (every ungrouped perceptron's readout range by its FX name,
    every operand bank's readout range by its bank's name, every consumer group of every grouped perceptron by its
    readers, as the operand arm's map of record declares them), each value its own and exact in float32 and not in
    float16, goes through Readout Law's own hook (``declare_readout_state``): every range and every group range is the
    declared one bit for bit (a grouped quantizer's range the largest of its groups, the quantizer's own rule), every
    quantizer carries the map's digest, and the report counts the map's sites section by section."""
    from mimarsinan.mapping.mappers.operand.consumer_groups import consumer_group_readers
    from mimarsinan.mapping.support.quantizer_state import DECLARED_STATE_ATTRIBUTE, perceptron_fx_names
    from mimarsinan.pipelining.pipeline_steps.quantization.declared_state import declare_readout_state

    flow, platform, banks, batches = _lowered(grouped=True)
    perceptrons, quantizers = _calibrated_readouts(flow, platform, banks, batches)
    repr_ = flow.get_mapper_repr()
    fx = {id(p): name for name, p in perceptron_fx_names(repr_).items()}
    bank_ids = {id(b) for b in banks}
    readers = consumer_group_readers(repr_)
    grouped = {id(p) for p, q in zip(perceptrons, quantizers) if getattr(q, "channel_groups", None) is not None}
    assert len(grouped) >= 2 and len(banks) >= 2, "fixture: two grouped perceptrons and two operand banks"
    sites = [("banks", str(p.name)) if id(p) in bank_ids else ("perceptrons", fx[id(p)]) for p in perceptrons]
    values = iter(_off_half(len(sites) - len(grouped) + sum(len(readers[pid]) for pid in grouped), 0.1, 0.0371))
    declared = {}  # site: its range, or its groups' (readers, range) in the partition's order
    for site, p in zip(sites, perceptrons):
        declared[site] = [(names, next(values)) for names in readers[id(p)]] if id(p) in grouped else next(values)
    document = {"perceptrons": {}, "banks": {}}
    for (section, name), value in declared.items():
        document[section][name] = ({"groups": [{"readers": [list(r) for r in names], "range": v} for names, v in value]}
                                   if isinstance(value, list) else {"readout_range": value})

    def held(quantizer, value):
        if isinstance(value, list):
            return [_bits(r) for r in quantizer.group_ranges.reshape(-1)], _bits(quantizer.range)
        return _bits(quantizer.range)

    def wanted(value):
        if isinstance(value, list):
            return [_bits(v) for _names, v in value], _bits(max(v for _names, v in value))
        return _bits(value)

    for site, q in zip(sites, quantizers):
        now, want = held(q, declared[site]), wanted(declared[site])
        pairs = zip(now[0], want[0]) if isinstance(now, tuple) else [(now, want)]
        assert all(a != b for a, b in pairs), ("fixture: the map moves every site", site)
    ref = _write(tmp_path, document)
    report = declare_readout_state(flow, perceptrons, quantizers, {**platform, "quantizer_state": ref})
    groups = [v for v in declared.values() if isinstance(v, list)]
    assert report == {"declared_ranges": len(document["perceptrons"]) - len(groups),
                      "declared_bank_ranges": len(document["banks"]),
                      "declared_groups": sum(len(g) for g in groups)}, report
    assert {site: held(q, declared[site]) for site, q in zip(sites, quantizers)} == {
        site: wanted(value) for site, value in declared.items()}
    assert all(getattr(q, DECLARED_STATE_ATTRIBUTE, None) == ref["sha256"] for q in quantizers)


_NO_SITE = "no_such_site"
#: One map per site kind (``SITE_KINDS`` but the lane's, which a lane chip's pools declare), each naming a site the
#: model does not hold, and the step whose hook installs that kind.
_UNHELD = {
    "entry": ("entry", {"perceptrons": {_NO_SITE: {"entry": {"scale": 2.0}}}}),
    "readout": ("readout", {"perceptrons": {_NO_SITE: {"readout_range": 1.0}}}),
    "group": ("readout", {"perceptrons": {_NO_SITE: {"groups": [{"readers": [["whole", ""]], "range": 1.0}]}}}),
    "bank_entry": ("entry", {"perceptrons": {}, "banks": {_NO_SITE: {"entry": {"scale": 2.0}}}}),
    "bank_readout": ("readout", {"perceptrons": {}, "banks": {_NO_SITE: {"readout_range": 1.0}}}),
}


@pytest.mark.parametrize("kind", sorted(_UNHELD))
def test_a_declared_site_the_model_does_not_hold_is_refused_by_name(tmp_path, kind):
    """[U-BRANGE2 stage 1d, the check's M-4 with the delta verify's M-8] A map naming a site the model does not hold is
    refused by name (``QuantizerStateError`` naming the site, never a ``KeyError``) at every site kind, through the
    hook of the step that installs that kind: Boundary Quantization's for the entries, Readout Law's for the readout
    ranges and the groups. Red on X65 (the bank readout site's known-bank check) and on K-C1 (the bank entry site's)."""
    from mimarsinan.pipelining.pipeline_steps.quantization.declared_state import (
        declare_entry_state,
        declare_readout_state,
    )

    step, document = _UNHELD[kind]
    flow, platform, banks, batches = _lowered(grouped=True)
    config = {**platform, "quantizer_state": _write(tmp_path, document)}
    if step == "entry":
        _fed, entries, quantizers = _calibrated_entries(flow, batches, 8)
        with pytest.raises(QuantizerStateError, match=_NO_SITE):
            declare_entry_state(flow, entries, quantizers, config)
    else:
        perceptrons, quantizers = _calibrated_readouts(flow, platform, banks, batches)
        with pytest.raises(QuantizerStateError, match=_NO_SITE):
            declare_readout_state(flow, perceptrons, quantizers, config)


# --- [U-BRANGE2 stage 1d] the loader judges every declared number and every item of every declared list at its own
# site (the sweep's NARROW, SIBLING, EARLY and ARITY kinds at the loader) ----------------------------------------------

#: Every section's numbers in one map: a perceptron's readout range and offset and its entry's scale and offset, a
#: grouped perceptron's group range ([U-BRANGE2 stage 1e] its groups alone, beside no range); a bank's entry scale and
#: offset and its readout range; a pool lane's readout range and offset.
_NUMBERS = ("readout_range", "readout_offset", "entry.scale", "entry.offset", "group.range",
            "bank.entry.scale", "bank.entry.offset", "bank.readout_range", "lane.readout_range", "lane.readout_offset")


def _numbers_map(values: dict) -> dict:
    v = values
    return {"perceptrons": {"proj": {"readout_range": v["readout_range"], "readout_offset": v["readout_offset"],
                                     "entry": {"scale": v["entry.scale"], "offset": v["entry.offset"]}},
                            "mix": {"groups": [{"readers": [["bank_q", "stream"]], "range": v["group.range"]}]}},
            "banks": {"bank_q": {"entry": {"scale": v["bank.entry.scale"], "offset": v["bank.entry.offset"]},
                                 "readout_range": v["bank.readout_range"]}},
            "lanes": {"pool": {"readout_range": v["lane.readout_range"], "readout_offset": v["lane.readout_offset"]}}}


def test_the_loader_holds_every_declared_number_bit_for_bit(tmp_path):
    """[U-BRANGE2 stage 1d, the sweep's NARROW kind] Every number a map declares, in every section (``_NUMBERS``), each
    exact in float32 and not in float16, is the loaded state's value bit for bit: none is narrowed on its way in."""
    values = dict(zip(_NUMBERS, _off_half(len(_NUMBERS), 0.1, 0.0371)))
    state = _load(tmp_path, _numbers_map(values))
    held = {"readout_range": state.readout_ranges["proj"], "readout_offset": state.readout_offsets["proj"],
            "entry.scale": state.entry_grids["proj"].scale, "entry.offset": state.entry_grids["proj"].offset,
            "group.range": dict(_field(state, "group_ranges")["mix"])[(("bank_q", "stream"),)],
            "bank.entry.scale": _field(state, "bank_entry_grids")["bank_q"].scale,
            "bank.entry.offset": state.bank_entry_grids["bank_q"].offset,
            "bank.readout_range": _field(state, "bank_readout_ranges")["bank_q"],
            "lane.readout_range": state.lane_ranges["pool"], "lane.readout_offset": state.lane_offsets["pool"]}
    assert {k: _bits(v) for k, v in held.items()} == {k: _bits(v) for k, v in values.items()}


#: Every positive number of the map, its site's name as the refusal states it.
_POSITIVES = {"readout_range": r"'proj'\.readout_range", "entry.scale": r"'proj'\.entry\.scale",
              "group.range": r"'mix'\.groups\[.*\]\.range", "bank.entry.scale": r"'bank_q'\.entry\.scale",
              "bank.readout_range": r"bank 'bank_q'\.readout_range",
              "lane.readout_range": r"lane 'pool'\.readout_range"}


@pytest.mark.parametrize("bad", [0.0, -0.5, float("inf"), True], ids=["zero", "negative", "infinite", "a boolean"])
@pytest.mark.parametrize("site", sorted(_POSITIVES))
def test_every_positive_number_is_judged_at_its_own_site(tmp_path, site, bad):
    """[U-BRANGE2 stage 1d, the sweep's SIBLING kind] A range or a scale is a finite number above 0 at every site that
    declares one, each judged at its own site: a zero, negative, infinite or boolean value at any one of them, the
    others well formed, is refused naming that site."""
    values = {**{k: 1.0 for k in _NUMBERS}, site: bad}
    with pytest.raises(QuantizerStateError, match=_POSITIVES[site] + r" must be a finite number above 0"):
        load_quantizer_state(_write(tmp_path, _numbers_map(values)))


#: An entry malformed in each way R5's loader names (its own cases, tests/unit/mapping/test_quantizer_state.py), with
#: the field the refusal names.
_MALFORMED_ENTRIES = [({"scale": 0.0}, "scale"), ({"scale": 1.0, "rounding": "up"}, "rounding"),
                      ({"scale": 1.0, "codes": [3, 1]}, "codes"), ({"scale": 1.0, "codes": [0.5, 3]}, "codes"),
                      ({"scale": 1.0, "codes": [0, 2.5]}, "codes"), ({"scale": 1.0, "codes": [0, 1, 2]}, "codes"),
                      ({"scale": 1.0, "offset": float("inf")}, "offset"), ({"scale": 1.0, "offset": True}, "offset"),
                      ({"scale": 1.0, "step": 2}, "step"), ({"offset": 0.5}, "scale"),
                      # [U-BRANGE2 stage 1e, the sweep's F-A06] one code alone is no pair: refused, never an IndexError
                      ({"scale": 1.0, "codes": [3]}, "codes")]


@pytest.mark.parametrize("body, named", _MALFORMED_ENTRIES)
@pytest.mark.parametrize("section", ["perceptrons", "banks"])
def test_every_entry_site_judges_its_entry(tmp_path, section, body, named):
    """[U-BRANGE2 stage 1d, the sweep's SIBLING and ARITY kinds] An entry grid is judged the same way at both sites that
    declare one, a segment entry (``perceptrons``) and an operand bank's host-fed entry (``banks``): each malformed
    form is refused by name at load at either site, its codes a pair of integers judged to the last."""
    document = {"perceptrons": {}, section: {"site_x": {"entry": body}}}
    with pytest.raises(QuantizerStateError, match=named):
        load_quantizer_state(_write(tmp_path, document))


#: Every list the loader judges, with one well-formed item first and a malformed one last (map keys sort: ``a_`` before
#: ``z_``), and the refusal that names it.
_LAST_ITEMS = {
    "trained grids, a site of no kind": ({"perceptrons": {"proj": {"readout_range": 1.0}},
                                          "trained": {"a_ok": "readout:proj", "z_bad": "weights:proj"}},
                                         "not '<kind>:<key>'"),
    "trained grids, a grid at no site": ({"perceptrons": {"proj": {"readout_range": 1.0}},
                                          "trained": {"a_ok": "readout:proj", "z_lost": None}},
                                         r"places \['z_lost'\] at no site"),
    "trained grids, an undeclared site": ({"perceptrons": {"proj": {"readout_range": 1.0}},
                                           "trained": {"a_ok": "readout:proj", "z_bad": "readout:absent"}},
                                          "z_bad -> readout:absent"),
    "a grid's sites, a site of no kind": ({"perceptrons": {"proj": {"readout_range": 1.0}},
                                           "trained": {"g": ["readout:proj", "weights:proj"]}}, "not '<kind>:<key>'"),
    "a grid's sites, an undeclared site": ({"perceptrons": {"proj": {"readout_range": 1.0}},
                                            "trained": {"g": ["readout:proj", "readout:absent"]}},
                                           "g -> readout:absent"),
    "a group's readers": ({"perceptrons": {"proj": {"groups": [
        {"readers": [["bank_k", "stream"], ["bank_q"]], "range": 1.0}]}}}, "groups entries hold readers"),
    "a perceptron's groups": ({"perceptrons": {"proj": {"groups": [
        {"readers": [["bank_k", "stream"]], "range": 1.0}, {"readers": [], "range": 1.0}]}}},
        "groups entries hold readers"),
    "the banks": ({"perceptrons": {}, "banks": {"a_ok": {"readout_range": 1.0}, "z_bad": {}}},
                  "bank 'z_bad' must declare"),
    "the perceptrons": ({"perceptrons": {"a_ok": {"readout_range": 1.0}, "z_bad": {}}}, "'z_bad' must declare"),
    "the lanes": ({"perceptrons": {}, "lanes": {"a_ok": {"readout_range": 1.0}, "z_bad": {}}},
                  "lane 'z_bad' must declare"),
}


@pytest.mark.parametrize("where", sorted(_LAST_ITEMS))
def test_the_loader_judges_every_item_of_every_declared_list(tmp_path, where):
    """[U-BRANGE2 stage 1d, the sweep's EARLY kind at the loader] Every list or mapping the loader judges item by item
    (the trained grids, a grid's sites, a group's readers, a perceptron's groups, the banks, the perceptrons, the lanes)
    is judged to its last item: a malformed item after a well-formed one is refused by name."""
    document, refusal = _LAST_ITEMS[where]
    with pytest.raises(QuantizerStateError, match=refusal):
        load_quantizer_state(_write(tmp_path, document))


# --- [U-BRANGE2 stage 1d] the identity output law at Readout Law's hook, on a model with two outputs (the sweep's
# EARLY kind in the hook's identity branch) -------------------------------------------------------------------------

_IDENTITY = {"readout_law": {"adc_bits": 40, "range": "calibrated", "mult_bits": 24, "output_law": "identity"},
             "activation_bits": 5}


def _two_outputs():
    """A hidden perceptron and two output heads side by side (``cat``): two model output perceptrons, each quantizer
    installed and calibrated as Readout Law does."""
    import contextlib
    import io

    import torch

    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step import (
        calibrate_readout_quantizers,
        install_readout_quantizers,
    )
    from mimarsinan.torch_mapping.converter import convert_torch_model

    class TwoHeads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden, self.left, self.right = torch.nn.Linear(8, 6), torch.nn.Linear(6, 3), torch.nn.Linear(6, 2)

        def forward(self, x):
            h = torch.relu(self.hidden(x))
            return torch.cat((self.left(h), self.right(h)), dim=1)

    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(TwoHeads().eval(), (8,), 5, device="cpu", packaging=MVM_PACKAGING).eval()
        perceptrons = list(flow.get_perceptrons())
        quantizers = install_readout_quantizers(perceptrons, 5)
        batches = [torch.randn(32, 8, generator=torch.Generator().manual_seed(1))]
        calibrate_readout_quantizers(flow, perceptrons, quantizers, batches, platform=_IDENTITY)
    return flow, perceptrons, quantizers


@pytest.mark.parametrize("declared", ["the hidden perceptron", "the first output", "the second output"])
def test_under_the_identity_output_law_every_output_is_inert_and_takes_no_declared_range(tmp_path, declared):
    """[U-BRANGE2 stage 1d, the sweep's EARLY kind] Under the identity output law Readout Law's hook makes every model
    output's quantizer inert (range 0) and counts every output, beside a declared range of a hidden perceptron; a map
    declaring a readout range for any one output, the second as well as the first, is refused naming it."""
    from mimarsinan.mapping.export.output_law import model_output_perceptrons
    from mimarsinan.mapping.support.quantizer_state import perceptron_fx_names
    from mimarsinan.pipelining.pipeline_steps.quantization.declared_state import declare_readout_state

    flow, perceptrons, quantizers = _two_outputs()
    repr_ = flow.get_mapper_repr()
    fx = {id(p): name for name, p in perceptron_fx_names(repr_).items()}
    outputs = model_output_perceptrons(repr_)
    hidden = [p for p in perceptrons if all(p is not o for o in outputs)]
    assert len(outputs) == 2 and len(hidden) == 1, "fixture: two output heads beside one hidden perceptron"
    site = {"the hidden perceptron": hidden[0], "the first output": outputs[0],
            "the second output": outputs[1]}[declared]
    ref = _write(tmp_path, {"perceptrons": {fx[id(site)]: {"readout_range": 0.625}}})
    if site is hidden[0]:
        report = declare_readout_state(flow, perceptrons, quantizers, {**_IDENTITY, "quantizer_state": ref})
        by_perceptron = {id(p): q for p, q in zip(perceptrons, quantizers)}
        assert report == {"declared_ranges": 1, "identity_outputs": 2}, report
        assert [float(by_perceptron[id(p)].range) for p in hidden + outputs] == [0.625, 0.0, 0.0]
    else:
        with pytest.raises(QuantizerStateError, match=repr(fx[id(site)])):
            declare_readout_state(flow, perceptrons, quantizers, {**_IDENTITY, "quantizer_state": ref})


# --- [U-BRANGE2 stage 1d, the blocker; stage 1e, its fix] the stale-state check reads every site kind the map declares,
# from the one inventory (``DECLARED_SITES``), with no kind named in the check -----------------------------------------

#: Stage 1d's five kinds, read where the inventory is absent (an earlier head, where these clauses read red).
_STAGE_1D_KINDS = ("entry", "readout", "group", "bank_entry", "bank_readout")
_FACES = ("read under another map", "read under no map", "a declared site no quantizer read", "its own tag missing")


def _inventory() -> tuple:
    """The inventory's kinds in its order (``DECLARED_SITES``), or stage 1d's five on a head without it."""
    try:
        from mimarsinan.mapping.support.quantizer_state_schema import DECLARED_SITES
    except ImportError:
        return _STAGE_1D_KINDS
    return tuple(DECLARED_SITES)


def _attention_sites():
    """The grouped attention fixture through both steps' install and calibration, and for each kind it holds the map
    declaring one of its sites at ``value``."""
    from mimarsinan.mapping.mappers.operand.consumer_groups import consumer_group_readers
    from mimarsinan.mapping.support.quantizer_state import perceptron_fx_names

    flow, platform, banks, batches = _lowered(grouped=True)
    fed, entries, equantizers = _calibrated_entries(flow, batches, 8)
    perceptrons, quantizers = _calibrated_readouts(flow, platform, banks, batches)
    repr_ = flow.get_mapper_repr()
    fx = {id(p): name for name, p in perceptron_fx_names(repr_).items()}
    readers = consumer_group_readers(repr_)
    grouped = next(pid for pid in readers if len(readers[pid]) > 1)
    documents = {
        "entry": lambda value: {"perceptrons": {fx[id(entries[0])]: {"entry": {"scale": value}}}},
        "readout": lambda value: {"perceptrons": {fx[id(perceptrons[0])]: {"readout_range": value}}},
        "group": lambda value: {"perceptrons": {fx[grouped]: {"groups": [
            {"readers": [list(r) for r in readers[grouped][0]], "range": value}]}}},
        "bank_entry": lambda value: {"perceptrons": {}, "banks": {str(fed[0].name): {"entry": {"scale": value}}}},
        "bank_readout": lambda value: {"perceptrons": {}, "banks": {str(banks[0].name): {"readout_range": value}}},
    }
    return flow, platform, (entries, equantizers), (perceptrons, quantizers), [], documents


def _signed_pool_net():
    """A clamped stem, two convolutions added on chip (a fused add with no clamp) and an average pool over the sum: on
    the lane chip a signed pool lane (an offset admitted) beside signed perceptron lanes."""
    import torch

    class SignedPoolNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = torch.nn.Conv2d(3, 8, 3, padding=1)
            self.a, self.b = torch.nn.Conv2d(8, 8, 3, padding=1), torch.nn.Conv2d(8, 8, 3, padding=1)
            self.pool, self.fc = torch.nn.AdaptiveAvgPool2d(1), torch.nn.Linear(8, 10)

        def forward(self, x):
            h = torch.relu(self.stem(x))
            return self.fc(self.pool(self.a(h) + self.b(h)).flatten(1))

    return SignedPoolNet()


def _lane_sites():
    """The signed-pool net on the lane chip through Boundary Quantization's and Readout Law's install and calibration
    (the lanes path, each lane at its end, as ``build_lane_program`` orders them), and for each lane kind the map
    declaring one of its sites at ``value``."""
    import contextlib
    import io

    import torch

    from mimarsinan.mapping.lanes.model_lanes import install_planned_lanes, lane_absorbed
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.mapping.support.quantizer_state import perceptron_fx_names
    from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import (
        calibrate_boundary_scales,
        install_boundary_quantizers,
    )
    from mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step import (
        ReadoutLawStep,
        install_readout_quantizers,
    )
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from mimarsinan.torch_mapping.encoding_layers import segment_entry_perceptrons
    from unit.mapping.lanes.lane_programs import lane_platform

    platform = lane_platform()
    bits = int(platform["activation_bits"])
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(_signed_pool_net().eval(), (3, 8, 8), 10, device="cpu",
                                   packaging=MVM_PACKAGING).eval()
    batches = [torch.randn(8, 3, 8, 8, generator=torch.Generator().manual_seed(1))]
    absorbed = lane_absorbed(flow, platform)
    entries = list(segment_entry_perceptrons(flow.get_mapper_repr(), absorbed))
    equantizers = install_boundary_quantizers(entries, bits)
    calibrate_boundary_scales(flow, entries, equantizers, batches)
    perceptrons = list(flow.get_perceptrons())
    quantizers = install_readout_quantizers(perceptrons, bits)
    lanes = install_planned_lanes(flow, platform, bits) or []
    every = quantizers + [q for _, q in lanes]
    with contextlib.redirect_stdout(io.StringIO()):
        ranges = ReadoutLawStep._calibrated(flow, perceptrons, every, batches, lanes, absorbed)
    for quantizer, value in zip(every, ranges):
        quantizer.calibrate(value)
    fx = {id(p): name for name, p in perceptron_fx_names(flow.get_mapper_repr()).items()}
    signed = [fx[id(p)] for p, q in zip(perceptrons, quantizers) if not q.lane_unsigned and not q.lane_continues]
    pools = [name for name, q in lanes if not q.lane_unsigned]
    assert signed and pools, ("fixture: a signed perceptron lane and a signed pool lane", signed, lanes)
    documents = {
        "readout_offset": lambda value: {"perceptrons": {signed[0]: {"readout_range": 1.5, "readout_offset": value}}},
        "lane": lambda value: {"perceptrons": {}, "lanes": {pools[0]: {"readout_range": value}}},
        "lane_offset": lambda value: {"perceptrons": {}, "lanes": {pools[0]: {"readout_range": 1.5,
                                                                               "readout_offset": value}}},
    }
    return flow, platform, (entries, equantizers), (perceptrons, quantizers), lanes, documents


#: Per kind of the inventory, the fixture that holds one of its sites. A kind added to the inventory needs its fixture
#: here: the clause reads every kind of the inventory and fails by name on a kind no fixture holds.
_SITE_FIXTURES = {"entry": _attention_sites, "readout": _attention_sites, "group": _attention_sites,
                  "bank_entry": _attention_sites, "bank_readout": _attention_sites,
                  "readout_offset": _lane_sites, "lane": _lane_sites, "lane_offset": _lane_sites}


def _stale_program(tmp_path, kind, *, declared: bool):
    """A program holding one site of ``kind``, with (``declared``) map A declaring that site through both steps' own
    hooks; and map B, the same site at another value (another digest)."""
    from mimarsinan.pipelining.pipeline_steps.quantization.declared_state import (
        declare_entry_state,
        declare_readout_state,
    )

    if kind not in _SITE_FIXTURES:
        pytest.fail(f"no fixture holds a {kind} site: every kind of the inventory needs one in _SITE_FIXTURES")
    flow, platform, (entries, equantizers), (perceptrons, quantizers), lanes, documents = _SITE_FIXTURES[kind]()
    a, b = _write(tmp_path, documents[kind](0.625), "a.json"), _write(tmp_path, documents[kind](0.75), "b.json")
    if declared:
        declare_entry_state(flow, entries, equantizers, {**platform, "quantizer_state": a})
        declare_readout_state(flow, perceptrons, quantizers, {**platform, "quantizer_state": a}, lanes)
    return flow, a, b


def _untag(flow, kind) -> None:
    """Take the tag of every ``kind`` site off the program, in the tag form the inventory names for the kind (a declared
    entry grid put back to its calibrated one through the quantizer's own ``calibrate``, or a quantizer's range or
    offset digest removed); the other kinds' tags stay."""
    from mimarsinan.mapping.support.quantizer_state import (
        DECLARED_OFFSET_ATTRIBUTE,
        DECLARED_SITES,
        DECLARED_STATE_ATTRIBUTE,
        site_owners,
    )
    from mimarsinan.models.nn.activations.value_quantizer import DeclaredBoundaryGrid, ValueGridQuantizer

    spec = DECLARED_SITES[kind]
    removed = 0
    for wire, quantizer in site_owners(flow.get_mapper_repr())[spec.owner].values():
        if spec.tag == "grid":
            for module in wire.modules() if wire is not None else ():
                if isinstance(module, ValueGridQuantizer) and isinstance(module.grid, DeclaredBoundaryGrid):
                    module.calibrate(float(module.scale))
                    removed += 1
            continue
        attribute = {"range": DECLARED_STATE_ATTRIBUTE, "offset": DECLARED_OFFSET_ATTRIBUTE}[spec.tag]
        if quantizer is not None and hasattr(quantizer, attribute):
            delattr(quantizer, attribute)
            removed += 1
    assert removed, ("fixture: the declared site carries its tag", kind)


@pytest.mark.parametrize("face", _FACES)
@pytest.mark.parametrize("kind", _inventory())
def test_the_stale_state_check_reads_every_declared_site_kind(tmp_path, kind, face):
    """[U-BRANGE2 stage 1d, the blocker the sweep's EARLY kind found in ``verify_declared_state``; stage 1e, read over
    the inventory itself] Soft Core Mapping's check (``install_boundary_grids`` -> ``verify_declared_state``: "a model
    whose grids or ranges are not exactly the declared quantizer_state's is refused by name") reads every site kind of
    the one inventory (``DECLARED_SITES``, parametrized here over the inventory, so a kind added to it is read here
    too): a program built under map A through both steps' hooks reads clean under A; read under another map or under
    no map, or under a map declaring a site no quantizer read, it is refused by name; with its own site's tag taken off
    (``_untag``), it is refused as unread naming that kind. RED ON THE FROZEN SOURCE fb0fa80f for bank_entry and
    bank_readout (every face) and for group (the unread face): the check walked the graph's perceptrons only, with
    four kinds listed by hand (quantizer_state.py:100-116, :129-130), while install_boundary_grids stamps the banks'
    grids too (boundary_grids.py:27-33)."""
    from mimarsinan.mapping.support.quantizer_state import verify_declared_state

    flow, a, b = _stale_program(tmp_path, kind, declared=face != "a declared site no quantizer read")
    if face in ("read under another map", "its own tag missing"):
        verify_declared_state(flow, a)  # the map it was built under reads clean
    if face == "its own tag missing":
        _untag(flow, kind)
        with pytest.raises(QuantizerStateError, match=rf"declares a {kind} \("):
            verify_declared_state(flow, a)
        return
    with pytest.raises(QuantizerStateError):
        verify_declared_state(flow, {"read under another map": b, "read under no map": None,
                                     "a declared site no quantizer read": b}[face])


def test_a_site_kind_added_to_the_inventory_is_read_with_no_edit_of_the_check(tmp_path, monkeypatch):
    """[U-BRANGE2 stage 1e] The check names no kind: a kind registered in the inventory (a test double: a synthetic kind
    whose declarations sit in a field of a stand-in state, its tag an attribute of a perceptron's readout slot read by
    a registered tag reader) is read by ``verify_declared_state`` as it stands. The declared site carrying the map's
    digest reads clean; carrying another digest it is refused as stale, carrying none as unread, each naming the kind;
    and with no map declared its tag alone is refused."""
    import dataclasses
    import types

    from mimarsinan.mapping.support import quantizer_state, quantizer_state_schema
    from mimarsinan.mapping.support.quantizer_state import SiteKind, perceptron_fx_names, verify_declared_state

    flow = _lowered(grouped=False)[0]
    name, perceptron = next(iter(perceptron_fx_names(flow.get_mapper_repr()).items()))
    ref = _write(tmp_path, {"perceptrons": {}})
    real = load_quantizer_state(ref)
    double = types.SimpleNamespace(**{f.name: getattr(real, f.name) for f in dataclasses.fields(real)},
                                   synthetic_sites={name: 1.0})
    inventory = types.MappingProxyType({**quantizer_state_schema.DECLARED_SITES,
                                        "synthetic": SiteKind("synthetic_sites", "perceptron", "synthetic")})
    tags = types.MappingProxyType({**quantizer_state.SITE_TAGS,
                                   "synthetic": lambda _wire, slot: getattr(slot, "synthetic_sha256", None)})
    monkeypatch.setattr(quantizer_state_schema, "DECLARED_SITES", inventory)
    monkeypatch.setattr(quantizer_state, "DECLARED_SITES", inventory)
    monkeypatch.setattr(quantizer_state, "SITE_TAGS", tags)
    monkeypatch.setattr(quantizer_state, "load_quantizer_state", lambda d: None if d is None else double)
    with pytest.raises(QuantizerStateError, match=rf"declares a synthetic \(synthetic_sites\) for \['{name}'\]"):
        verify_declared_state(flow, ref)
    setattr(perceptron.scaler, "synthetic_sha256", real.sha256)
    verify_declared_state(flow, ref)
    setattr(perceptron.scaler, "synthetic_sha256", "0" * 64)
    with pytest.raises(QuantizerStateError, match=rf"\['{name}'\] carry a synthetic \(synthetic_sites\) from a stale"):
        verify_declared_state(flow, ref)
    with pytest.raises(QuantizerStateError, match=rf"synthetic:{name}"):
        verify_declared_state(flow, None)


def test_the_inventory_holds_every_declared_field_of_the_state_and_a_reader_of_each_kind():
    """[U-BRANGE2 stage 1e, the class sweep] The one inventory is complete: its fields are exactly the declared state's
    site fields, one kind each (every ``QuantizerState`` field but its digest and the trained-grid inventory), each
    kind's owner is a class ``site_owners`` walks and its tag one ``SITE_TAGS`` reads, and the trained-grid sites' kinds
    are the inventory's (``SITE_KINDS``). A section added to the state without its kind, or a kind without its reader,
    is red here."""
    import dataclasses

    from mimarsinan.mapping.support.quantizer_state import (
        DECLARED_SITES,
        SITE_KINDS,
        SITE_TAGS,
        QuantizerState,
        site_owners,
    )

    fields = {f.name for f in dataclasses.fields(QuantizerState)} - {"sha256", "trained"}
    assert sorted(spec.field for spec in DECLARED_SITES.values()) == sorted(fields)
    owners = site_owners(_lowered(grouped=False)[0].get_mapper_repr())
    assert {spec.owner for spec in DECLARED_SITES.values()} <= set(owners), owners.keys()
    assert {spec.tag for spec in DECLARED_SITES.values()} <= set(SITE_TAGS)
    assert SITE_KINDS == tuple(DECLARED_SITES)


def test_every_entry_grid_the_check_reads_is_the_grid_soft_core_mapping_stamps(tmp_path):
    """[U-BRANGE2 stage 1e, the class sweep] The stamp (``boundary_grids._perceptron_grids``) and the check read one set
    of entry grid sites: with every segment entry and every host-fed bank entry declared through Boundary
    Quantization's hook, each declared grid the check reads (every grid-tagged kind of the inventory, over its owners)
    is a grid the stamp puts on a core, the very object, and the stamp puts no other declared grid."""
    from mimarsinan.mapping.support.boundary_grids import _perceptron_grids
    from mimarsinan.mapping.support.quantizer_state import (
        DECLARED_SITES,
        SITE_TAGS,
        perceptron_fx_names,
        site_owners,
    )
    from mimarsinan.models.nn.activations.value_quantizer import DeclaredBoundaryGrid, boundary_grid_of
    from mimarsinan.pipelining.pipeline_steps.quantization.declared_state import declare_entry_state

    flow, platform, _banks, batches = _lowered(grouped=True)
    fed, entries, quantizers = _calibrated_entries(flow, batches, 8)
    repr_ = flow.get_mapper_repr()
    fx = {id(p): name for name, p in perceptron_fx_names(repr_).items()}
    fed_ids = {id(p) for p in fed}
    document = {"perceptrons": {fx[id(p)]: {"entry": {"scale": 2.0}} for p in entries if id(p) not in fed_ids},
                "banks": {str(p.name): {"entry": {"scale": 3.0}} for p in fed}}
    assert declare_entry_state(flow, entries, quantizers, {**platform, "quantizer_state": _write(tmp_path, document)}) \
        == len(entries)
    repr_.assign_perceptron_indices()
    stamped = [g for g in _perceptron_grids(repr_).values() if isinstance(g, DeclaredBoundaryGrid)]
    owners = site_owners(repr_)
    read = [boundary_grid_of(wire) for spec in DECLARED_SITES.values() if spec.tag == "grid"
            for wire, quantizer in owners[spec.owner].values() if SITE_TAGS[spec.tag](wire, quantizer) is not None]
    assert len(read) == len(entries) and len(fed) >= 2, "fixture: every entry declared, two host-fed banks among them"
    assert sorted(id(g) for g in read) == sorted(id(g) for g in stamped)


# --- [U-BRANGE2 stage 1e] a whole reader reads every channel (stage 1d's survivor D-E49) ------------------------------


@pytest.mark.parametrize("form", ["two bank readers and a whole reader", "one bank reader and a whole reader"])
def test_a_whole_reader_reads_every_channel_of_its_perceptron(form):
    """[U-BRANGE2 stage 1e, stage 1d's survivor D-E49] A reader of a perceptron beside its products (or the output
    layer) reads every one of its channels, so every consumer group of such a perceptron names the whole reader
    ``("whole", "")`` and every channel lies in one of them: a whole reader added to some channels only would split the
    perceptron into groups with and without it. Red on D-E49 (consumer_groups.py:86 reading the first channel only)."""
    import torch

    from mimarsinan.mapping.mappers.operand.consumer_groups import consumer_group_partition

    torch.manual_seed(0)
    model = {"two bank readers and a whole reader": lambda: _self_product(shared=True),
             "one bank reader and a whole reader": _product_beside}[form]()
    flow, _banks, readers = _readers_of(model)
    partition = consumer_group_partition(flow.get_mapper_repr())
    whole = [pid for pid, groups in readers.items() if any(("whole", "") in names for names in groups)]
    assert whole, "fixture: a perceptron read whole beside a product"
    for pid in whole:
        assert all(("whole", "") in names for names in readers[pid]), (pid, readers[pid])
        assert sorted(set(partition[pid].channels.tolist())) == list(range(partition[pid].count))


# --- [U-BRANGE2 stage 1e] the sweep's survivors in the check (F-E09, F-E21, F-E22 of the regenerated list) ----------


def test_a_declared_entry_grid_behind_another_wire_op_is_read(tmp_path):
    """[U-BRANGE2 stage 1e, the sweep's F-E09] A segment entry whose input wire already holds an op before Boundary
    Quantization appends its quantizer (``append_input_wire_op``: the wire becomes a chain) carries its declared grid
    inside that chain; the stale-state check reads the grid there, as the stamp does (``boundary_grid_of`` walks the
    chain): the program reads clean under its map and is refused under no map. Red on F-E09 (the wire's first module
    read alone: the chain, so the grid unread)."""
    import torch

    from mimarsinan.mapping.support.quantizer_state import perceptron_fx_names, verify_declared_state
    from mimarsinan.models.nn.activations.value_quantizer import DeclaredBoundaryGrid, boundary_grid_of
    from mimarsinan.pipelining.pipeline_steps.quantization.declared_state import declare_entry_state
    from mimarsinan.torch_mapping.encoding_layers import segment_entry_perceptrons

    flow, platform, _banks, batches = _lowered(grouped=False)
    first = list(segment_entry_perceptrons(flow.get_mapper_repr()))[0]
    first.append_input_wire_op(torch.nn.Dropout(p=0.0))  # an op already on the wire (inert in eval)
    _fed, entries, quantizers = _calibrated_entries(flow, batches, 8)
    assert isinstance(first.input_activation, torch.nn.Sequential), "fixture: the entry's wire is a chain"
    name = {id(p): n for n, p in perceptron_fx_names(flow.get_mapper_repr()).items()}[id(first)]
    ref = _write(tmp_path, {"perceptrons": {name: {"entry": {"scale": 2.0}}}})
    assert declare_entry_state(flow, entries, quantizers, {**platform, "quantizer_state": ref}) == 1
    assert isinstance(boundary_grid_of(first.input_activation), DeclaredBoundaryGrid)
    verify_declared_state(flow, ref)
    with pytest.raises(QuantizerStateError, match=rf"entry:{name}"):
        verify_declared_state(flow, None)


def _two_readout_sites():
    """The attention fixture's readouts calibrated, and the FX names of the first two perceptrons in the order the
    check walks them (``site_owners``)."""
    from mimarsinan.mapping.support.quantizer_state import site_owners

    flow, platform, banks, batches = _lowered(grouped=False)
    perceptrons, quantizers = _calibrated_readouts(flow, platform, banks, batches)
    first, second = list(site_owners(flow.get_mapper_repr())["perceptron"])[:2]
    return flow, platform, perceptrons, quantizers, first, second


def test_with_no_map_declared_every_site_carrying_a_digest_is_named(tmp_path):
    """[U-BRANGE2 stage 1e, the sweep's F-E21] Read under no map, a program carrying declared state at several sites of
    one kind is refused naming every one of them (``<kind>:<key>``), not the first alone. Red on F-E21."""
    from mimarsinan.mapping.support.quantizer_state import verify_declared_state
    from mimarsinan.pipelining.pipeline_steps.quantization.declared_state import declare_readout_state

    flow, platform, perceptrons, quantizers, first, second = _two_readout_sites()
    ref = _write(tmp_path, {"perceptrons": {first: {"readout_range": 0.625}, second: {"readout_range": 0.75}}})
    declare_readout_state(flow, perceptrons, quantizers, {**platform, "quantizer_state": ref})
    verify_declared_state(flow, ref)
    with pytest.raises(QuantizerStateError) as refused:
        verify_declared_state(flow, None)
    assert f"readout:{first}" in str(refused.value) and f"readout:{second}" in str(refused.value), refused.value


def test_a_kind_s_sites_carrying_two_maps_digests_are_refused_at_the_stale_one(tmp_path):
    """[U-BRANGE2 stage 1e, the sweep's F-E22] Every site of a kind is compared with the map, not the first alone: a
    program built under map A at two readout sites, one of them declared again under map B through Readout Law's
    hook, is refused under A naming the site B wrote (the site A wrote reads fresh) and under B naming the other. Red
    on F-E22 (the stale comparison stopping at the kind's first site)."""
    from mimarsinan.mapping.support.quantizer_state import verify_declared_state
    from mimarsinan.pipelining.pipeline_steps.quantization.declared_state import declare_readout_state

    flow, platform, perceptrons, quantizers, first, second = _two_readout_sites()
    a = _write(tmp_path, {"perceptrons": {first: {"readout_range": 0.625}, second: {"readout_range": 0.75}}}, "a.json")
    b = _write(tmp_path, {"perceptrons": {second: {"readout_range": 0.875}}}, "b.json")
    declare_readout_state(flow, perceptrons, quantizers, {**platform, "quantizer_state": a})
    verify_declared_state(flow, a)
    declare_readout_state(flow, perceptrons, quantizers, {**platform, "quantizer_state": b})
    with pytest.raises(QuantizerStateError, match=rf"\['{second}'\] carry a readout .* from a stale"):
        verify_declared_state(flow, a)
    with pytest.raises(QuantizerStateError, match=rf"\['{first}'\] carry a readout .* from a stale"):
        verify_declared_state(flow, b)


# --- [U-BRANGE2 stage 1e] every guard of the loader refuses its malformed document by name (the sweep's SIBLING kind
# at the loader: F-R19, F-R20, F-R22, F-R26 and their siblings) --------------------------------------------------------

#: A malformed document per guard of the loader, with the words its refusal names.
_MALFORMED_DOCUMENTS = {
    "an offset without its range": ({"perceptrons": {"p": {"readout_offset": 0.5}}},
                                    r"'p' declares a readout_offset without its readout_range"),
    "a lane offset without its range": ({"perceptrons": {}, "lanes": {"pool": {"readout_offset": 0.5}}},
                                        r"lane 'pool' must declare readout_range"),
    "an infinite offset": ({"perceptrons": {"p": {"readout_range": 1.0, "readout_offset": float("inf")}}},
                           r"'p'\.readout_offset must be a finite number"),
    "a boolean offset": ({"perceptrons": {"p": {"readout_range": 1.0, "readout_offset": True}}},
                         r"'p'\.readout_offset must be a finite number"),
    "an infinite lane offset": ({"perceptrons": {}, "lanes": {"pool": {"readout_range": 1.0,
                                                                        "readout_offset": float("inf")}}},
                                r"'pool'\.readout_offset must be a finite number"),
    "banks not a mapping": ({"perceptrons": {}, "banks": [["bank_q", {"readout_range": 1.0}]]},
                            "banks and trained are mappings"),
    "trained not a mapping": ({"perceptrons": {}, "trained": [["g", "readout:p"]]}, "banks and trained are mappings"),
    "perceptrons not a mapping": ({"perceptrons": [["p", {"readout_range": 1.0}]]}, "document holds format"),
    "lanes not a mapping": ({"perceptrons": {}, "lanes": [["pool", {"readout_range": 1.0}]]}, "document holds format"),
    "a section of no kind": ({"perceptrons": {}, "weights": {}}, "document holds format"),
    "a perceptron field of no site": ({"perceptrons": {"p": {"readout_range": 1.0, "weights": 1.0}}},
                                      r"'p' must declare readout_range, entry, groups"),
    "a lane field of no site": ({"perceptrons": {}, "lanes": {"pool": {"readout_range": 1.0, "entry": {"scale": 1.0}}}},
                                r"lane 'pool' must declare readout_range"),
    "another format": ({"format": "mimarsinan.quantizer_state.v0", "perceptrons": {}}, "is not of format"),
}


@pytest.mark.parametrize("case", sorted(_MALFORMED_DOCUMENTS))
def test_every_guard_of_the_loader_refuses_its_malformed_document_by_name(tmp_path, case):
    """[U-BRANGE2 stage 1e, the sweep's SIBLING kind at the loader] Each guard of the map's loader refuses the malformed
    document it guards against by name, at load, before any step reads the map: an offset declared without its range
    (the rule the lane hooks' coverage rests on), a non-finite or boolean offset, a banks, trained, perceptrons or
    lanes section that is no mapping, a section of no kind, a perceptron's or a lane's field of no site, another
    format. Red on F-R19, F-R20, F-R22 and F-R26 (each guard disabled at its site), on F-R49 and F-R50 (the offset
    unjudged at either section), on F-A13 and F-A14 (a field of no site admitted), and on F-R25 with the release's
    own clause."""
    document, refusal = _MALFORMED_DOCUMENTS[case]
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"format": QUANTIZER_STATE_FORMAT, **document}, sort_keys=True))
    with pytest.raises(QuantizerStateError, match=refusal):
        load_quantizer_state({"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})


# --- [U-BRANGE2 stage 1e] the installers' and the hooks' guards the sweep left unpinned (F-R02, F-R04, F-R06) --------


@pytest.mark.parametrize("declared", ["a readout offset", "a pool lane's range"])
def test_lane_grids_declared_on_a_chip_without_executed_lanes_are_refused_by_name(tmp_path, declared):
    """[U-BRANGE2 stage 1e, the sweep's F-R02] A readout offset or a pool lane's range is a lane grid: declared on a
    chip whose lanes do not run, Readout Law's hook refuses it by name (declare onchip_arithmetic with readout_law),
    never installs an offset grid on a readout that is no lane, nor reports the lane as unknown. Red on F-R02 (the
    guard at declared_state.py's lane branch disabled)."""
    from mimarsinan.mapping.support.quantizer_state import perceptron_fx_names
    from mimarsinan.pipelining.pipeline_steps.quantization.declared_state import declare_readout_state

    flow, platform, banks, batches = _lowered(grouped=False)
    perceptrons, quantizers = _calibrated_readouts(flow, platform, banks, batches)
    name = next(iter(perceptron_fx_names(flow.get_mapper_repr())))
    document = ({"perceptrons": {name: {"readout_range": 1.5, "readout_offset": 0.25}}}
                if declared == "a readout offset" else {"perceptrons": {}, "lanes": {"pool": {"readout_range": 1.5}}})
    ref = _write(tmp_path, document)
    with pytest.raises(QuantizerStateError, match="lane grids: declare onchip_arithmetic with readout_law"):
        declare_readout_state(flow, perceptrons, quantizers, {**platform, "quantizer_state": ref})


def test_two_perceptrons_carrying_one_fx_name_are_refused_by_name():
    """[U-BRANGE2 stage 1e, the sweep's F-R04] The map keys a perceptron's sites by its FX node name, so two perceptrons
    under one name would make a declared site ambiguous: ``perceptron_fx_names`` refuses them by name (a graph read
    through a stand-in with two nodes of one name), while one perceptron met twice under its own name is one site."""
    import types

    from mimarsinan.mapping.support.quantizer_state import perceptron_fx_names

    a, b = object(), object()

    def graph(*nodes):
        return types.SimpleNamespace(_ensure_exec_graph=lambda: (list(nodes), None, None))

    with pytest.raises(QuantizerStateError, match="two perceptrons carry the FX node name 'twin'"):
        perceptron_fx_names(graph(types.SimpleNamespace(perceptron=a, name="twin"),
                                  types.SimpleNamespace(perceptron=b, name="twin")))
    assert perceptron_fx_names(graph(types.SimpleNamespace(perceptron=a, name="twin"),
                                     types.SimpleNamespace(perceptron=a, name="twin"))) == {"twin": a}


def test_a_readout_range_for_a_perceptron_without_a_readout_quantizer_is_refused_by_name(tmp_path):
    """[U-BRANGE2 stage 1e, the sweep's F-R06] A declared readout range whose perceptron carries no readout quantizer
    (the perceptron absent from the quantizers handed in) is refused by name, never an ``AttributeError``: the bank
    readout site's refusal (C-B10) at the perceptron's own site."""
    from mimarsinan.mapping.support.quantizer_state import declare_readout_ranges, perceptron_fx_names

    flow, platform, banks, batches = _lowered(grouped=False)
    perceptrons, quantizers = _calibrated_readouts(flow, platform, banks, batches)
    name, target = next(iter(perceptron_fx_names(flow.get_mapper_repr()).items()))
    kept = [(p, q) for p, q in zip(perceptrons, quantizers) if p is not target]
    state = load_quantizer_state(_write(tmp_path, {"perceptrons": {name: {"readout_range": 1.0}}}))
    with pytest.raises(QuantizerStateError, match=rf"readout range for '{name}', which carries no readout quantizer"):
        declare_readout_ranges(flow, [p for p, _ in kept], [q for _, q in kept], state)
