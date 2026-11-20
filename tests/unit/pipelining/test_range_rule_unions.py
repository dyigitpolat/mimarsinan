"""[U-BRANGE2 stage 1, the review's m-3] The grids one reader reads together are one range site of the rule.

The law solvers give every core one input grid: producers read together by one core (a concatenation join, an
operand product's banks, a consumer-grouped producer read whole) share the widest range of their union
(``law_grids.assign_core_grids``). A range chosen per member and then widened is a range coupled to another site's
range, the drive's form: the narrower member's model clamps where its chip does not, so the trained model and the
deployed program part (the attention fixture's banks at their own least-error ranges: up to 8 codes apart, where the
former floor had hidden it). Under the rule such a union is one site: its values pooled over every member, its range
the least squared error of the one grid the members share, and every member takes it. Every error here is read
directly from the grid (``range_rule_reference``).
"""

from __future__ import annotations

import contextlib
import io

import pytest
import torch

from unit.pipelining.range_rule_reference import RELATIVE, grid_error, least_error, signed_grid

try:
    from mimarsinan.mapping.support.range_rule import least_squares_pooled
except ImportError:  # R5 and revision 1: every range site is its own
    least_squares_pooled = None


def _need():
    if least_squares_pooled is None:
        pytest.fail("the rule reads no union of grids read together: each member keeps its own range")


def test_a_pooled_union_is_one_site_fed_every_member_s_values():
    _need()
    from mimarsinan.mapping.support.range_rule import RangeSite, least_squares_ranges

    g = torch.Generator().manual_seed(4)
    a, b = torch.nn.Identity(), torch.nn.Identity()
    model = torch.nn.Sequential(a, b)
    batches = [torch.randn(16, 64, generator=g) * 2.0 for _ in range(2)]
    sites = [RangeSite(a, 127), RangeSite(b, 127, lambda x: (x * 3.0).abs())]
    pooled = least_squares_pooled(model, sites, [[0, 1]], batches)
    assert pooled[0] == pooled[1]
    probe = torch.nn.Identity()
    joined = [torch.cat([x.abs().reshape(-1), (x * 3.0).abs().reshape(-1)]) for x in batches]
    (alone,) = least_squares_ranges(probe, [RangeSite(probe, 127)], joined)
    # the same choice; the error summed over the members' calls (another association than one call) to the last bits
    assert pooled[0].scale == alone.scale and pooled[0].squared_error == pytest.approx(alone.squared_error, rel=1e-9)


def test_members_on_different_grids_stay_apart():
    _need()
    from mimarsinan.mapping.support.range_rule import RangeSite, least_squares_ranges

    a, b = torch.nn.Identity(), torch.nn.Identity()
    model = torch.nn.Sequential(a, b)
    batches = [torch.randn(16, 64, generator=torch.Generator().manual_seed(s)) for s in (1, 2)]
    sites = [RangeSite(a, 127), RangeSite(b, 7)]
    pooled = least_squares_pooled(model, sites, [[0, 1]], batches)
    assert pooled == least_squares_ranges(model, sites, batches)


def test_a_concatenation_join_s_producers_take_one_pooled_range():
    """The drive unit's joined fixture: ``c`` reads ``cat(a, b)``, so ``a`` and ``b`` are one site at Readout Law; ``d``
    (read alone) keeps its own. Red before: ``a`` and ``b`` each took their own range."""
    from mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step import (
        calibrate_readout_quantizers,
        install_readout_quantizers,
    )
    from unit.mapping.test_bias_row_drive_program import _Joined
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron

    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(_Joined().eval(), (16,), 8, device="cpu", packaging=MVM_PACKAGING).eval()
    perceptrons = list(flow.get_perceptrons())
    for perceptron in perceptrons:
        fuse_into_perceptron(perceptron, device="cpu")
    with torch.no_grad():  # b's outputs about three times a's: one shared grid must hold both
        perceptrons[1].layer.weight.mul_(3.0)
    quantizers = install_readout_quantizers(perceptrons, 8)
    by_name = {str(p.name).split(".")[-1].strip("_"): (p, q) for p, q in zip(perceptrons, quantizers)}
    seen = {name: [] for name in ("a", "b")}
    handles = [by_name[name][1].register_forward_pre_hook(
        lambda module, args, name=name: seen[name].append(args[0].detach().clone())) for name in seen]
    batches = [torch.randn(16, 16, generator=torch.Generator().manual_seed(s)) for s in (7, 8)]
    platform = {"activation_bits": 8, "readout_law": {"adc_bits": 32, "mult_bits": 24, "range": "calibrated"}}
    try:
        calibrate_readout_quantizers(flow, perceptrons, quantizers, batches, platform=platform)
    finally:
        for handle in handles:
            handle.remove()
    ra, rb = float(by_name["a"][1].range), float(by_name["b"][1].range)
    assert ra == rb, (ra, rb)
    pooled = [torch.cat([x.abs().reshape(-1) for x in seen[name][:2]]) for name in ("a", "b")]
    pooled = [torch.cat([pooled[0], pooled[1]])]
    assert grid_error(pooled, ra, signed_grid(8)) <= least_error(pooled, signed_grid(8)) * (1 + RELATIVE)


def test_an_operand_product_s_banks_read_by_one_core_take_one_pooled_range():
    """The attention fixture: a product whose banks one core reads (P.V into the projection) is one site, so its banks
    take one range; a product a host op reads (Q.K into the softmax) keeps a grid per bank. Red before: each bank its
    own (the fixture's matmul_3 banks 2.0e-4 and 3.4e-4 under the rule without the union)."""
    try:
        from mimarsinan.mapping.support.reader_grids import read_together
    except ImportError:
        pytest.fail("no reader of the grids read together on this head")
    from unit.chip_simulation.value_run.operand_programs import attention_program

    model = attention_program("relu").nf()
    unions = read_together(model.get_mapper_repr())
    banks = 0
    for union in unions:
        ranges = {float(q.group_ranges[g]) if q.channel_groups is not None else float(q.range) for q, g in union}
        assert len(ranges) == 1, sorted(ranges)
        banks += sum(1 for q, _g in union if any(q is op.scaler for op in _banks(model)))
    assert banks >= 4  # two products read by a core, two banks each


def _banks(model):
    from mimarsinan.mapping.mappers.operand import operand_perceptrons

    return operand_perceptrons(model.get_mapper_repr())


# --- [U-BRANGE2 stage 1b] every install site takes the rule's range: the bank payload and the executed lanes ----------
# The stage's verify: M-2 (VM18, a bank payload grid installs the calibration maximum) and M-3 (VM19, the executed-lanes
# path installs the calibration maximum) survived every clause. The first two clauses are the verify's own
# (probe/test_v_sites.py), adopted; the third reads the step itself on the lanes path, where the step installs them.


def _capture(monkeypatch):
    import mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step as readout_step

    captured = []
    rule = readout_step.least_squares_pooled

    def recording(model, sites, unions, batches, **kwargs):
        rows = rule(model, sites, unions, batches, **kwargs)
        captured.append((list(sites), list(rows)))
        return rows

    monkeypatch.setattr(readout_step, "least_squares_pooled", recording)
    return captured


def test_a_bank_payload_grid_takes_the_rule_s_range(monkeypatch):
    """Every re-snapped bank's payload grid is calibrated at its census scale, and the fixture holds one whose rule
    range is below its calibration maximum (so the clause tells the rule from the maximum). Red on VM18."""
    _need()
    import mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step as readout_step
    from mimarsinan.mapping.mappers.operand import operand_perceptrons
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.mapping.support.operand_lowering import lower_contractions
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from unit.chip_simulation.value_run.operand_programs import attention_block, operand_platform

    platform = operand_platform(payload_bits=4, grouped=False)
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(attention_block(), (3, 8, 8), 5, device="cpu", packaging=MVM_PACKAGING).eval()
    lower_contractions(flow.get_mapper_repr(), platform)
    banks = [op for op in operand_perceptrons(flow.get_mapper_repr()) if op.payload_quantizer is not None]
    assert banks, "fixture: a re-snapped bank"
    captured = _capture(monkeypatch)
    perceptrons = list(flow.get_perceptrons()) + banks
    quantizers = readout_step.install_readout_quantizers(perceptrons, 8)
    batches = [torch.randn(4, 3, 8, 8, generator=torch.Generator().manual_seed(s)) for s in (5, 6)]
    readout_step.calibrate_readout_quantizers(flow, perceptrons, quantizers, batches, platform=platform)
    (sites, rows), = captured
    census = {id(s.module): r for s, r in zip(sites, rows)}
    payload = [op.payload_quantizer for op in banks]
    assert any(census[id(q)].scale != census[id(q)].maximum for q in payload), "fixture: a payload below its maximum"
    for q in payload:
        assert float(q.grid.scale) == census[id(q)].scale, (float(q.grid.scale), census[id(q)].scale)


def _lane_flow():
    from mimarsinan.mapping.lanes.model_lanes import lane_absorbed
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import (
        calibrate_boundary_scales,
        install_boundary_quantizers,
    )
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from mimarsinan.torch_mapping.encoding_layers import segment_entry_perceptrons
    from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron
    from unit.chip_simulation.value_run.integer_programs import residual_block_net
    from unit.mapping.lanes.lane_programs import lane_platform

    platform = lane_platform(("L", "E", "P"), activation_bits=4)
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(residual_block_net().eval(), (3, 8, 8), 10, device="cpu",
                                   packaging=MVM_PACKAGING).eval()
    for perceptron in flow.get_perceptrons():
        fuse_into_perceptron(perceptron, device="cpu")
    batches = [torch.randn(32, 3, 8, 8, generator=torch.Generator().manual_seed(1))]
    absorbed = lane_absorbed(flow, platform)
    entries = list(segment_entry_perceptrons(flow.get_mapper_repr(), absorbed))
    calibrate_boundary_scales(flow, entries, install_boundary_quantizers(entries, 4), batches)
    return flow, platform, batches, absorbed


def test_the_executed_lanes_path_takes_the_rule_s_ranges(monkeypatch):
    """Readout Law's lanes path returns each readout's and lane's census scale, and the fixture holds one below its
    maximum. Red on VM19."""
    _need()
    import mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step as readout_step
    from mimarsinan.mapping.lanes.model_lanes import install_planned_lanes

    flow, platform, batches, absorbed = _lane_flow()
    perceptrons = list(flow.get_perceptrons())
    quantizers = readout_step.install_readout_quantizers(perceptrons, 4)
    lanes = install_planned_lanes(flow, platform, 4) or []
    assert lanes, "fixture: executed lanes"
    every = quantizers + [q for _, q in lanes]
    captured = _capture(monkeypatch)
    values = readout_step.ReadoutLawStep._calibrated(flow, perceptrons, every, batches, lanes, absorbed)
    (_sites, rows), = captured
    assert any(r.scale != r.maximum for r in rows), "fixture: a lane or readout below its maximum"
    assert values == [r.scale for r in rows]


def test_the_step_installs_the_rule_s_range_on_every_readout_and_lane(monkeypatch):
    """The step itself on the lanes path (its trainer and batches stubbed): every readout and every lane quantizer
    holds its census scale after the step (the sites in the step's own order: the readouts, then the lanes). Red on
    VM19 and on any install of the step's lanes branch that is not the rule's (``ReadoutLawStep.process``)."""
    _need()
    from types import SimpleNamespace

    import mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step as readout_step

    flow, platform, batches, _absorbed = _lane_flow()
    monkeypatch.setattr(readout_step, "make_basic_trainer", lambda pipeline, m: SimpleNamespace(test=lambda: 0.0))
    monkeypatch.setattr(readout_step, "iter_val_batches", lambda trainer, n: [(x, None) for x in batches])
    installed = []
    plan = readout_step.install_planned_lanes

    def recording_lanes(model, config, bits):
        lanes = plan(model, config, bits)
        installed.append(lanes)
        return lanes

    monkeypatch.setattr(readout_step, "install_planned_lanes", recording_lanes)
    captured = _capture(monkeypatch)
    config = {"core_semantics": "mvm", "device": "cpu", "weight_quantization": True, **platform}

    class _Pipeline:
        def __init__(self):
            self.config = config
            self.entries = {"model": flow, "platform_constraints_resolved": platform}

        def get_entry(self, _step, key):
            return self.entries[key]

        def add_entry(self, _step, key, value, _strategy="basic"):
            self.entries[key] = value

        def update_entry(self, _step, key, value, _strategy="basic"):
            self.entries[key] = value

    with contextlib.redirect_stdout(io.StringIO()):
        readout_step.ReadoutLawStep(_Pipeline()).run()
    (lanes,) = installed
    assert lanes, "fixture: executed lanes"
    every = [p.scaler for p in flow.get_perceptrons()] + [q for _, q in lanes]
    (_sites, rows), = captured
    assert len(rows) == len(every) and any(r.scale != r.maximum for r in rows), "fixture: a site below its maximum"
    assert [float(q.range) for q in every] == [r.scale for r in rows]


def test_a_signed_member_and_an_unsigned_member_of_one_width_stay_apart():
    """Sweep SW15: a union's members are one site only on one grid, its levels and its form: a signed site (a window
    whose low end is searched) and a site of magnitudes at the same levels are not pooled."""
    _need()
    from mimarsinan.mapping.support.range_rule import RangeSite, least_squares_ranges

    a, b = torch.nn.Identity(), torch.nn.Identity()
    model = torch.nn.Sequential(a, b)
    batches = [torch.randn(16, 64, generator=torch.Generator().manual_seed(s)) + 0.5 for s in (1, 2)]
    sites = [RangeSite(a, 15), RangeSite(b, 15, lambda x: x, signed=True)]
    pooled = least_squares_pooled(model, sites, [[0, 1]], batches)
    assert pooled == least_squares_ranges(model, sites, batches)
    assert pooled[0] != pooled[1]


def test_the_lanes_path_pools_the_grids_one_reader_reads_together():
    """Sweep SW10: the lanes path (lanes declared, Readout Law's ``_calibrated``) pools a union as the readouts' path
    does: the drive unit's joined fixture (``c`` reads ``cat(a, b)``, ``b``'s outputs about three times ``a``'s) under a
    clamp-lane declaration gives ``a`` and ``b`` one range, where each alone would take another."""
    _need()
    import mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step as readout_step
    from mimarsinan.mapping.lanes.model_lanes import install_planned_lanes, lane_absorbed, lane_site
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.mapping.support.range_rule import least_squares_ranges
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron
    from unit.mapping.lanes.lane_programs import lane_platform
    from unit.mapping.test_bias_row_drive_program import _Joined

    platform = lane_platform(("L",), activation_bits=8)
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(_Joined().eval(), (16,), 8, device="cpu", packaging=MVM_PACKAGING).eval()
    perceptrons = list(flow.get_perceptrons())
    for perceptron in perceptrons:
        fuse_into_perceptron(perceptron, device="cpu")
    with torch.no_grad():
        perceptrons[1].layer.weight.mul_(3.0)
    quantizers = readout_step.install_readout_quantizers(perceptrons, 8)
    lanes = install_planned_lanes(flow, platform, 8)
    assert lanes is not None, "fixture: the lanes path"
    every = quantizers + [q for _, q in lanes]
    batches = [torch.randn(64, 16, generator=torch.Generator().manual_seed(s)) for s in (3, 4)]
    values = readout_step.ReadoutLawStep._calibrated(flow, perceptrons, every, batches, lanes,
                                                     lane_absorbed(flow, platform))
    index = {str(p.name).split(".")[-1].strip("_"): i for i, p in enumerate(perceptrons)}
    a, b = index["a"], index["b"]
    assert values[a] == values[b], (values[a], values[b])
    alone = least_squares_ranges(flow, [lane_site(quantizers[a]), lane_site(quantizers[b])], batches)
    assert alone[0].scale != alone[1].scale, "fixture: a and b alone take different ranges"


def test_on_the_lanes_path_a_declared_fixed_range_is_every_readout_s_and_lane_s(monkeypatch):
    """Sweep SW11: the step's lanes path under a declared ``range_value`` installs it on every readout and lane, and no
    site is searched."""
    _need()
    from types import SimpleNamespace

    import mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step as readout_step

    flow, platform, batches, _absorbed = _lane_flow()
    platform = {**platform, "readout_law": {**platform["readout_law"], "range_value": 1.25}}
    monkeypatch.setattr(readout_step, "make_basic_trainer", lambda pipeline, m: SimpleNamespace(test=lambda: 0.0))
    monkeypatch.setattr(readout_step, "iter_val_batches", lambda trainer, n: [(x, None) for x in batches])
    installed = []
    plan = readout_step.install_planned_lanes

    def recording_lanes(model, config, bits):
        lanes = plan(model, config, bits)
        installed.append(lanes)
        return lanes

    monkeypatch.setattr(readout_step, "install_planned_lanes", recording_lanes)
    captured = _capture(monkeypatch)
    config = {"core_semantics": "mvm", "device": "cpu", "weight_quantization": True, **platform}

    class _Pipeline:
        def __init__(self):
            self.config = config
            self.entries = {"model": flow, "platform_constraints_resolved": platform}

        def get_entry(self, _step, key):
            return self.entries[key]

        def add_entry(self, _step, key, value, _strategy="basic"):
            self.entries[key] = value

        def update_entry(self, _step, key, value, _strategy="basic"):
            self.entries[key] = value

    with contextlib.redirect_stdout(io.StringIO()):
        readout_step.ReadoutLawStep(_Pipeline()).run()
    (lanes,) = installed
    assert lanes, "fixture: executed lanes"
    every = [p.scaler for p in flow.get_perceptrons()] + [q for _, q in lanes]
    assert all(float(q.range) == 1.25 for q in every)
    assert not captured, "a site was searched under a declared fixed range"


def test_the_step_s_unions_read_the_lane_absorbed_nodes_of_its_declaration(monkeypatch):
    """Sweep SW13: the step reads its unions through the host ops its declared lanes absorb (an absorbed clamp passes
    its producer's lane grid to its reader), the set ``lane_absorbed`` gives for the step's config: on the residual
    block with lanes declared, the step hands that set (not empty) to the union source."""
    _need()
    from types import SimpleNamespace

    import mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step as readout_step
    from mimarsinan.mapping.lanes.model_lanes import lane_absorbed

    flow, platform, batches, absorbed = _lane_flow()
    assert absorbed, "fixture: lanes absorb host ops"
    monkeypatch.setattr(readout_step, "make_basic_trainer", lambda pipeline, m: SimpleNamespace(test=lambda: 0.0))
    monkeypatch.setattr(readout_step, "iter_val_batches", lambda trainer, n: [(x, None) for x in batches])
    read = []
    source = readout_step.read_together

    def recording(model_repr, *, absorbed=frozenset()):
        read.append(absorbed)
        return source(model_repr, absorbed=absorbed)

    monkeypatch.setattr(readout_step, "read_together", recording)
    config = {"core_semantics": "mvm", "device": "cpu", "weight_quantization": True, **platform}

    class _Pipeline:
        def __init__(self):
            self.config = config
            self.entries = {"model": flow, "platform_constraints_resolved": platform}

        def get_entry(self, _step, key):
            return self.entries[key]

        def add_entry(self, _step, key, value, _strategy="basic"):
            self.entries[key] = value

        def update_entry(self, _step, key, value, _strategy="basic"):
            self.entries[key] = value

    with contextlib.redirect_stdout(io.StringIO()):
        readout_step.ReadoutLawStep(_Pipeline()).run()
    assert read and all(r == lane_absorbed(flow, config) == absorbed for r in read), (read, absorbed)


def test_a_reader_of_host_values_and_core_outputs_makes_no_union():
    """Sweep SW60: a union is the producer grids a core reads on one DAC grid; a reader of host values and core outputs
    has no one grid (the law solver refuses it by name), so its producers stay separate sites: ``m`` reads
    ``cat(a(x), b(x), sigmoid(x))`` and no union is read; ``c`` reading ``cat(a(x), b(x))`` alone makes one."""
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.mapping.support.reader_grids import read_together
    from mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step import install_readout_quantizers
    from mimarsinan.torch_mapping.converter import convert_torch_model

    class _MixedTwo(torch.nn.Module):
        def __init__(self, host: bool):
            super().__init__()
            self.host = host
            self.a, self.b = torch.nn.Linear(16, 6), torch.nn.Linear(16, 6)
            self.m = torch.nn.Linear(28 if host else 12, 4)

        def forward(self, x):
            parts = [self.a(x), self.b(x)] + ([torch.sigmoid(x)] if self.host else [])
            return self.m(torch.cat(parts, dim=1))

    for host, unions in ((True, 0), (False, 1)):
        torch.manual_seed(0)
        with contextlib.redirect_stdout(io.StringIO()):
            flow = convert_torch_model(_MixedTwo(host).eval(), (16,), 4, device="cpu", packaging=MVM_PACKAGING).eval()
        install_readout_quantizers(list(flow.get_perceptrons()), 8)
        assert len(read_together(flow.get_mapper_repr())) == unions, host


# --- [U-BRANGE2 stage 1c] the lanes path refuses a zero site by name, every site named (the delta verify's M-2) -------


def _zeroed(flow, perceptrons):
    """Every listed perceptron's weight and bias set to 0 (its readout's calibration values all zero)."""
    with torch.no_grad():
        for perceptron in perceptrons:
            perceptron.layer.weight.zero_()
            if perceptron.layer.bias is not None:
                perceptron.layer.bias.zero_()
    return flow


def test_the_lanes_path_refuses_a_zero_site_by_name():
    """[U-BRANGE2 stage 1c, the delta verify's M-2] On the lanes path a readout whose calibration values are all zero
    is refused by name (``ReadoutLawCalibrationError``), never installed at 0 (an identity output), as the
    per-perceptron path refuses it. Red on X11 (the lanes path's refusal removed)."""
    _need()
    from mimarsinan.mapping.lanes.model_lanes import install_planned_lanes
    from mimarsinan.pipelining.pipeline_steps.quantization import readout_law_step as readout_step

    flow, platform, batches, absorbed = _lane_flow()
    perceptrons = list(flow.get_perceptrons())
    dead = perceptrons[-1]
    _zeroed(flow, [dead])
    quantizers = readout_step.install_readout_quantizers(perceptrons, 4)
    lanes = install_planned_lanes(flow, platform, 4) or []
    assert lanes, "fixture: executed lanes"
    every = quantizers + [q for _, q in lanes]
    with pytest.raises(readout_step.ReadoutLawCalibrationError, match=str(dead.name)):
        readout_step.ReadoutLawStep._calibrated(flow, perceptrons, every, batches, lanes, absorbed)


def test_the_lanes_path_names_every_zero_site_its_lanes_included():
    """[U-BRANGE2 stage 1c, sweeps C-L2 and C-L3] The lanes path's refusal names each zero site by its own name, in the
    sites' own order: the perceptrons' readouts, then the lanes by their FX names. With every perceptron's weights and
    biases at 0, every readout and every lane end reads only zeros, and the refusal names every one of them."""
    _need()
    from mimarsinan.mapping.lanes.model_lanes import install_planned_lanes
    from mimarsinan.pipelining.pipeline_steps.quantization import readout_law_step as readout_step

    flow, platform, batches, absorbed = _lane_flow()
    perceptrons = list(flow.get_perceptrons())
    _zeroed(flow, perceptrons)
    quantizers = readout_step.install_readout_quantizers(perceptrons, 4)
    lanes = install_planned_lanes(flow, platform, 4) or []
    assert lanes, "fixture: executed lanes"
    every = quantizers + [q for _, q in lanes]
    with pytest.raises(readout_step.ReadoutLawCalibrationError) as caught:
        readout_step.ReadoutLawStep._calibrated(flow, perceptrons, every, batches, lanes, absorbed)
    message = str(caught.value)
    names = [str(p.name) for p in perceptrons] + [str(name) for name, _ in lanes]
    assert all(f"'{name}'" in message or name in message for name in names), (names, message)
    positions = [message.find(name) for name in names]
    assert positions == sorted(positions), (names, message)


# --- [U-BRANGE2 stage 1c] the invariants two of stage 1b's equivalence arguments rest on, pinned in the unit (the
# delta verify's n-4: stage 1b's arguments 1 and 10) -------------------------------------------------------------------


def test_the_resolved_platform_carries_the_lane_declaration_and_the_readout_law_verbatim():
    """[U-BRANGE2 stage 1c, n-4 for SW14] The step reads its lanes from the resolved platform and its absorbed nodes
    from the config; the two are one declaration because the resolver carries ``onchip_arithmetic`` and
    ``readout_law`` verbatim (registry passthrough keys), so the lanes, the absorbed host ops and whether lanes run are
    the same read from either."""
    from mimarsinan.mapping.lanes.declaration import LANE_KEY, lanes_executed
    from mimarsinan.mapping.lanes.model_lanes import lane_absorbed
    from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved

    flow, platform, _batches, absorbed = _lane_flow()
    config = {"core_semantics": "mvm", "device": "cpu", "weight_quantization": True, **platform}
    resolved = build_platform_constraints_resolved(config)
    assert resolved.get(LANE_KEY) == config[LANE_KEY] and resolved.get("readout_law") == config["readout_law"]
    assert lanes_executed(resolved) == lanes_executed(config) is not None
    assert lane_absorbed(flow, resolved) == lane_absorbed(flow, config) == absorbed and absorbed


def test_readout_law_installs_every_readout_and_lane_at_the_step_s_one_width():
    """[U-BRANGE2 stage 1c, n-4 for SW58] Every readout and every lane quantizer Readout Law installs reads the step's
    one ``activation_bits``, so every ungrouped key a union joins reads one level count (``readout_key_grid``): an
    ungrouped join of two widths is never built, which the union refusal's grouped condition relies on."""
    from mimarsinan.mapping.export.law_grids import readout_grid_keys, readout_key_grid
    from mimarsinan.mapping.lanes.model_lanes import install_planned_lanes
    from mimarsinan.models.nn.activations.integer_law import grid_levels
    from mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step import install_readout_quantizers

    flow, platform, _batches, _absorbed = _lane_flow()
    bits = 4
    quantizers = install_readout_quantizers(list(flow.get_perceptrons()), bits)
    lanes = install_planned_lanes(flow, platform, bits) or []
    assert lanes, "fixture: pool lanes"
    every = quantizers + [q for _, q in lanes]
    assert {int(q.activation_bits) for q in every} == {bits}
    counts = {readout_key_grid(q, g)[1] for q in every for _id, g in readout_grid_keys(q)}
    assert counts == {grid_levels(bits)}, counts
