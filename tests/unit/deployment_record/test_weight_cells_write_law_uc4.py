"""[U-C4, B3 intake open item 4] Cells per stored weight in the write law, the chip's physical quantities and validity.

A weight write writes every cell of the weight: on NeuRRAM (two RRAM devices per signed weight, the differential pair
of wan2022a Methods, PDF p.10; ``G[0::2, :] = (W>0) * W * scale`` / ``G[1::2, :] = -((W<0) * W) * scale`` in the code
release's ``python/nn_layers.py:33-37``) one written weight is two written cells, so its write energy is
2 x (e_write_per_cell + e_verify_read_per_cell) x 8.52 pulses per device, the B3 intake's 91.5 / 1,017 / 11,362 pJ per
signed weight (``fetch_intake/constants_write_energy_b3.md`` sec. 2.3), and at one cell per write step it takes two
write steps. The wear of a cell is unchanged: both cells of a weight take each of its writes, so the most-written
cell's writes per inference, and the lifetime, are the weight's. The chip's physical quantities (cells, row and column
slots: area, row drivers, converters, static per-slot rows) and the core the aggregates' validity is judged at count
the array's cells, which on every chip of one cell per weight are its weights, unchanged.
"""

import math
from types import SimpleNamespace

import pytest

from mimarsinan.deployment_record.build.write_census import PassRow, ProgramPasses
from mimarsinan.deployment_record.cost.absolute import price_write_terms
from mimarsinan.deployment_record.cost.absolute.validity import CoreGeometry, geometry_of
from mimarsinan.deployment_record.design_point.timeline import (
    ProgramWrites, TimelineError, pass_timeline, program_writes, writes_summary,
)
from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform
from mimarsinan.deployment_record.objectives.views import chip_param_capacity
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.quantities.from_candidate import from_candidate
from mimarsinan.deployment_record.quantities.from_level import LevelViewError, chip_facts, level_chip
from mimarsinan.deployment_record.quantities.from_record import from_record
from mimarsinan.deployment_record.quantities.write_law import (
    WritePolicy, pass_units, plane_write_census, record_write_census, write_policy_of, write_quantities,
)
from mimarsinan.deployment_record.quantities.write_technology import TechnologySplit
from mimarsinan.mapping.platform.target_bundles import target_bundles
from mimarsinan.mapping.support.period_census import PassCensus, PassPeriodCensus

from unit.deployment_record.g3_fixtures import record_with, schedule, segment

#: The B3 band per pulse into one RRAM device and its verify read, pJ (the U-C3 rows of neurram.json).
E_WRITE = (5.328, math.sqrt(5.328 * 650.8), 650.8)
E_VERIFY = (0.04, 0.8, 16.0)
PULSES = 8.52
#: One signed weight: its two devices, each (e_write + e_verify) x 8.52, pJ (the intake's 91.5 / 1,017 / 11,362).
PER_WEIGHT_PJ = tuple(2 * (w + v) * PULSES for w, v in zip(E_WRITE, E_VERIFY))
#: The write step of one cell: t_write_per_unit x write_verify_iterations, us (neurram.json: 2 / 6.5728 / 11 us).
STEP_US = tuple(t * PULSES for t in (2.0, 6.5728, 11.0))


def _neurram(**platform):
    return {**target_bundles()["neurram"].values, "inference_batch_K": 1, **platform}


def _census(cells=1, producer=0):
    return PassPeriodCensus(passes=(PassCensus(position=0, segment_index=0, resident=False, cores=1, resident_cores=0,
                                               written_cells_backing_store=cells, written_cells_producer=producer,
                                               payload_bytes=0, compute_steps=1),))


def _terms(pricing):
    return {t.name: t.band for t in pricing.terms}


def _corners(band):
    return band.low, band.nominal, band.high


class TestPolicy:
    def test_the_neurram_policy_reads_two_cells_on_two_rows(self):
        policy = write_policy_of(_neurram())
        assert (policy.cells_per_weight, policy.weight_cell_rows) == (2, 2)
        assert policy.core_technologies is None and policy.mixed_core_writes is None

    def test_a_chip_of_one_cell_per_weight_reads_one(self):
        policy = write_policy_of({**target_bundles()["hermes_64"].values, "inference_batch_K": 1})
        assert (policy.cells_per_weight, policy.weight_cell_rows) == (1, None)
        assert (WritePolicy(batch_k=1).cells_per_weight, WritePolicy(batch_k=1).weight_cell_rows) == (1, None)


class TestWrittenCells:
    def test_one_written_weight_writes_two_cells(self):
        values = write_quantities(_census(cells=1), write_policy_of(_neurram()))
        assert values["runtime_written_cells"].value == 2.0
        assert values["written_units"].value == 2.0 and values["write_events"].value == 2.0

    def test_operand_writes_count_cells_too(self):
        values = write_quantities(_census(cells=0, producer=3), write_policy_of(_neurram()))
        assert values["runtime_written_cells"].value == 6.0 and values["write_events"].value == 6.0

    def test_the_period_census_of_two_rotating_passes(self):
        """Two passes of 3 and 5 weights rotate on the 48 cores: 16 cells written per period, and 2 + 3 payload bytes
        of 4-bit weights (the payload counts weights, never cells)."""
        rows = tuple(PassRow(segment_index=i, resident=False, core_cells=(n,), marks=frozenset(),
                             core_extents=((n, 1),), compute_steps=1, core_types=((128, 256),))
                     for i, n in enumerate((3, 5)))
        platform = _neurram()
        census, reason = plane_write_census(ProgramPasses(rows=rows), platform, weight_bits=4, timesteps=None,
                                            label="test")
        assert reason is None
        values = write_quantities(census.census, write_policy_of(platform), written_extents=census.written_extents)
        assert values["runtime_written_cells"].value == 16.0
        assert values["period_rewrite_bytes"].value == 5.0

    @pytest.mark.parametrize("granularity, count, rows, units", [
        ("cell", 2, 2, (14,)), ("row", 2, 2, (6,)), ("column", 2, 2, (2,)),
        ("row", 2, 1, (3,)), ("column", 2, 1, (4,)), ("core", 2, 2, (1,)),
        ("cell", 4, 2, (28,)), ("row", 4, 2, (6,)), ("column", 4, 2, (4,)),
    ])
    def test_the_write_units_read_the_block(self, granularity, count, rows, units):
        """One rewritten core of 7 weights over 3 used weight rows and 2 used weight columns: a cell write per cell
        (7 x the count); a pair on two rows takes two cell rows per weight row and one cell column per weight column, a
        pair on one row the other way round, a 2 x 2 block two of each; a core write is one unit whatever its cells."""
        census = PassPeriodCensus(passes=(PassCensus(position=0, segment_index=0, resident=False, cores=1,
                                                     resident_cores=0, written_cells_backing_store=7,
                                                     written_cells_producer=0, payload_bytes=0, compute_steps=1),))
        policy = WritePolicy(batch_k=1, write_granularity=granularity, cells_per_weight=count, weight_cell_rows=rows)
        assert pass_units(census, policy, ((((3, 2),),))) == units


class TestWriteEnergy:
    def test_one_written_weight_at_the_declared_band(self):
        band = _terms(price_write_terms(_census(cells=1), get_platform_physics("neurram"),
                                        write_policy_of(_neurram())))["energy_write_mj"]
        assert _corners(band) == pytest.approx(tuple(p * 1e-9 for p in PER_WEIGHT_PJ), rel=1e-12)
        assert PER_WEIGHT_PJ == pytest.approx((91.47072, 1017.0351744039082, 11362.272), rel=1e-12)
        assert "cells_per_weight 2" in band.basis

    def test_one_written_operand_weight_per_inference(self):
        band = _terms(price_write_terms(_census(cells=0, producer=1), get_platform_physics("neurram"),
                                        write_policy_of(_neurram(inference_batch_K=4))))["energy_write_mj"]
        assert band.nominal == pytest.approx(PER_WEIGHT_PJ[1] * 1e-9, rel=1e-12)

    def test_a_chip_of_one_cell_per_weight_prices_as_before(self):
        band = _terms(price_write_terms(_census(cells=1), get_platform_physics("neurram"),
                                        WritePolicy(batch_k=1)))["energy_write_mj"]
        assert band.nominal == pytest.approx(PER_WEIGHT_PJ[1] / 2 * 1e-9, rel=1e-12)
        assert band.basis == "e_write_per_cell x runtime_written_cells x write_verify_iterations"


class TestWriteTime:
    def test_one_written_weight_takes_two_cell_write_steps(self):
        band = _terms(price_write_terms(_census(cells=1), get_platform_physics("neurram"),
                                        write_policy_of(_neurram())))["latency_write_exposed_s"]
        assert _corners(band) == pytest.approx(tuple(2 * s * 1e-6 for s in STEP_US), rel=1e-12)


class TestWear:
    def test_each_cell_of_the_pair_takes_every_write_of_its_weight(self):
        """The lifetime reads the most-written cell: both cells of a weight take each of its writes, so the pair
        wears each device as one cell per weight would."""
        platform = _neurram(inference_rate_hz=1.0)
        physics = get_platform_physics("neurram")
        pair = price_write_terms(_census(cells=1), physics, write_policy_of(platform), most_written=1)
        one = price_write_terms(_census(cells=1), physics,
                                WritePolicy(batch_k=1, write_granularity="cell", inference_rate_hz=1.0),
                                most_written=1)
        for name in ("lifetime_inferences", "lifetime_hours"):
            assert _corners(_terms(pair)[name]) == _corners(_terms(one)[name])
        values = write_quantities(_census(cells=1), write_policy_of(platform), most_written=1)
        assert values["max_writes_per_cell_per_inference"].value == 1.0


class TestTechnologies:
    def test_core_types_that_differ_in_cells_per_weight_are_priced_apart(self):
        platform = {"inference_batch_K": 1, "cores": [
            {"max_axons": 128, "max_neurons": 256, "count": 2, "write_granularity": "cell", "cells_per_weight": 2,
             "weight_cell_rows": 2},
            {"max_axons": 64, "max_neurons": 64, "count": 2, "write_granularity": "cell"}]}
        policy = write_policy_of(platform)
        assert policy.core_technologies is not None
        assert sorted((t.cells_per_weight, t.weight_cell_rows or 0) for t in policy.core_technologies) == [
            (1, 0), (2, 2)]


#: [targeted M2] A chip of two core types that store a weight in different numbers of cells: 128 x 256 weights of two
#: cells each (a pair on two rows) beside 64 x 64 weights of one cell, both written a cell at a time.
_PAIR, _ONE = (128, 256), (64, 64)
_MIXED_CORES = (
    {"max_axons": 128, "max_neurons": 256, "count": 2, "write_granularity": "cell", "cells_per_weight": 2,
     "weight_cell_rows": 2},
    {"max_axons": 64, "max_neurons": 64, "count": 2, "write_granularity": "cell"})


def _mixed(*cores, **platform):
    return {"cores": [dict(c) for c in (cores or _MIXED_CORES)], "weight_bits": 4, "simulation_steps": 32,
            "inference_batch_K": 1, **platform}


def _mixed_rows(producer=0, types=(_PAIR, _ONE)):
    """Two writing passes, which rotate: 10 weights on a pair core, then 3 on a one-cell core that also takes
    ``producer`` operand cells per inference; ``types`` None states no core type."""
    def row(i, cells, kind, produced):
        return PassRow(segment_index=i, resident=False, core_cells=(cells,), marks=frozenset(),
                       core_extents=((cells, 1),), compute_steps=1, stage_latencies=(0,),
                       core_types=None if kind is None else (kind,), producer_cells=produced,
                       producer_cores=frozenset({0}) if produced else frozenset())
    kinds = types or (None, None)
    return ProgramPasses(rows=(row(0, 10, kinds[0], 0), row(1, 3, kinds[1], producer)))


def _candidate(platform, passes):
    context = candidate_context_from_platform(platform, passes=passes)
    return context.writes, from_candidate(layout=None, chip_param_capacity=None, total_params=None,
                                          host_side_segment_count=None, context=context)


def _parts(split):
    """Each write technology's share of the census at its own count: what the write terms are priced on."""
    return [write_quantities(p.census, p.policy, written_extents=p.written_extents) for p in split.parts]


def _summed(parts, key):
    return sum(q[key].value for q in parts)


def _hybrid():
    """The built program of ``_mixed_rows()``: two passes of one segment, one core each, of the two core types."""
    def stage(q, cells, kind):
        core = SimpleNamespace(axons_per_core=kind[0], neurons_per_core=kind[1], available_axons=kind[0] - cells,
                               available_neurons=kind[1] - 1, unusable_space=0, latency=0)
        return SimpleNamespace(kind="neural", schedule_segment_index=0, schedule_pass_index=q, input_map=[],
                               output_map=[], hard_core_mapping=SimpleNamespace(
                                   cores=[core], soft_core_placements_per_hard_core=[[]]))
    return SimpleNamespace(stages=[stage(0, 10, _PAIR), stage(1, 3, _ONE)])


class TestCoreTypesOfDifferentCounts:
    """[targeted M2] Every written-cell quantity counts each core type's cells: on a chip whose core types store a
    weight in different numbers of cells, the record, candidate and level planes and the design-point timeline state
    the written cells, write units and write events as the sum of the write technologies' shares, each at its own
    count, the shares the write terms are priced on (``cost/absolute/writes.py``); where no such split is stated they
    are withheld by name, never counted at one cell per weight."""

    def test_the_record_plane(self):
        record = record_with(schedule((segment(1, 0, 0, [10], types=[_PAIR]), segment(2, 0, 1, [3], types=[_ONE]))),
                             cores=[dict(c) for c in _MIXED_CORES], inference_batch_K=1)
        quantities = from_record(record)
        parts = _parts(record_write_census(record).by_technology)
        for key in ("runtime_written_cells", "written_units", "write_events"):
            assert quantities.get(key).value == _summed(parts, key) == 10 * 2 + 3, key

    def test_the_candidate_plane_with_operand_cells(self):
        writes, quantities = _candidate(_mixed(), _mixed_rows(producer=5))
        parts = _parts(writes.by_technology)
        for key in ("runtime_written_cells", "written_units", "write_events"):
            assert quantities.get(key).value == _summed(parts, key) == 10 * 2 + 3 + 5, key
        assert quantities.get("period_rewrite_bytes").value == 5 + 2  # payload bytes count weights: 4-bit, 10 and 3

    def test_the_level_plane_prices_one_core_type(self):
        """A level is priced on one declared core type (``level_chip``), so it never carries two counts."""
        with pytest.raises(LevelViewError, match="one declared core type"):
            level_chip(_mixed(), 4)

    def test_the_timeline(self):
        hybrid = _hybrid()
        writes = program_writes(hybrid, _mixed(), weight_bits=4)
        summary = writes_summary(writes)
        parts = _parts(writes.by_technology)
        assert summary["static_written_cells_per_period"] == _summed(parts, "runtime_written_cells") == 23
        assert summary["write_events_per_inference"] == _summed(parts, "write_events") == 23
        passes = [e for e in pass_timeline(hybrid, writes, {}) if e["kind"] == "pass"]
        assert [e["static_written_cells_per_period"] for e in passes] == [20, 3]
        assert [e["write_units_per_period"] for e in passes] == [20, 3]
        assert [e["write_events_per_period"] for e in passes] == [20, 3]

    def test_a_census_split_by_no_technology_withholds_the_written_cells_by_name(self):
        """Rows that state no core type on a chip of two counts: the split refuses, so no count is known."""
        _, quantities = _candidate(_mixed(), _mixed_rows(types=None))
        assert not quantities.has("runtime_written_cells")
        assert "different numbers of cells" in quantities.withheld_reason("runtime_written_cells")

    def test_shares_that_miss_a_written_weight_withhold_the_written_cells_by_name(self):
        """A fused core (two 128 x 256 cores, axons summed) is of no declared type: the split by technology does
        not hold its weights, so the shares count none of its cells, and the plane withholds them by name."""
        _, quantities = _candidate(_mixed(), _mixed_rows(types=((256, 256), _ONE)))
        assert not quantities.has("runtime_written_cells")
        assert "hold 3 of the 13 written weights" in quantities.withheld_reason("runtime_written_cells")

    def test_the_timeline_refuses_a_census_split_by_no_technology_by_name(self):
        platform = _mixed()
        census, _ = plane_write_census(_mixed_rows(), platform, weight_bits=4, timesteps=None, label="record")
        writes = ProgramWrites(write_policy_of(platform), census, None, TechnologySplit(refused="a named refusal"))
        with pytest.raises(TimelineError, match="different numbers of cells.*a named refusal"):
            writes_summary(writes)

    def test_core_types_of_one_count_keep_the_one_law(self):
        """Two write technologies of one count (a write physics on one type): the whole census at the chip's count,
        as before, which is also the shares' sum."""
        cores = ({**_MIXED_CORES[0], "write_physics": "neurram"}, _MIXED_CORES[1])
        cores = tuple({k: v for k, v in c.items() if k not in ("cells_per_weight", "weight_cell_rows")} for c in cores)
        writes, quantities = _candidate(_mixed(*cores), _mixed_rows(producer=5))
        assert writes.policy.core_technologies is not None
        whole = write_quantities(writes.census.census, writes.policy, written_extents=writes.census.written_extents)
        for key in ("runtime_written_cells", "written_units", "write_events"):
            assert quantities.get(key).value == whole[key].value == _summed(_parts(writes.by_technology), key) == 18


class TestPhysicalQuantities:
    """The declared chip's cells, rows and columns are its array's: area, row drivers and converters scale with them."""

    def test_the_neurram_chip_keeps_its_256_by_256_cells(self):
        cores = target_bundles()["neurram"].values["cores"]
        assert chip_param_capacity(cores) == 48 * 256 * 256
        facts = chip_facts({"cores": [dict(cores[0])], "weight_bits": 4})
        assert (facts["cells_physical"].value, facts["axons_physical"].value, facts["neurons_physical"].value) == (
            48 * 256 * 256, 48 * 256, 48 * 256)

    def test_isaac_like_keeps_its_128_by_128_cells(self):
        cores = target_bundles()["isaac_like"].values["cores"]
        assert chip_param_capacity(cores) == 16128 * 128 * 128
        facts = chip_facts({"cores": [dict(cores[0])]})
        assert (facts["axons_physical"].value, facts["neurons_physical"].value) == (16128 * 128, 16128 * 128)

    def test_isaac_like_s_converters_are_one_per_array(self):
        from mimarsinan.deployment_record.platform_physics.conversion import conversion_model_for
        from mimarsinan.deployment_record.quantities.spec import Quantities
        model = conversion_model_for(get_platform_physics("isaac_like").conversion_model)
        core = dict(target_bundles()["isaac_like"].values["cores"][0])
        derived = model.derive(Quantities(chip_facts({"cores": [core]})))
        assert derived["adc_count"].value == 16128

    def test_a_record_s_declared_chip(self):
        from dataclasses import replace
        from mimarsinan.deployment_record.quantities.from_record import from_record
        from unit.deployment_record.record_fixtures import make_full_record
        record = make_full_record()
        record = replace(record, identity=replace(record.identity, platform={
            "cores": [dict(c) for c in target_bundles()["neurram"].values["cores"]], "weight_bits": 4}))
        quantities = from_record(record)
        assert (quantities.get("cells_physical").value, quantities.get("axons_physical").value,
                quantities.get("neurons_physical").value) == (48 * 256 * 256, 48 * 256, 48 * 256)


class TestBlock:
    """W38's arm W form: an 8-bit weight on two 4-bit differential pairs (bit slices across adjacent columns), a block
    of 2 x 2 cells: 128 x 128 weights per core on NeuRRAM's 256 x 256 cells, four written cells per written weight."""

    CORE = {"max_axons": 128, "max_neurons": 128, "count": 48, "write_granularity": "cell", "cells_per_weight": 4,
            "weight_cell_rows": 2}

    def test_the_array_and_the_physical_quantities(self):
        from mimarsinan.mapping.platform.weight_cells import array_extent
        assert array_extent(self.CORE) == (256, 256)
        facts = chip_facts({"cores": [dict(self.CORE)]})
        assert (facts["cells_physical"].value, facts["axons_physical"].value, facts["neurons_physical"].value) == (
            48 * 256 * 256, 48 * 256, 48 * 256)
        assert geometry_of([self.CORE]) == (CoreGeometry(256, 256, 48),)

    def test_one_written_weight_writes_four_cells(self):
        policy = write_policy_of({"cores": [dict(self.CORE)], "inference_batch_K": 1})
        values = write_quantities(_census(cells=1), policy)
        assert values["runtime_written_cells"].value == 4.0 and values["write_events"].value == 4.0

    def test_a_count_without_a_block_is_refused_by_name(self):
        from mimarsinan.mapping.platform.weight_cells import WeightCellsError, array_extent
        with pytest.raises(WeightCellsError, match="cells_per_weight 4 states no weight_cell_rows"):
            array_extent({"max_axons": 128, "max_neurons": 128, "cells_per_weight": 4})


class TestValidity:
    def test_the_aggregates_are_judged_at_the_array(self):
        assert geometry_of(target_bundles()["neurram"].values["cores"]) == (CoreGeometry(256, 256, 48),)
        assert geometry_of(target_bundles()["isaac_like"].values["cores"]) == (CoreGeometry(128, 128, 16128),)

    def test_a_core_of_one_cell_per_weight_is_judged_at_its_geometry(self):
        assert geometry_of([{"max_axons": 64, "max_neurons": 32, "count": 3}]) == (CoreGeometry(64, 32, 3),)
