"""E49: the pass-period census of both planes, its one rule, and the presence rule of its quantities."""

from types import SimpleNamespace

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.build.write_census import (
    PassRow,
    ProgramPasses,
    SlotWrites,
    WriteCensusError,
    core_capacity_of,
    period_census,
)
from mimarsinan.deployment_record.build.program_census import (
    candidate_program_passes,
    mapping_program_passes,
)
from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform
from mimarsinan.deployment_record.quantities.from_candidate import from_candidate
from mimarsinan.deployment_record.quantities.from_record import from_record
from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs, price_write_inputs
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.quantities.write_law import (
    WRITE_KEYS,
    plane_write_census,
    plane_writes,
    record_write_census,
)
from mimarsinan.mapping.support.period_census import PassBoundaryCensus

from unit.deployment_record.g3_fixtures import HandPassProgram, record_with, schedule, segment

_SPIKING = SimpleNamespace(retimed=False, is_cycle=False, is_cascade=False,
                           latency_group_count=None, chip_latency=None)
_WRITE_QUANTITIES = ("period_rewrite_bytes", "runtime_written_cells", "written_units",
                     "max_writes_per_cell_per_inference", "write_events")


def _row(cells, *, resident=False, marks=(), segment_index=0, latencies=(0,), types=None):
    return PassRow(segment_index=segment_index, resident=resident, core_cells=tuple(cells),
                   marks=frozenset(marks), core_extents=tuple((c, 1) for c in cells),
                   compute_steps=None, stage_latencies=tuple(latencies),
                   core_types=None if types is None else tuple(types))


def _census(rows, cores, bits=8):
    """The census on a chip of ``cores`` cores of one type (every row typed by the chip)."""
    return period_census(ProgramPasses(rows=tuple(rows)), weight_bits=bits, capacity={_T: cores}).census


#: Two core geometries of the t0_41 chip: (axons, neurons).
_A, _B = (784, 512), (512, 256)
_T = (256, 256)


def _typed(rows, capacity):
    return period_census(ProgramPasses(rows=tuple(rows)), weight_bits=8, capacity=capacity)


class TestTheRule:
    def test_a_flat_fit_rewrites_nothing(self):
        census = _census([_row([100, 50, 30])], cores=8)
        assert all(p.resident for p in census.passes)
        assert sum(p.payload_bytes for p in census.passes) == 0

    def test_a_capacity_split_rewrites_every_reprogram_pass_per_period(self):
        census = _census([_row([100, 50]), _row([30, 20])], cores=3)
        assert [p.written_cells_backing_store for p in census.passes] == [150, 50]
        assert [p.payload_bytes for p in census.passes] == [150, 50]
        assert not any(p.resident for p in census.passes)

    def test_a_residency_streamed_program_with_a_second_segment_rotates_its_two_writing_passes(self):
        """t0_44-class: pass 0 programs the bank, later passes reuse it (U1's flag), and the head segment's pass
        loads from the first core too, so the bank and the head rewrite every period although F <= C (finding 4,
        option (a): the program as executed; before, a fit was priced as loaded once)."""
        rows = [_row([60, 60])] + [_row([60, 60], resident=True) for _ in range(3)] + [_row([8], segment_index=1)]
        census = _census(rows, cores=19)
        assert [p.payload_bytes for p in census.passes] == [120, 0, 0, 0, 8]
        assert [p.resident for p in census.passes] == [False, True, True, True, False]

    def test_residency_marks_keep_their_cores(self):
        census = _census([_row([100, 50], marks={0}), _row([30, 20], marks={1})], cores=3)
        assert [p.resident_cores for p in census.passes] == [1, 1]
        assert [p.written_cells_backing_store for p in census.passes] == [50, 30]
        assert [p.payload_bytes for p in census.passes] == [50, 30]

    def test_payload_rounds_up_at_the_weight_width(self):
        census = _census([_row([3]), _row([5])], cores=1, bits=4)
        assert [p.payload_bytes for p in census.passes] == [2, 3]


class TestTheExecutedRotation:
    """Interaction finding 4, option (a): a program is priced by the rotation it executes (R3: every execution
    stage loads its cores from its core type's first core), never by a loaded-once fit. Two or more writing passes
    rewrite their unkept cores every period; a lone writing pass is never displaced; the resident case is priced
    only where residency marks (an executed residency plan) or U1's whole-pass flag keep the cores."""

    def test_several_writing_passes_rotate_although_their_tiles_fit_the_cores(self):
        written = _typed([_row([10, 10]), _row([10])], capacity={_T: 8})
        assert [p.payload_bytes for p in written.census.passes] == [20, 10]
        assert [p.resident for p in written.census.passes] == [False, False]
        assert written.slot_writes == (SlotWrites(_T, (2, 1)),) and written.most_written_core == 2

    def test_a_segment_multiplexed_program_prices_alike_at_c_equal_f_and_below(self):
        """The finding's witness shape: three segments of one pass each, 33 tiles at C = F = 33 and at C = 32."""
        rows = [_row([100] * 16), _row([110] * 16, segment_index=1), _row([44], segment_index=2)]
        at_f, below = _typed(rows, capacity={_T: 33}), _typed(rows, capacity={_T: 32})
        assert at_f.census == below.census and at_f.slot_writes == below.slot_writes
        assert at_f.most_written_core == 3
        assert sum(p.payload_bytes for p in at_f.census.passes) == 16 * 100 + 16 * 110 + 44

    def test_a_lone_writing_pass_is_never_displaced(self):
        """U1's bank passes after the first stay resident, so one pass writes: nothing is rewritten per period."""
        rows = [_row([60, 60])] + [_row([60, 60], resident=True) for _ in range(3)]
        written = _typed(rows, capacity={_T: 2})
        assert sum(p.payload_bytes for p in written.census.passes) == 0
        assert all(p.resident for p in written.census.passes)
        assert written.slot_writes == () and written.most_written_core == 0

    def test_marks_that_keep_every_core_give_the_resident_census(self):
        """A residency plan's floor at F <= C pins every tile of every pass: the resident census."""
        rows = [_row([10, 10], marks={0, 1}), _row([10], marks={0}, segment_index=1)]
        written = _typed(rows, capacity={_T: 3})
        assert sum(p.payload_bytes for p in written.census.passes) == 0
        assert [p.resident_cores for p in written.census.passes] == [2, 1]
        assert all(p.resident for p in written.census.passes) and written.slot_writes == ()

    def test_a_pass_is_resident_exactly_when_it_writes_nothing(self):
        written = _typed([_row([10, 10], marks={0, 1}), _row([10]), _row([10])], capacity={_T: 3})
        assert [(p.resident, p.resident_cores, p.payload_bytes) for p in written.census.passes] == [
            (True, 2, 0), (False, 0, 10), (False, 0, 10)]

    def test_the_rotation_is_decided_without_core_types(self):
        """Untyped rows on a chip of several core types: the rotation needs only the writing passes, so the census
        is stated; only the physical core of a rewritten tile is unknown."""
        rotating = _typed([_row([10, 10]), _row([10])], capacity={_A: 1, _B: 6})
        assert [p.payload_bytes for p in rotating.census.passes] == [20, 10]
        assert rotating.slot_writes is None and "no core type" in rotating.census.label
        lone = _typed([_row([10, 10])], capacity={_A: 1, _B: 6})
        assert sum(p.payload_bytes for p in lone.census.passes) == 0 and lone.most_written_core == 0


class TestTheMostWrittenCore:
    """Endurance is a property of the physical core: every pass that rewrites a core in the period wears it."""

    def test_three_passes_on_three_cores_write_the_first_core_three_times(self):
        """t0_53-class: a 3-core chip split into passes of 3, 3 and 1 cores (7 core-writes on 3 cores)."""
        written = _typed([_row([10, 10, 10]), _row([10, 10, 10]), _row([10])], capacity={_T: 3})
        assert written.slot_writes == (SlotWrites(_T, (3, 2, 2)),)
        assert written.most_written_core == 3

    def test_two_passes_that_fit_side_by_side_rotate_on_the_first_cores(self):
        """Finding 4, option (a): both passes load from the first core (before: a fit, no core written)."""
        written = _typed([_row([10, 10]), _row([10])], capacity={_T: 3})
        assert written.slot_writes == (SlotWrites(_T, (2, 1)),) and written.most_written_core == 2

    def test_a_flat_fit_writes_no_core(self):
        written = _typed([_row([10, 10, 10])], capacity={_T: 3})
        assert written.slot_writes == () and written.most_written_core == 0

    def test_each_core_type_rotates_on_its_own_cores(self):
        """The t0_41 3 + 3 program: five reprogram segments, four of them write a 512 x 256 core."""
        rows = [_row([5400], types=[_B]), _row([20592, 20592], types=[_B, _B]), _row([75990], types=[_A]),
                _row([7650], types=[_B]), _row([750], types=[_B])]
        written = _typed(rows, capacity={_A: 3, _B: 3})
        assert set(written.slot_writes) == {SlotWrites(_A, (1,)), SlotWrites(_B, (4, 1))}
        assert written.most_written_core == 4

    def test_a_residency_pin_holds_its_core_and_the_other_writes_rotate_on_the_rest(self):
        written = _typed([_row([10, 10], marks={0}), _row([10, 10])], capacity={_T: 3})
        assert written.slot_writes == (SlotWrites(_T, (2, 1)),)

    def test_pins_that_leave_a_pass_too_few_cores_refuse_by_name(self):
        with pytest.raises(WriteCensusError) as info:
            _typed([_row([10, 10], marks={0, 1}), _row([10])], capacity={_T: 2})
        assert isinstance(info.value, ToolchainRefusal)

    def test_a_plane_without_core_types_on_a_mixed_chip_states_no_core_writes(self):
        """Three untyped tiles on a chip of two cores provably rewrite, but on no stated core."""
        written = _typed([_row([10]), _row([10]), _row([10])], capacity={_A: 1, _B: 1})
        assert written.slot_writes is None and written.most_written_core is None
        assert "no core type" in written.census.label and "no rewritten core is placed" in written.census.label

    def test_a_pin_is_counted_on_its_own_core_type(self):
        """A mark on the 512 x 256 core of a two-type pass pins that type, never the first core's."""
        written = _typed([_row([10, 10], types=[_A, _B], marks={1}), _row([10], types=[_A])],
                         capacity={_A: 1, _B: 1})
        assert written.slot_writes == (SlotWrites(_A, (2,)),)
        assert [p.resident_cores for p in written.census.passes] == [1, 0]

    def test_a_reprogram_pass_whose_every_core_is_pinned_reads_resident(self):
        """``resident`` means the pass writes nothing per period (the plan census's meaning; before, it was U1's
        whole-pass flag or a flat fit only); per-core pins count in resident_cores."""
        written = _typed([_row([10, 10], marks={0, 1}), _row([10]), _row([10])], capacity={_T: 3})
        first = written.census.passes[0]
        assert (first.resident, first.resident_cores, first.payload_bytes) == (True, 2, 0)
        assert written.slot_writes == (SlotWrites(_T, (2,)),)


class TestThePhysicalCores:
    """CL3 (d) on the executed programs: a fused core (``canonical_fuse_hardcores``: axons summed over cores of one
    declared type) occupies its component cores, and pins the builder fits by its dominance matching (a tile on a
    core at least its size, ``residency_rules.check_pinned_fit``) never refuse the census."""

    _F = (2048, 512)  # two fused 1024 x 512 cores

    def test_a_fused_core_occupies_its_component_cores(self):
        rows = [_row([300, 10], types=[self._F, (512, 256)]), _row([20], types=[(1024, 512)])]
        written = _typed(rows, capacity={(1024, 512): 4, (512, 256): 4})
        assert [p.payload_bytes for p in written.census.passes] == [310, 20]
        assert set(written.slot_writes) == {SlotWrites((1024, 512), (2, 1)), SlotWrites((512, 256), (1,))}
        assert written.pass_core_types == ((((512, 256), 1), ((1024, 512), 2)), (((1024, 512), 1),))

    def test_a_pinned_fused_core_pins_every_component(self):
        """Two of four cores pinned by one fused core: the two rotating passes share the other two."""
        pinned = _row([300], types=[self._F], marks={0})
        written = _typed([pinned, _row([20, 20], types=[(1024, 512)] * 2), _row([20], types=[(1024, 512)])],
                         capacity={(1024, 512): 4})
        assert written.slot_writes == (SlotWrites((1024, 512), (2, 1)),)
        with pytest.raises(WriteCensusError):
            _typed([pinned, _row([20] * 3, types=[(1024, 512)] * 3), _row([20], types=[(1024, 512)])],
                   capacity={(1024, 512): 4})

    def test_a_core_that_decomposes_into_no_declared_core_refuses_by_name(self):
        with pytest.raises(WriteCensusError) as info:
            _typed([_row([10], types=[(3000, 512)])], capacity={(1024, 512): 4, (512, 256): 4})
        assert "3000 x 512" in str(info.value)

    def test_pins_the_builder_fits_on_larger_cores_state_the_census_without_placing_its_rewrites(self):
        """The t0_44r floor shape: 512 x 256 pins and a pass's 512 x 256 tiles exceed that type's cores, while the
        idle 784 x 512 cores hold the excess (a dominance matching): the census is stated, no rewrite is placed."""
        small, large = (512, 256), (784, 512)
        rows = [_row([10] * 3, types=[small] * 3, marks={0, 1}), _row([10] * 3, types=[small] * 3),
                _row([10], types=[large])]
        written = _typed(rows, capacity={small: 4, large: 4})
        assert [p.payload_bytes for p in written.census.passes] == [10, 30, 10]
        assert written.slot_writes is None and "at least its size" in written.census.label
        with pytest.raises(WriteCensusError):
            _typed(rows, capacity={small: 4, large: 0})


class TestPerTypeCapacity:
    """Each core type rotates on its own cores; capacity places the rewritten cores and never makes a program of
    several writing passes loaded once (finding 4, option (a); before, tiles that fit their types were loaded once)."""

    @pytest.mark.parametrize("capacity, tiles, slots", [
        ({_A: 1, _B: 6}, [_A] + [_B] * 6, {SlotWrites(_A, (1,)), SlotWrites(_B, (3, 2, 1))}),
        ({_A: 6, _B: 3}, [_A] * 4 + [_B] * 3, {SlotWrites(_A, (2, 2)), SlotWrites(_B, (2, 1))})])
    def test_seven_tiles_that_fit_their_types_rotate_on_each_type(self, capacity, tiles, slots):
        rows = [_row([10] * 2, types=tiles[:2]), _row([10] * 3, types=tiles[2:5]), _row([10] * 2, types=tiles[5:])]
        written = _typed(rows, capacity=capacity)
        assert [p.payload_bytes for p in written.census.passes] == [20, 30, 20]
        assert not any(p.resident for p in written.census.passes)
        assert set(written.slot_writes) == slots

    def test_tiles_equal_to_the_cores_rotate(self):
        written = _typed([_row([10, 10]), _row([10])], capacity={_T: 3})
        assert [p.payload_bytes for p in written.census.passes] == [20, 10]

    def test_two_declared_entries_of_one_geometry_pool_their_cores(self):
        platform = {"cores": [{"max_axons": 256, "max_neurons": 256, "count": 2},
                              {"max_axons": 256, "max_neurons": 256, "count": 1}]}
        assert core_capacity_of(platform) == {_T: 3}
        written = _typed([_row([10, 10]), _row([10])], capacity=core_capacity_of(platform))
        assert written.slot_writes == (SlotWrites(_T, (2, 1)),)

    def test_one_overflowing_type_rewrites_the_program(self):
        """Three tiles fit the 0.8x budget of 8, but two 784 x 512 tiles share one such core."""
        rows = [_row([10, 10], types=[_A, _B]), _row([10], types=[_A])]
        written = _typed(rows, capacity={_A: 1, _B: 9})
        assert [p.payload_bytes for p in written.census.passes] == [20, 10]
        assert written.most_written_core == 2



class TestAPlaneWithoutCoreTypes:
    """On a chip of several core types, untyped rows rotate by the same rule as typed ones (the rotation needs
    no core type); only the physical core of a rewritten tile is unknown, so no core is placed. Before finding 4,
    option (a), tiles within the declared cores were undecidable (loaded once or not) and refused by name."""

    @pytest.mark.parametrize("capacity, tiles", [({_A: 1, _B: 6}, 6), ({_A: 1, _B: 6}, 7),
                                                 ({_A: 2, _B: 2}, 3)])
    def test_untyped_tiles_that_fit_the_declared_cores_rotate_on_no_stated_core(self, capacity, tiles):
        rows = [_row([10] * (tiles - 1)), _row([10])]
        written = _typed(rows, capacity=capacity)
        assert [p.payload_bytes for p in written.census.passes] == [10 * (tiles - 1), 10]
        assert written.slot_writes is None and "no core type" in written.census.label

    def test_untyped_rows_rewrite_what_typed_rows_rewrite(self):
        untyped = _typed([_row([10, 10]), _row([10])], capacity={_A: 1, _B: 9})
        typed = _typed([_row([10, 10], types=[_A, _B]), _row([10], types=[_A])], capacity={_A: 1, _B: 9})
        assert [p.payload_bytes for p in typed.census.passes] == [20, 10]
        assert untyped.census.passes == typed.census.passes and untyped.slot_writes is None

    def test_untyped_tiles_beyond_every_declared_core_rewrite_exactly_as_typed(self):
        untyped = _typed([_row([10, 10]), _row([10])], capacity={_A: 1, _B: 1})
        typed = _typed([_row([10, 10], types=[_A, _B]), _row([10], types=[_B])], capacity={_A: 1, _B: 1})
        assert [p.payload_bytes for p in untyped.census.passes] == [20, 10]
        assert untyped.census.passes == typed.census.passes
        assert "no core type" in untyped.census.label and "no core type" not in typed.census.label

    def test_a_row_on_an_undeclared_core_type_refuses_by_name(self):
        with pytest.raises(WriteCensusError) as info:
            _typed([_row([10], types=[(128, 128)])], capacity={_A: 1, _B: 1})
        assert "128 x 128" in str(info.value)

    @pytest.mark.parametrize("capacity", [None, {}])
    def test_a_chip_without_a_complete_core_geometry_refuses_by_name(self, capacity):
        with pytest.raises(WriteCensusError) as info:
            period_census(ProgramPasses(rows=(_row([10]),)), weight_bits=8, capacity=capacity)
        assert "core geometry" in str(info.value)

    def test_resident_rows_program_nothing_on_an_untyped_plane(self):
        """Three resident (U1) tiles program nothing; the two writing passes rotate."""
        rows = [_row([10] * 4), _row([10]), _row([10] * 3, resident=True)]
        written = _typed(rows, capacity={_A: 1, _B: 6})
        assert [(p.resident, p.payload_bytes) for p in written.census.passes] == [(False, 40), (False, 10), (True, 0)]

    def test_an_incomplete_core_list_refuses_by_name_through_the_plane(self):
        """Targeted pass: a declared core type without max_axons reaches ``plane_writes`` as a refusal by
        name (WriteCensusError's reason), never a KeyError."""
        platform = {"cores": [{"max_axons": 256, "max_neurons": 256, "count": 2}, {"max_neurons": 128, "count": 1}],
                    "weight_bits": 8, "inference_batch_K": 1}
        assert core_capacity_of(platform) is None
        writes = plane_writes(ProgramPasses(rows=(_row([10]),)), platform, weight_bits=8, timesteps=32,
                              label="hand")
        assert writes.census is None and "complete core geometry" in writes.unavailable
        with pytest.raises(WriteCensusError):
            period_census(ProgramPasses(rows=(_row([10]),)), weight_bits=8, capacity=core_capacity_of(platform))


class TestARetimedRewritingProgram:
    """Targeted pass: under per-hop re-timing a pass runs one execution stage per depth level, and R3 loads
    each stage from its type's first core; the census states no core of each stage, so it places no
    rewritten core (the lifetime is the pigeonhole bound) and says why."""

    def test_a_rewritten_row_of_several_execution_stages_places_no_core(self):
        written = _typed([_row([10, 10], latencies=(0, 0)), _row([10])], capacity={_T: 2})
        assert written.slot_writes is None and written.most_written_core is None
        assert "execution stages" in written.census.label
        assert [p.payload_bytes for p in written.census.passes] == [20, 10]

    def test_a_resident_row_of_several_stages_writes_nothing_and_keeps_its_placement(self):
        written = _typed([_row([10, 10], latencies=(0, 0))], capacity={_T: 2})
        assert written.slot_writes == () and written.most_written_core == 0

    def test_the_retimed_candidate_plane_states_its_level_stages(self):
        retimed = SimpleNamespace(**{**vars(_SPIKING), "retimed": True})
        passes = candidate_program_passes(_softcores(), _candidate_noc(), semantics=retimed, timesteps=4)
        assert [len(row.stage_latencies) for row in passes.rows] == [2, 1]
        census, reason = plane_write_census(passes, _platform(inference_batch_K=1), weight_bits=8, timesteps=4,
                                            label="candidate")
        assert reason is None and census.most_written_core is None
        plain = candidate_program_passes(_softcores(), _candidate_noc(), semantics=_SPIKING, timesteps=4)
        census, _ = plane_write_census(plain, _platform(inference_batch_K=1), weight_bits=8, timesteps=4,
                                       label="candidate")
        assert census.most_written_core == 2


def _mixed_platform(**extra):
    base = {"cores": [{"max_axons": 784, "max_neurons": 512, "count": 1, "write_granularity": "cell"},
                      {"max_axons": 512, "max_neurons": 256, "count": 9, "write_granularity": "cell"}],
            "weight_bits": 8, "inference_batch_K": 1, "inference_rate_hz": 30.0, "min_lifetime_hours": 43_800.0}
    base.update(extra)
    return base


def _priced(rows, platform, label="candidate"):
    writes = plane_writes(ProgramPasses(rows=tuple(rows)), platform, weight_bits=8, timesteps=32, label=label)
    return writes, price_write_inputs(WriteInputs.of(writes), get_platform_physics("isaac_like"))


class TestUndecidableWritesArePricedNowhere:
    """The reviewer's reverse witness: an untyped plane never derives admitted. Under the executed rotation
    (finding 4, option (a)) it states its rewrite, so its write terms price as the typed rows do and its lifetime
    is the pigeonhole upper bound, which can prove inadmissibility only (before, the whole census was refused as
    undecidable)."""

    def test_an_untyped_mixed_chip_prices_the_writes_and_bounds_the_lifetime(self):
        writes, pricing = _priced([_row([1000, 10]), _row([1000])], _mixed_platform())
        typed, exact = _priced([_row([1000, 10], types=[_A, _B]), _row([1000], types=[_A])], _mixed_platform())
        assert writes.census is not None and writes.census.most_written_core is None
        terms, typed_terms = {t.name: t for t in pricing.terms}, {t.name: t for t in exact.terms}
        assert terms["energy_write_mj"].to_dict() == typed_terms["energy_write_mj"].to_dict()
        assert "upper bound" in terms["lifetime_inferences"].source
        verdict, = pricing.verdicts
        assert verdict.verdict == "inadmissible" and "upper bound" in verdict.reason

    def test_typed_rows_on_the_same_chip_price_the_rewrite(self):
        _, pricing = _priced([_row([1000, 10], types=[_A, _B]), _row([1000], types=[_A])], _mixed_platform())
        assert [v.verdict for v in pricing.verdicts] == ["inadmissible"]
        assert {t.name: t for t in pricing.terms}["energy_write_mj"].value > 0

    def test_untyped_tiles_beyond_every_core_keep_the_bound_and_its_inadmissible_proof(self):
        platform = _mixed_platform(cores=[{"max_axons": 784, "max_neurons": 512, "count": 1, "write_granularity": "cell"},
                                          {"max_axons": 512, "max_neurons": 256, "count": 1, "write_granularity": "cell"}])
        untyped, bound = _priced([_row([1000, 10]), _row([1000])], platform)
        typed, exact = _priced([_row([1000, 10], types=[_A, _B]), _row([1000], types=[_A])], platform)
        assert untyped.census.most_written_core is None and typed.census.most_written_core == 2
        bound_terms = {t.name: t for t in bound.terms}
        exact_terms = {t.name: t for t in exact.terms}
        for name in ("energy_write_mj", "latency_write_exposed_s"):
            assert bound_terms[name].to_dict() == exact_terms[name].to_dict(), name
        assert "upper bound" in bound_terms["lifetime_inferences"].source
        verdict, = bound.verdicts
        assert verdict.verdict == "inadmissible" and "upper bound" in verdict.reason


def _softcores():
    return [SimpleNamespace(segment_id=0, latency_tag=0, output_count=10),
            SimpleNamespace(segment_id=0, latency_tag=1, output_count=6),
            SimpleNamespace(segment_id=0, latency_tag=2, output_count=4)]


def _candidate_noc(marks=None, first_marks=None):
    programs = (HandPassProgram(core_cells=(100, 50), schedule_resident_cores=first_marks),
                HandPassProgram(core_cells=(30,), schedule_resident_cores=marks))
    return SimpleNamespace(pass_placements=(((0, 0), (1, 1)), ((2, 0),)), pass_programs=programs,
                           census=SimpleNamespace(pair_wires={(0, 1): 5, (1, 2): 6, (0, 2): 3}))


def _hard_core(rows, cols, latency):
    return SimpleNamespace(axons_per_core=256, neurons_per_core=256, available_axons=256 - rows,
                           available_neurons=256 - cols, unusable_space=0, latency=latency)


def _typed_core(rows, cols, latency, axons, neurons):
    return SimpleNamespace(axons_per_core=axons, neurons_per_core=neurons, available_axons=axons - rows,
                           available_neurons=neurons - cols, unusable_space=0, latency=latency)


def _slice(node, size):
    return SimpleNamespace(node_id=node, size=size)


def _hybrid(marks=None, first_marks=None):
    first = SimpleNamespace(kind="neural", schedule_segment_index=0, schedule_pass_index=0,
                            hard_core_mapping=SimpleNamespace(cores=[_hard_core(10, 10, 0), _hard_core(5, 10, 1)]),
                            input_map=[], output_map=[_slice(0, 10), _slice(1, 6)])
    second = SimpleNamespace(kind="neural", schedule_segment_index=0, schedule_pass_index=1,
                             hard_core_mapping=SimpleNamespace(cores=[_hard_core(30, 1, 2)]),
                             input_map=[_slice(0, 10), _slice(1, 6)], output_map=[_slice(2, 4)])
    if marks is not None:
        second.schedule_resident_cores = marks
    if first_marks is not None:
        first.schedule_resident_cores = first_marks
    return SimpleNamespace(stages=[first, second])


def _streamed_noc():
    """t0_44-class on the candidate plane: pass 0 programs a two-core bank, two later passes reuse it."""
    programs = (HandPassProgram(core_cells=(60, 60)), HandPassProgram(core_cells=(60, 60), resident=True),
                HandPassProgram(core_cells=(60, 60), resident=True))
    placements = (((0, 0), (1, 1)), ((2, 0), (3, 1)), ((4, 0), (5, 1)))
    return SimpleNamespace(pass_placements=placements, pass_programs=programs, census=None)


def _streamed_softcores():
    return [SimpleNamespace(segment_id=0, latency_tag=0, output_count=6) for _ in range(6)]


def _streamed_hybrid():
    """The same program as built: three stages of one segment, the later two on the bank's weights."""
    def stage(index, resident):
        return SimpleNamespace(kind="neural", schedule_segment_index=0, schedule_pass_index=index,
                               schedule_weights_resident=resident,
                               hard_core_mapping=SimpleNamespace(cores=[_hard_core(60, 1, 0), _hard_core(60, 1, 0)]),
                               input_map=[], output_map=[])
    return SimpleNamespace(stages=[stage(0, False), stage(1, True), stage(2, True)])


def _platform(**extra):
    base = {"cores": [{"max_axons": 256, "max_neurons": 256, "count": 2}], "weight_bits": 8,
            "simulation_steps": 32}
    base.update(extra)
    return base


class TestBothPlanes:
    def test_the_candidate_rows_follow_the_pass_structure(self):
        passes = candidate_program_passes(_softcores(), _candidate_noc(), semantics=_SPIKING, timesteps=4)
        assert [row.core_cells for row in passes.rows] == [(100, 50), (30,)]
        assert [row.stage_latencies for row in passes.rows] == [(1,), (0,)]
        assert [row.compute_steps for row in passes.rows] == [4 + 1 + 1, 4 + 0 + 1]
        assert passes.boundary_widths == ((0, 16, 16),)

    def test_the_built_program_rows_follow_its_stages(self):
        passes = mapping_program_passes(_hybrid())
        assert [row.core_cells for row in passes.rows] == [(100, 50), (30,)]
        assert [row.stage_latencies for row in passes.rows] == [(1,), (0,)]
        assert passes.boundary_widths == ((0, 16, 16),)

    @pytest.mark.parametrize("value_domain", [False, True])
    def test_candidate_equals_record_on_a_capacity_split(self, value_domain):
        platform = _platform(**({"core_semantics": "mvm", "activation_bits": 8} if value_domain else {}))
        candidate = plane_write_census(
            candidate_program_passes(_softcores(), _candidate_noc(), semantics=_SPIKING, timesteps=32),
            platform, weight_bits=8, timesteps=32, label="")[0].census
        built = plane_write_census(mapping_program_passes(_hybrid()), platform, weight_bits=8,
                                   timesteps=32, label="")[0].census
        fields = ("position", "segment_index", "resident", "cores", "resident_cores",
                  "written_cells_backing_store", "written_cells_producer", "payload_bytes")
        for a, b in zip(candidate.passes, built.passes):
            assert [getattr(a, f) for f in fields] == [getattr(b, f) for f in fields]
        assert candidate.boundaries == built.boundaries
        width_bytes = 16 if value_domain else 16 * 4
        assert candidate.boundaries == (PassBoundaryCensus(0, width_bytes, width_bytes),)
        if value_domain:
            assert candidate == built  # the value window reads the same stage latencies on both planes
        assert sum(p.payload_bytes for p in candidate.passes) == 180

    @pytest.mark.parametrize("count", [3, 8])
    def test_candidate_equals_record_on_two_passes_that_fit(self, count):
        """Three programmed tiles in two passes on a chip of 3 or 8 cores: the executed rotation on both planes
        (finding 4, option (a); before, loaded once on both planes)."""
        platform = _platform(cores=[{"max_axons": 256, "max_neurons": 256, "count": count}],
                             core_semantics="mvm", activation_bits=8)
        candidate = plane_write_census(
            candidate_program_passes(_softcores(), _candidate_noc(), semantics=_SPIKING, timesteps=32),
            platform, weight_bits=8, timesteps=32, label="")[0]
        built = plane_write_census(mapping_program_passes(_hybrid()), platform, weight_bits=8,
                                   timesteps=32, label="")[0]
        assert candidate.census == built.census
        assert [(p.resident, p.payload_bytes) for p in built.census.passes] == [(False, 150), (False, 30)]
        assert candidate.slot_writes == built.slot_writes == (SlotWrites((256, 256), (2, 1)),)

    def test_candidate_equals_record_on_a_residency_streamed_program(self):
        """t0_44-class: the bank stays programmed, so a chip of two cores rewrites nothing per period."""
        platform = _platform(core_semantics="mvm", activation_bits=8)
        candidate = plane_write_census(
            candidate_program_passes(_streamed_softcores(), _streamed_noc(), semantics=_SPIKING, timesteps=32),
            platform, weight_bits=8, timesteps=32, label="")[0]
        built = plane_write_census(mapping_program_passes(_streamed_hybrid()), platform, weight_bits=8,
                                   timesteps=32, label="")[0]
        assert candidate.census == built.census
        assert sum(p.payload_bytes for p in built.census.passes) == 0
        assert [p.resident for p in built.census.passes] == [True, True, True]
        assert candidate.most_written_core == built.most_written_core == 0

    def test_candidate_equals_record_on_the_most_written_core(self):
        platform = _platform()
        candidate = plane_write_census(
            candidate_program_passes(_softcores(), _candidate_noc(), semantics=_SPIKING, timesteps=32),
            platform, weight_bits=8, timesteps=32, label="")[0]
        built = plane_write_census(mapping_program_passes(_hybrid()), platform, weight_bits=8,
                                   timesteps=32, label="")[0]
        assert candidate.slot_writes == built.slot_writes == (SlotWrites((256, 256), (2, 1)),)

    @pytest.mark.parametrize("typed", [True, False])
    def test_a_mixed_type_chip_needs_the_candidate_core_types(self, typed):
        """Two types: with each core's geometry the planes agree core for core; without it the three
        tiles still exceed the chip's two cores, so the candidate rewrites but places no core."""
        geometry = ((784, 512), (512, 256)) if typed else None
        noc = SimpleNamespace(pass_placements=(((0, 0), (1, 1)), ((2, 0),)), census=None, pass_programs=(
            HandPassProgram(core_cells=(100, 50), core_geometry=geometry),
            HandPassProgram(core_cells=(30,), core_geometry=None if geometry is None else geometry[1:])))
        platform = _platform(cores=[{"max_axons": 784, "max_neurons": 512, "count": 1},
                                    {"max_axons": 512, "max_neurons": 256, "count": 1}])
        candidate = plane_write_census(
            candidate_program_passes(_softcores(), noc, semantics=_SPIKING, timesteps=32),
            platform, weight_bits=8, timesteps=32, label="candidate")[0]
        hybrid = _hybrid()
        hybrid.stages[0].hard_core_mapping.cores = [_typed_core(10, 10, 0, 784, 512), _typed_core(5, 10, 1, 512, 256)]
        hybrid.stages[1].hard_core_mapping.cores = [_typed_core(30, 1, 2, 512, 256)]
        built = plane_write_census(mapping_program_passes(hybrid), platform, weight_bits=8, timesteps=32,
                                   label="candidate")[0]
        assert built.slot_writes == (SlotWrites((512, 256), (2,)), SlotWrites((784, 512), (1,)))
        if typed:
            assert candidate.slot_writes == built.slot_writes
            assert candidate.census.passes == tuple(
                type(p)(**{**p.__dict__, "compute_steps": c.compute_steps})
                for p, c in zip(built.census.passes, candidate.census.passes))
        else:
            assert candidate.slot_writes is None and "no core type" in candidate.census.label

    def test_the_record_plane_types_each_core_of_a_non_square_chip(self):
        """t0_41's two core types on the sealed record: each written core wears its own type's cores."""
        cores = [{"max_axons": 784, "max_neurons": 512, "count": 1},
                 {"max_axons": 512, "max_neurons": 256, "count": 2}]
        stages = (segment(1, 0, 0, [100, 50], types=[_A, _B]), segment(2, 0, 1, [30, 20], types=[_A, _B]),
                  segment(3, 0, 2, [10], types=[_B]))
        sealed = record_write_census(record_with(schedule(stages), cores=cores, inference_batch_K=1))
        assert sealed.census is not None, sealed.unavailable
        assert set(sealed.census.slot_writes) == {SlotWrites(_A, (2,)), SlotWrites(_B, (3,))}
        assert sealed.census.most_written_core == 3
        hybrid = SimpleNamespace(stages=[
            SimpleNamespace(kind="neural", schedule_segment_index=0, schedule_pass_index=q, input_map=[],
                            output_map=[], hard_core_mapping=SimpleNamespace(cores=[
                                _typed_core(c, 1, 0, *kind) for c, kind in zip(cells, kinds)]))
            for q, (cells, kinds) in enumerate((([100, 50], [_A, _B]), ([30, 20], [_A, _B]), ([10], [_B])))])
        built = plane_write_census(mapping_program_passes(hybrid), {"cores": cores, "weight_bits": 8},
                                   weight_bits=8, timesteps=32, label="record")[0]
        assert built.slot_writes == sealed.census.slot_writes

    def test_the_same_marks_give_the_same_census(self):
        """Acceptance 17 on hand objects: a marked core is kept on both planes, and the other two
        tiles rotate on the one core the pin leaves."""
        platform = _platform(core_semantics="mvm", activation_bits=8)
        candidate = plane_write_census(
            candidate_program_passes(_softcores(), _candidate_noc(first_marks=frozenset({0})),
                                     semantics=_SPIKING, timesteps=32),
            platform, weight_bits=8, timesteps=32, label="")[0]
        built = plane_write_census(mapping_program_passes(_hybrid(first_marks=frozenset({0}))), platform,
                                   weight_bits=8, timesteps=32, label="")[0]
        assert candidate.census == built.census
        assert [p.written_cells_backing_store for p in built.census.passes] == [50, 30]
        assert built.census.passes[0].resident_cores == 1
        assert candidate.slot_writes == built.slot_writes == (SlotWrites((256, 256), (2,)),)

    def test_a_pin_that_leaves_a_pass_too_few_cores_refuses_by_name(self):
        """Pass 1's tile pinned on one of two cores leaves pass 0 one core for its two tiles."""
        census, reason = plane_write_census(mapping_program_passes(_hybrid(frozenset({0}))), _platform(),
                                            weight_bits=8, timesteps=32, label="record")
        assert census is None and "residency marks pin 1 of the 2 cores" in reason


def _candidate_quantities(pcfg):
    passes = candidate_program_passes(_softcores(), _candidate_noc(), semantics=_SPIKING, timesteps=32)
    context = candidate_context_from_platform(pcfg, passes=passes)
    return from_candidate(layout=None, chip_param_capacity=None, total_params=None,
                          host_side_segment_count=None, context=context)


def _record(**platform):
    stages = (segment(1, 0, 0, [100, 50]), segment(2, 0, 1, [30]))
    platform.setdefault("cores", [{"max_axons": 256, "max_neurons": 256, "count": 2}])
    return record_with(schedule(stages), **platform)


class TestThePresenceRule:
    def test_the_write_keys_are_the_s0_policy_keys(self):
        assert WRITE_KEYS == ("inference_batch_K", "write_overlap", "write_ports", "inference_rate_hz",
                              "min_lifetime_hours")

    def test_without_a_batch_no_write_quantity_is_present_on_either_plane(self):
        candidate = _candidate_quantities(_platform())
        record = from_record(_record())
        for key in _WRITE_QUANTITIES:
            assert not candidate.has(key) and not record.has(key), key
        assert record_write_census(_record()) is None

    def test_with_a_batch_the_period_rewrite_is_present_and_equal_on_both_planes(self):
        candidate = _candidate_quantities(_platform(inference_batch_K=16))
        record = from_record(_record(inference_batch_K=16))
        assert candidate.get("period_rewrite_bytes").value == record.get("period_rewrite_bytes").value == 180.0
        assert candidate.get("runtime_written_cells").value == pytest.approx(180 / 16)
        # Pass 0 writes cores {0, 1} and pass 1 writes core {0}: core 0 is written twice per period.
        for plane in (candidate, record):
            assert plane.get("max_writes_per_cell_per_inference").value == pytest.approx(2 / 16)

    @pytest.mark.parametrize("k", [1, 16])
    def test_three_passes_on_three_cores_wear_the_first_core_three_times_on_both_planes(self, k):
        """t0_53-class: passes of 3, 3 and 1 cores on a 3-core chip."""
        programs = (HandPassProgram(core_cells=(10, 10, 10)), HandPassProgram(core_cells=(10, 10, 10)),
                    HandPassProgram(core_cells=(10,)))
        placements = (((0, 0), (1, 1), (2, 2)), ((3, 0), (4, 1), (5, 2)), ((6, 0),))
        noc = SimpleNamespace(pass_placements=placements, pass_programs=programs, census=None)
        softcores = [SimpleNamespace(segment_id=0, latency_tag=i // 3, output_count=1) for i in range(7)]
        cores = [{"max_axons": 256, "max_neurons": 256, "count": 3}]
        context = candidate_context_from_platform(
            _platform(cores=cores, inference_batch_K=k),
            passes=candidate_program_passes(softcores, noc, semantics=_SPIKING, timesteps=32))
        candidate = from_candidate(layout=None, chip_param_capacity=None, total_params=None,
                                   host_side_segment_count=None, context=context)
        stages = (segment(1, 0, 0, [10, 10, 10]), segment(2, 0, 1, [10, 10, 10]), segment(3, 0, 2, [10]))
        record = from_record(record_with(schedule(stages), cores=cores, inference_batch_K=k))
        for plane in (candidate, record):
            assert plane.get("max_writes_per_cell_per_inference").value == pytest.approx(3 / k)


class TestBoundaryAttribution:
    def _passes(self):
        return candidate_program_passes(_softcores(), _candidate_noc(), semantics=_SPIKING, timesteps=32)

    def test_spans_that_account_for_the_carry_census_are_split_per_boundary(self):
        carry = {"boundary_out_bytes": 16 * 4, "boundary_in_bytes": 16 * 4}
        census = plane_write_census(self._passes(), _platform(), weight_bits=8, timesteps=32, label="c",
                                    carry=carry)[0].census
        assert census.boundaries == (PassBoundaryCensus(0, 64, 64),)

    def test_a_carry_census_with_trains_the_spans_do_not_attribute_withholds_the_split(self):
        """A delayed slice re-injects one more train (U1 pass projection): the totals stand, no split."""
        carry = {"boundary_out_bytes": 16 * 4, "boundary_in_bytes": (16 + 6) * 4}
        census = plane_write_census(self._passes(), _platform(), weight_bits=8, timesteps=32, label="c",
                                    carry=carry)[0].census
        assert census.boundaries == ()
        assert "withheld" in census.label
