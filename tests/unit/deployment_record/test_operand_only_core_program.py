"""[U-C C-6, seam 2] ``e_core_program`` is charged to a core when its static program is written: a core whose every
placement reads a bound bank (an operand-only core, N2) holds no static program, so it carries no per-core
programming charge, neither in the write law's period writes nor in the per-load programming term. Its operand-bank
writes stay priced by the operand write law, and the cells of its used rectangle stay in the census as they are on
every core (one dense-rectangle rule). Every plane states the operand-only cores: the built mapping and the candidate
plane from the placements, the sealed record from a mark the build seals on each core of a chip with an operand core
type; a record of such a chip sealed without the marks withholds the per-core count by name."""

from __future__ import annotations

from dataclasses import replace

import pytest

from mimarsinan.deployment_record.build.from_mapping import schedule_record_from_mapping
from mimarsinan.deployment_record.build.program_census import mapping_program_passes, record_program_passes
from mimarsinan.deployment_record.build.write_census import PassRow, ProgramPasses, period_census
from mimarsinan.deployment_record.cost.absolute.writes import ENERGY_WRITE, price_write_terms
from mimarsinan.deployment_record.quantities.from_record import from_record as quantities_from_record
from mimarsinan.deployment_record.quantities.write_law import WritePolicy, programmed_cores, written_cores
from mimarsinan.deployment_record.schema import SegmentCoreRecord
from mimarsinan.mapping.support.period_census import PassCensus, PassPeriodCensus
from unit.chip_simulation.value_run.operand_programs import attention_program
from unit.deployment_record.g3_fixtures import constant, physics

GEOMETRY = dict(axons=64, neurons=64, count=3, passes=128)
OPERAND_CORES = [{"max_axons": 64, "max_neurons": 64, "count": 3, "has_bias": True,
                  "operand": {"payload_bits": 8, "write_source": "host"}}]
CORE = constant(512.0, unit="pJ")


def _physics(**extra):
    return physics("generic_estimated_22nm", t_write_per_unit=constant(10.0, unit="ns"), **extra)


@pytest.fixture(scope="module")
def mapping():
    return attention_program("softmax").packed_mapping(**GEOMETRY)


def _operand_only_by_placement(mapping):
    """Per neural stage, the cores whose every placement reads a bound bank, read off the placements directly."""
    banks = set(mapping.dynamic_bank_bindings)
    out = []
    for stage in mapping.stages:
        if stage.kind != "neural":
            continue
        per = stage.hard_core_mapping.soft_core_placements_per_hard_core
        out.append(frozenset(i for i, placements in enumerate(per) if placements and all(
            p.get("weight_bank_id") is not None and int(p["weight_bank_id"]) in banks for p in placements)))
    return out


def _row(cells, *, operand_only=frozenset(), producer_cores=frozenset(), producer=0):
    return PassRow(segment_index=0, resident=False, core_cells=tuple(cells), marks=frozenset(), core_extents=None,
                   compute_steps=1, producer_cells=producer, producer_cores=frozenset(producer_cores),
                   operand_only_cores=frozenset(operand_only))


class TestCensus:
    def test_the_period_census_counts_the_rewritten_operand_only_cores(self):
        rows = (_row((100, 400, 40), operand_only={1}, producer_cores={1, 2}, producer=150),
                _row((100, 400), operand_only={1}, producer_cores={1}, producer=100))
        census = period_census(ProgramPasses(rows=rows), weight_bits=8, capacity={(64, 64): 3}).census
        assert [p.operand_only_cores for p in census.passes] == [1, 1]
        assert written_cores(census) == 5 and programmed_cores(census) == 3

    def test_a_kept_operand_only_core_is_not_counted(self):
        rows = (replace(_row((100, 400), operand_only={1}, producer_cores={1}, producer=100), marks=frozenset({1})),
                _row((100, 400), operand_only={1}, producer_cores={1}, producer=100))
        census = period_census(ProgramPasses(rows=rows), weight_bits=8, capacity={(64, 64): 3}).census
        assert [p.operand_only_cores for p in census.passes] == [0, 1]

    def test_a_census_without_operand_only_cores_serializes_byte_for_byte(self):
        plain = PassCensus(position=0, segment_index=0, resident=False, cores=2, resident_cores=0,
                           written_cells_backing_store=10, written_cells_producer=0, payload_bytes=10, compute_steps=1)
        assert "operand_only_cores" not in PassPeriodCensus(passes=(plain,)).to_dict()["passes"][0]
        marked = replace(plain, operand_only_cores=1)
        data = PassPeriodCensus(passes=(marked,)).to_dict()
        assert data["passes"][0]["operand_only_cores"] == 1
        assert PassPeriodCensus.from_dict(data).passes[0] == marked

    def test_operand_only_cores_cannot_exceed_the_rewritten_cores(self):
        with pytest.raises(ValueError):
            PassCensus(position=0, segment_index=0, resident=False, cores=2, resident_cores=1,
                       written_cells_backing_store=10, written_cells_producer=0, payload_bytes=10, compute_steps=1,
                       operand_only_cores=2)


class TestWriteLaw:
    def _priced(self, census, **extra):
        policy = WritePolicy(batch_k=4, write_granularity="cell", producer_write_source="host", payload_bits=8)
        return price_write_terms(census, _physics(e_core_program=CORE, **extra), policy, most_written=2)

    def test_e_core_program_skips_operand_only_cores(self):
        """Two passes of three rewritten cores, one of them operand-only in each: the charge is 4 cores / K."""
        passes = tuple(PassCensus(position=i, segment_index=0, resident=False, cores=3, resident_cores=0,
                                  written_cells_backing_store=900, written_cells_producer=50, payload_bytes=900,
                                  compute_steps=1, operand_only_cores=1) for i in (0, 1))
        plain = tuple(replace(p, operand_only_cores=0) for p in passes)
        with_only = {t.name: t for t in self._priced(PassPeriodCensus(passes=passes)).terms}[ENERGY_WRITE]
        every = {t.name: t for t in self._priced(PassPeriodCensus(passes=plain)).terms}[ENERGY_WRITE]
        assert every.value - with_only.value == pytest.approx(512e-12 * 2 / 4 * 1e3, rel=1e-9, abs=0)
        assert "cores holding a static program" in with_only.source
        assert "x rewritten_cores / K" in every.source

    def test_the_operand_bank_writes_stay_priced(self):
        """Only the per-core charge moves: the operand cells' commit and transfer are the same."""
        passes = (PassCensus(position=0, segment_index=0, resident=False, cores=1, resident_cores=0,
                             written_cells_backing_store=400, written_cells_producer=100, payload_bytes=400,
                             compute_steps=1, operand_only_cores=1),
                  PassCensus(position=1, segment_index=0, resident=False, cores=1, resident_cores=0,
                             written_cells_backing_store=300, written_cells_producer=0, payload_bytes=300,
                             compute_steps=1))
        chip = _physics(e_core_program=CORE)
        no_core = replace(chip, constants={k: v for k, v in chip.constants.items() if k != "e_core_program"})
        policy = WritePolicy(batch_k=4, write_granularity="cell", producer_write_source="host", payload_bits=8)
        without_core = {t.name: t for t in price_write_terms(PassPeriodCensus(passes=passes), no_core, policy,
                                                             most_written=1).terms}[ENERGY_WRITE]
        with_core = {t.name: t for t in self._priced(PassPeriodCensus(passes=passes)).terms}[ENERGY_WRITE]
        assert with_core.value - without_core.value == pytest.approx(512e-12 * 1 / 4 * 1e3, rel=1e-9, abs=0)
        assert "host-sourced payload bytes" in without_core.source and without_core.value > 0


class TestPlanes:
    def test_the_mapping_plane_states_the_operand_only_cores(self, mapping):
        rows = mapping_program_passes(mapping).rows
        expected = _operand_only_by_placement(mapping)
        assert [row.operand_only_cores for row in rows] == expected
        assert any(expected), "the vehicle holds operand-only cores"
        for row in rows:
            assert row.operand_only_cores <= row.producer_cores

    def test_the_record_seals_the_marks_and_reads_them_back(self, mapping):
        schedule = schedule_record_from_mapping(mapping, weight_bits=8, params_reloaded=0)
        segments = [s for s in schedule.stages if hasattr(s, "cores")]
        for segment, only in zip(segments, _operand_only_by_placement(mapping)):
            assert [c.operand_only for c in segment.cores] == [i in only for i in range(len(segment.cores))]

    def test_a_chip_without_operand_regions_seals_no_mark(self):
        core = SegmentCoreRecord(core_index=0, axons=4, neurons=4, axons_used=4, neurons_used=4, cells_used=16,
                                 params_bytes=16, connectivity_entries=4, static_delay_levels=None)
        assert "operand_only" not in core.to_dict()


class TestRecordPlane:
    def test_reprogrammed_cores_leave_out_the_marked_operand_only_cores(self, mapping):
        from unit.deployment_record.record_fixtures import make_full_record, make_identity
        schedule = schedule_record_from_mapping(mapping, weight_bits=8, params_reloaded=0)
        platform = {"cores": OPERAND_CORES, "weight_bits": 8, "core_semantics": "mvm"}
        record = replace(make_full_record(), schedule=schedule, identity=replace(make_identity(), platform=platform))
        segments = [s for s in schedule.stages if hasattr(s, "cores") and s.programming == "reprogram"]
        every = sum(len(s.cores) for s in segments)
        only = sum(1 for s in segments for c in s.cores if c.operand_only)
        assert only > 0
        q = quantities_from_record(record)
        assert q.get("reprogrammed_cores").value == every - only
        rows = record_program_passes(record).rows
        assert [row.operand_only_cores for row in rows] == [
            frozenset(i for i, c in enumerate(s.cores) if c.operand_only)
            for s in schedule.stages if hasattr(s, "cores")]

    def test_an_operand_chip_without_bound_banks_marks_every_core(self):
        """[targeted M4] On a chip with an operand core type every core seals its mark, bound banks or none: a
        program that binds no bank marks each core False, so its record prices every reprogrammed core."""
        from unit.deployment_record.g3_fixtures import built_chain
        from unit.deployment_record.record_fixtures import make_full_record, make_identity

        mapping = built_chain()
        assert not dict(getattr(mapping, "dynamic_bank_bindings", None) or {})
        passes = dict(weight_bits=8, params_reloaded=0, timesteps=4, pass_transfer="collapse")
        marked = schedule_record_from_mapping(mapping, operand_chip=True, **passes)
        plain = schedule_record_from_mapping(mapping, **passes)
        cores = [c for s in marked.stages if hasattr(s, "cores") for c in s.cores]
        assert cores and all(c.operand_only is False for c in cores)
        assert all(c.operand_only is None for s in plain.stages if hasattr(s, "cores") for c in s.cores)
        static = {"max_axons": 8, "max_neurons": 8, "count": 1}  # the chain's own geometry, beside an operand type
        platform = {"cores": [static, {**static, "operand": {"payload_bits": 8, "write_source": "host"}}],
                    "weight_bits": 8, "core_semantics": "mvm"}
        record = replace(make_full_record(), schedule=marked, identity=replace(make_identity(), platform=platform))
        reprogrammed = sum(len(s.cores) for s in marked.stages if hasattr(s, "cores") and s.programming == "reprogram")
        assert quantities_from_record(record).get("reprogrammed_cores").value == reprogrammed

    def test_an_operand_chip_record_without_marks_withholds_the_count_by_name(self, mapping):
        from unit.deployment_record.record_fixtures import make_full_record, make_identity
        schedule = schedule_record_from_mapping(mapping, weight_bits=8, params_reloaded=0)
        unmarked = replace(schedule, stages=tuple(
            replace(s, cores=tuple(replace(c, operand_only=None) for c in s.cores)) if hasattr(s, "cores") else s
            for s in schedule.stages))
        platform = {"cores": OPERAND_CORES, "weight_bits": 8, "core_semantics": "mvm"}
        record = replace(make_full_record(), schedule=unmarked, identity=replace(make_identity(), platform=platform))
        q = quantities_from_record(record)
        assert not q.has("reprogrammed_cores")
        assert "operand" in q.withheld["reprogrammed_cores"]


class TestProgrammingTerm:
    """[targeted M4] The per-load programming term names a withheld ``reprogrammed_cores``: beside another priced
    component as an unpriced part of the term, and as the term's refusal by name when ``e_core_program`` is its only
    component (never a silent absence)."""

    REASON = "the cores holding a static program are not determined (test)"

    #: [U-ENERGY] Every program load is the one load law over its census, which counts the programmed cores itself
    #: (``write_law.programmed_cores``: an operand-only core holds no static program). A per-core row never meets a
    #: withheld core count there: a view that states no pass structure hands the law no census, and each load figure
    #: is refused by that name (never a silent absence, never a partial price).

    def _priced(self, values, drop=(), loads=None):
        from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
        from mimarsinan.deployment_record.quantities.spec import Quantities

        chip = _physics(e_core_program=CORE)
        chip = replace(chip, constants={k: v for k, v in chip.constants.items() if k not in drop})
        pricing = price_absolute(Quantities(values, withheld={"reprogrammed_cores": self.REASON}), chip, loads=loads)
        return ({t.name: t for t in pricing.terms}, {r.name: r.reason for r in pricing.refusals}), chip

    @staticmethod
    def _loads():
        """Two written passes, rewritten each round; pass 0's second core holds operand regions only."""
        from mimarsinan.deployment_record.quantities.planes.loads import load_inputs

        rows = (_row((100, 50), operand_only={1}), _row((60,)))
        rows = tuple(replace(r, segment_index=i, core_extents=tuple((c, 1) for c in r.core_cells))
                     for i, r in enumerate(rows))
        platform = {"cores": [{"max_axons": 128, "max_neurons": 128, "count": 2, "write_granularity": "cell"}],
                    "weight_bits": 8}
        return load_inputs(ProgramPasses(rows=rows), platform, weight_bits=8, label="test")

    def test_beside_a_priced_payload_the_term_names_the_withheld_charge(self):
        from mimarsinan.deployment_record.quantities.spec import QuantityValue
        from unit.deployment_record.load_census_fixtures import law_energy, steady

        (terms, refusals), _chip = self._priced({"reprogrammed_bytes": QuantityValue(1000.0, "measured")})
        assert "energy_programming_mj" not in terms and not steady(refusals).get("energy_programming_mj")
        loads = self._loads()
        (terms, _), chip = self._priced({"reprogrammed_bytes": QuantityValue(1000.0, "measured")}, loads=loads)
        charge = law_energy(loads.period, chip).nominal - law_energy(loads.period, chip, without=("e_core_program",))\
            .nominal
        assert charge == pytest.approx(2 * 512e-12 * 1e3, rel=1e-9)  # the two cores holding a static program
        assert "e_core_program" in terms["energy_programming_mj"].source

    def test_as_the_only_component_the_term_refuses_by_name(self):
        from unit.deployment_record.load_census_fixtures import steady

        (terms, refusals), _chip = self._priced({})
        assert "energy_programming_mj" not in terms
        steady(refusals)  # each load figure refused by the law's named reason (never a silent absence)

    def test_a_payload_whose_byte_energies_are_undeclared_is_named_in_the_refusal(self):
        from mimarsinan.deployment_record.quantities.spec import QuantityValue

        (terms, _), _chip = self._priced({"reprogrammed_bytes": QuantityValue(1000.0, "measured")},
                                         drop=("e_dma_per_byte", "e_program_per_byte"), loads=self._loads())
        source = terms["energy_programming_mj"].source
        assert "unpriced: e_dma_per_byte(write transfer)" in source, source  # the transfer, named by the law

    def test_a_counted_charge_prices_as_before(self):
        from unit.deployment_record.load_census_fixtures import law_energy

        loads = self._loads()
        (terms, _), chip = self._priced({}, loads=loads)
        for corner in ("low", "nominal", "high"):
            charge = getattr(law_energy(loads.first, chip), corner) - getattr(
                law_energy(loads.first, chip, without=("e_core_program",)), corner)
            assert charge == pytest.approx(2 * 512e-12 * 1e3, rel=1e-9), corner
        assert "e_core_program" in terms["energy_load_mj"].source


class TestEmissionSite:
    """[targeted M4] The hard-core mapping step's emission marks the cores of a chip whose declared core types include
    an operand core type (``operand_chip`` from the run's platform), with no bound bank needed; on any other chip no
    core carries the mark, so its record is byte for byte unchanged. The partition and utilization fragments are
    stubbed; the schedule fragment is the step's own."""

    def _schedule(self, monkeypatch, cores):
        from types import SimpleNamespace

        from mimarsinan.pipelining.pipeline_steps.mapping import deployment_record_emission as emission
        from unit.deployment_record.g3_fixtures import built_chain

        monkeypatch.setattr(emission, "compute_partition_record", lambda *args, **kwargs: None)
        monkeypatch.setattr(emission, "utilization_record_from_mapping",
                            lambda *args, **kwargs: SimpleNamespace(to_dict=dict))
        entries = {}
        step = SimpleNamespace(pipeline=SimpleNamespace(config={"simulation_steps": 4, "cores": []}),
                               add_entry=lambda key, value, kind: entries.__setitem__(key, value))
        emission.emit_deployment_record_hcm(
            step, built_chain(), platform_constraints={"cores": cores, "weight_bits": 8},
            scm_fragment={"reuse_plan": {"params_reloaded": 0}, "relay_cores_inserted": 0},
            programming=SimpleNamespace(neural_stages=0, programming_events=0, params_programmed=0,
                                        params_unique=0),
            crossbar_report=None, spike_gate_result=None, accuracy=0.5, observes_values=True, model=None,
            ir_graph=None, input_shape=(1,), num_classes=2, encoding_placement="subsume")
        return entries["deployment_record_hcm"]["schedule"]

    @staticmethod
    def _cores(schedule):
        return [c for s in schedule["stages"] if "cores" in s for c in s["cores"]]

    def test_an_operand_chip_marks_every_core(self, monkeypatch):
        static = {"max_axons": 8, "max_neurons": 8, "count": 1}
        cores = self._cores(self._schedule(monkeypatch, [static, {**static, "operand": {"payload_bits": 8,
                                                                                        "write_source": "host"}}]))
        assert cores and all(c["operand_only"] is False for c in cores)

    def test_any_other_chip_seals_no_mark(self, monkeypatch):
        cores = self._cores(self._schedule(monkeypatch, [{"max_axons": 8, "max_neurons": 8, "count": 1}]))
        assert cores and all("operand_only" not in c for c in cores)
