"""E85.W: exposed write time at batch K, serial and overlapped, from the period census."""

import random

import pytest

from mimarsinan.deployment_record.build.write_census import PassRow, ProgramPasses
from mimarsinan.deployment_record.cost.absolute import price_write_terms
from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs, price_write_inputs
from mimarsinan.deployment_record.platform_physics import get_platform_physics, profile_from_dict
from mimarsinan.deployment_record.quantities.write_law import WritePolicy, plane_writes
from mimarsinan.mapping.support.period_census import PassCensus, PassPeriodCensus

from unit.deployment_record.g3_fixtures import constant

_TRUENORTH = get_platform_physics("truenorth")


def _pass(position, *, payload, steps=0, cores=4, resident=False):
    return PassCensus(position=position, segment_index=0, resident=resident, cores=cores,
                      resident_cores=cores if resident else 0, written_cells_backing_store=0 if resident else payload,
                      written_cells_producer=0, payload_bytes=0 if resident else payload, compute_steps=steps)


def _exposed(census, physics, policy):
    pricing = price_write_terms(census, physics, policy)
    terms = {t.name: t for t in pricing.terms}
    return terms.get("latency_write_exposed_s"), {r.name: r.reason for r in pricing.refusals}


def _policy(k, mode="serial", **extra):
    """A one-type chip of ``cores`` declared cores (64 by default)."""
    cores = extra.pop("cores", 64)
    return WritePolicy(batch_k=k, overlap_mode=mode, core_capacity=(((256, 256), cores),), **extra)


class TestSerial:
    @pytest.mark.parametrize("k", [1, 16, 64])
    def test_a_flat_fit_exposes_no_write(self, k):
        census = PassPeriodCensus(passes=(_pass(0, payload=1000, resident=True),))
        for mode in ("serial", "overlapped"):
            term, _ = _exposed(census, _TRUENORTH, _policy(k, mode))
            assert term.value == 0.0 and term.band.high == 0.0
        energy = {t.name: t for t in price_write_terms(census, _TRUENORTH, _policy(k)).terms}["energy_write_mj"]
        assert energy.value == 0.0

    @pytest.mark.parametrize("k", [1, 16, 64])
    def test_an_ex_e_like_program_exposes_its_payload_over_k(self, k):
        """9 of 9 passes reprogram: the serial term is t_program_per_byte x bytes / K."""
        payloads = [41_000 + 977 * i for i in range(9)]
        census = PassPeriodCensus(passes=tuple(_pass(i, payload=p, steps=187) for i, p in enumerate(payloads)))
        term, _ = _exposed(census, _TRUENORTH, _policy(k))
        expected = _TRUENORTH.band("t_program_per_byte").nominal * sum(payloads) / k
        assert abs(term.value - expected) <= 1e-12 * expected


def _overlap_physics():
    return profile_from_dict({
        "format_version": 1, "name": "o", "display_name": "O", "description_file": "o.md",
        "validity": {"measurement_kind": "projection", "technology_node_nm": 22.0},
        "constants": {"t_program_per_byte": constant(10.0, unit="ns", low=5.0, high=20.0),
                      "t_cycle": constant(1.0, unit="us", low=0.5, high=2.0)},
    })


def _random_census(rng):
    passes = tuple(_pass(i, payload=rng.randint(0, 5000), steps=rng.randint(0, 40), cores=rng.randint(1, 8))
                   for i in range(rng.randint(1, 6)))
    return PassPeriodCensus(passes=passes)


class TestOverlapped:
    def test_overlapped_equals_serial_when_nothing_computes(self):
        census = PassPeriodCensus(passes=(_pass(0, payload=3000), _pass(1, payload=1000)))
        serial, _ = _exposed(census, _overlap_physics(), _policy(4))
        overlapped, _ = _exposed(census, _overlap_physics(), _policy(4, "overlapped"))
        assert (overlapped.band.low, overlapped.value, overlapped.band.high) == pytest.approx(
            (serial.band.low, serial.value, serial.band.high))

    def test_a_write_hides_behind_the_previous_pass_compute(self):
        census = PassPeriodCensus(passes=(_pass(0, payload=100, steps=1), _pass(1, payload=3000, steps=1)))
        overlapped, _ = _exposed(census, _overlap_physics(), _policy(1, "overlapped"))
        # Nominal: max(0, 3000 x 10 ns - 1 us) + max(0, 100 x 10 ns - 1 us) = 29 us + 0.
        assert overlapped.value == pytest.approx(29e-6)

    def test_each_write_hides_behind_the_compute_of_the_pass_before_it(self):
        """Unequal compute steps: pass q+1's write overlaps pass q's compute, corner by corner."""
        census = PassPeriodCensus(passes=(_pass(0, payload=1000, steps=1), _pass(1, payload=3000, steps=2),
                                          _pass(2, payload=200, steps=5)))
        overlapped, _ = _exposed(census, _overlap_physics(), _policy(1, "overlapped"))
        # t_w = payload x (5, 10, 20) ns, t_c = steps x (0.5, 1, 2) us; write corner low pairs compute high.
        # q=0: max(0, 15 - 2) | 30 - 1 | 60 - 0.5; q=1: max(0, 1 - 4) | max(0, 2 - 2) | 4 - 1;
        # q=2 (wraps to pass 0's write): max(0, 5 - 10) | 10 - 5 | 20 - 2.5   [us]
        assert (overlapped.band.low, overlapped.value, overlapped.band.high) == pytest.approx(
            (13e-6, 34e-6, 80e-6), rel=1e-12)

    def test_200_seeded_fixtures_overlap_at_most_serial_and_never_rise_with_k(self):
        rng = random.Random(0)
        physics = _overlap_physics()
        for _ in range(200):
            census = _random_census(rng)
            previous = None
            for k in (1, 2, 4, 16, 64, 256):
                serial, _ = _exposed(census, physics, _policy(k))
                overlapped, _ = _exposed(census, physics, _policy(k, "overlapped"))
                for corner in ("low", "nominal", "high"):
                    assert getattr(overlapped.band, corner) <= getattr(serial.band, corner) + 1e-18
                if previous is not None:
                    for corner in ("low", "nominal", "high"):
                        assert getattr(serial.band, corner) <= getattr(previous[0].band, corner) + 1e-18
                        assert getattr(overlapped.band, corner) <= getattr(previous[1].band, corner) + 1e-18
                previous = (serial, overlapped)

    def test_a_pass_wider_than_the_shadowed_cores_refuses_by_name(self):
        census = PassPeriodCensus(passes=(_pass(0, payload=100, cores=8), _pass(1, payload=100, cores=8)))
        term, refusals = _exposed(census, _overlap_physics(),
                                  _policy(4, "overlapped", cores=8, shadow_capacity_fraction=0.25))
        assert term is None
        assert "shadow_capacity_fraction" in refusals["latency_write_exposed_s"]
        assert "256 x 256" in refusals["latency_write_exposed_s"]

    def test_the_shadow_room_is_the_declared_cores_never_the_planner_budget(self):
        """8 declared cores at s = 0.25 keep 6 to compute: a 6-core pass overlaps (the planner's 0.8x
        budget would have left only floor(6.4 x 0.75) = 4)."""
        census = PassPeriodCensus(passes=(_pass(0, payload=100, cores=6, steps=1), _pass(1, payload=100, cores=6)))
        term, _ = _exposed(census, _overlap_physics(), _policy(4, "overlapped", cores=8, shadow_capacity_fraction=0.25))
        assert term is not None

    def test_the_shadow_room_is_exact_at_a_decimal_fraction(self):
        """10 cores at s = 0.9 keep exactly 1 (10 x (1 - 0.9) is 0.9999999999999998 in floats)."""
        census = PassPeriodCensus(passes=(_pass(0, payload=100, cores=1, steps=1),))
        term, _ = _exposed(census, _overlap_physics(), _policy(4, "overlapped", cores=10, shadow_capacity_fraction=0.9))
        assert term is not None

    def test_a_policy_without_declared_core_types_refuses_by_name(self):
        census = PassPeriodCensus(passes=(_pass(0, payload=100, steps=1),))
        policy = WritePolicy(batch_k=1, overlap_mode="overlapped", shadow_capacity_fraction=0.5)
        term, refusals = _exposed(census, _overlap_physics(), policy)
        assert term is None and "core type" in refusals["latency_write_exposed_s"]


    def test_overlap_needs_the_compute_steps(self):
        census = PassPeriodCensus(passes=(_pass(0, payload=100),))
        pricing = price_write_terms(census, _overlap_physics(), _policy(4, "overlapped"), compute_sealed=False)
        assert "compute steps" in {r.name: r.reason for r in pricing.refusals}["latency_write_exposed_s"]

    def test_without_a_write_time_the_term_refuses_by_name(self):
        physics = profile_from_dict({
            "format_version": 1, "name": "n", "display_name": "N", "description_file": "n.md",
            "validity": {"measurement_kind": "projection"}, "constants": {}})
        census = PassPeriodCensus(passes=(_pass(0, payload=100),))
        _, refusals = _exposed(census, physics, _policy(1))
        assert "t_write_per_unit" in refusals["latency_write_exposed_s"]

    def test_the_unit_path_reads_granularity_iterations_and_setup(self):
        physics = profile_from_dict({
            "format_version": 1, "name": "u", "display_name": "U", "description_file": "u.md",
            "validity": {"measurement_kind": "projection"},
            "constants": {"t_write_per_unit": constant(1.0, unit="us"),
                          "t_write_setup_per_event": constant(0.5, unit="us"),
                          "write_verify_iterations": constant(4.0, unit="iterations")}})
        census = PassPeriodCensus(passes=(_pass(0, payload=10), _pass(1, payload=6)))
        policy = _policy(2, write_granularity="cell", write_units_in_parallel=2, write_ports=1)
        term, _ = _exposed(census, physics, policy)
        # ceil(10 / 2) + ceil(6 / 2) = 8 write steps of (4 x 1 us + 0.5 us), per period over K = 2.
        assert term.value == pytest.approx(8 * 4.5e-6 / 2)
        _, refusals = _exposed(census, physics, _policy(2))
        assert "granularity" in refusals["latency_write_exposed_s"]


_A, _B = (784, 512), (512, 256)


def _typed_row(cells, types, steps):
    return PassRow(segment_index=0, resident=False, core_cells=tuple(cells), marks=frozenset(), core_extents=None,
                   compute_steps=steps, stage_latencies=None, core_types=None if types is None else tuple(types))


def _plane(rows, counts, mode, fraction=0.5):
    platform = {"cores": [{"max_axons": 784, "max_neurons": 512, "count": counts[0], "write_granularity": "cell"},
                          {"max_axons": 512, "max_neurons": 256, "count": counts[1], "write_granularity": "cell"}],
                "weight_bits": 8, "inference_batch_K": 1,
                "write_overlap": {"mode": mode, "shadow_capacity_fraction": fraction}}
    writes = plane_writes(ProgramPasses(rows=tuple(rows)), platform, weight_bits=8, timesteps=32, label="hand")
    pricing = price_write_inputs(WriteInputs.of(writes), get_platform_physics("isaac_like"))
    return {t.name: t for t in pricing.terms}, {r.name: r.reason for r in pricing.refusals}


class TestTheShadowRoomOfEachCoreType:
    """The reviewer's witness: passes (A, B) and (A) on A = 1, B = 9 at s = 0.5. The chip-wide 0.8x budget
    left int(8 x 0.5) = 4 cores to compute and admitted the overlap, but type A has one core, so no shadow
    core of type A exists to take the next pass's A tile while the current pass computes on A."""

    _ROWS = (_typed_row([4000, 1000], [_A, _B], 200), _typed_row([4000], [_A], 200))

    def test_a_type_with_no_shadow_core_refuses_the_overlap_by_name(self):
        terms, refusals = _plane(self._ROWS, (1, 9), "overlapped")
        assert "latency_write_exposed_s" not in terms
        reason = refusals["latency_write_exposed_s"]
        assert "784 x 512" in reason and "shadow_capacity_fraction" in reason
        serial, _ = _plane(self._ROWS, (1, 9), "serial")
        assert serial["latency_write_exposed_s"].value > 0

    def test_each_type_with_shadow_room_overlaps(self):
        """Three A tiles on two A cores rewrite; each pass computes on one A core, inside floor(2 x 0.5)."""
        rows = self._ROWS + (_typed_row([4000], [_A], 200),)
        terms, refusals = _plane(rows, (2, 9), "overlapped")
        serial, _ = _plane(rows, (2, 9), "serial")
        assert "latency_write_exposed_s" in terms, refusals
        assert 0 < serial["latency_write_exposed_s"].value
        assert terms["latency_write_exposed_s"].value <= serial["latency_write_exposed_s"].value

    def test_an_untyped_plane_on_a_mixed_chip_refuses_the_overlap_by_name(self):
        """Untyped passes rotate (serial prices); the shadow room of each type is unknown, so the overlapped
        term refuses by name."""
        rows = (_typed_row([4000, 1000], None, 200), _typed_row([4000], None, 200))
        terms, refusals = _plane(rows, (1, 1), "overlapped")
        assert "latency_write_exposed_s" not in terms
        assert "no core type" in refusals["latency_write_exposed_s"] and "undecidable" in refusals["latency_write_exposed_s"]
        serial, _ = _plane(rows, (1, 1), "serial")
        assert serial["latency_write_exposed_s"].value > 0
