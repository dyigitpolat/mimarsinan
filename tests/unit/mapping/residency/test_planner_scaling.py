"""[F3 round, planner timing] The residency planner scales to programs of thousands of units, with the same plans.

Two terms dominated the planner's wall on route (b)'s lane programs (thousands of tile-group units at C of
thousands of tiles; MEASURED on the unit tree, ``11_wave2/resid/report.md`` section 6):

- the units' live sets: one ``PassCut.cut_width`` per unit boundary, each a scan of every carried wire with a range
  built per wire, so a quadratic-to-cubic count of steps; ``PassCut.cut_widths`` states every boundary's width in one
  pass over the wires (a difference array), the same widths;
- the store front: every position's entries scanned again for every position whose phase window holds it (window x
  front candidates per position), and every position's whole front stored (gigabytes at small r on a 48,499-unit
  program at C = 43,930). The front of a window of positions is the Pareto front of the union of their
  entries, an associative merge, so the store front keeps the window's front by a two-stack queue of merged fronts
  (each entry merged a bounded number of times): the phase excess is 0 over the positions within r of the end and
  (P_j - r) - P_i over the rest, so the window splits into two queues, one keyed by e and one by e - P_i. The merge
  keeps the front's exact tie order (the later start among equal entries, and with a segment cap the order that keeps
  either rule's winner), so the store is the window scan's entry for entry.

The step function needs a front only where it reads a carry: r* is bisected on the exact feasibility of a plan (the
least excess of a partition into at most qmax phases, ``plan_exists``), the floor's carry is probed when a step is
first read, and the stateless plan reads the front at r = C alone.

A walk step asks only whether a probed r carries less than the last step: the probe is the planned store asked for
plans of carry at most the last step's less one, which answers the minimal (carry, Q) exactly where one exists and
None otherwise, and a probe that only reads the best entry keeps the end's entries alone. The walk searches each next
step outward from the last step's r + 1 (that r's probe first, then doubling strides, then a bisection inside the
bracket): route (b)'s lane programs step at nearly every r, where a bisection over (r, C] probed about log2(C) times
per step (MEASURED: C5's whole function, 919 steps, in 3,829 s; S1's first 8 steps in 1,166 s).

The planner reads the store through ``planned_store`` (``bounded_store`` where the bound's tables, qmax rows per weight,
cost less than the windows it prunes): per r, the least excess and least weighted carry of every
completion (``residency_front_bounds``, sliding minima over the phase windows) bound each entry's best final carry
from below and a weighted plan that keeps the budget bounds the minimal carry from above; an entry no minimal plan
passes through is not stored. The tests are monotone in the entry, so the best entry and the phases are the full
store's, and a probe where no completion keeps the budget answers None without a front.
"""

from __future__ import annotations

import bisect
import math
import random

import numpy as np
import pytest

import mimarsinan.mapping.support.schedule.residency_front as RF
from mimarsinan.mapping.support.residency_front_bounds import carry_bound, completion_tables, phase_windows, plan_exists
from mimarsinan.mapping.support.schedule.residency_units import residency_steps, stateless_step
from mimarsinan.mapping.support.residency_front_reference import front_mincarry_np
from mimarsinan.mapping.support.residency_front_store import carries_below, front_mincarry_below, front_mincarry_store
from mimarsinan.mapping.support.residency_front_store import (
    FrontStore, _pareto, bounded_store, front_store, lowest_starts, operand_bound, planned_store,
)
from mimarsinan.mapping.support.schedule.pass_cut import CutNode, PassCut
from mimarsinan.mapping.support.schedule.residency_front import Q_SPAN, plan_steps, q_fit


def _scan_store(sizes, live, C, r, qmax, forced=(), segment_cap=None, operand=()):
    """The store front as the window scan builds it (W03's ``front_store``, with F3's operand bound, verbatim): every
    position's entries in the window of each end position, filtered and kept by ``_pareto``. The reference."""
    n = len(sizes)
    prefix = [0]
    for size in sizes:
        prefix.append(prefix[-1] + int(size))
    P = np.asarray(prefix, dtype=np.int64)
    budget, starts, held = int(C) - int(r), lowest_starts(n, forced), operand_bound(operand)
    ends = set(int(p) for p in forced) | {n}
    cap = 1024
    cols = np.zeros((6, cap), dtype=np.int64)
    cols[4, 0] = -1
    off, size = [0, 1], 1
    for j in range(1, n + 1):
        lo = max(starts[j], bisect.bisect_left(prefix, prefix[j] - int(C), 0, j))
        if held is not None:
            lo = max(lo, bisect.bisect_left(held, held[j] - int(r), 0, j))
        a, b = off[lo], off[j]
        if a < b:
            e0, c0, q0, pos, s0 = cols[0, a:b], cols[1, a:b], cols[2, a:b], cols[3, a:b], cols[5, a:b]
            e = e0 + np.maximum(0, prefix[j] - P[pos] - int(r))
            q, s = q0 + 1, s0 + 1
            ok = (e <= budget) & (q <= int(qmax))
            if segment_cap is not None:
                ok &= s <= int(segment_cap)
            idx = np.nonzero(ok)[0]
            if idx.size:
                cut = 2 * int(live[j - 1]) if j < n else 0
                interior = segment_cap is not None and j not in ends
                e, q, s, c, src = e[idx], q[idx], s[idx], c0[idx] + cut, idx + a
                pick = _pareto(e, c, q, s if interior else None, (src, -pos[idx]))
                m = int(pick.size)
                while size + m > cap:
                    cap *= 2
                    grown = np.zeros((6, cap), dtype=np.int64)
                    grown[:, :size] = cols[:, :size]
                    cols = grown
                cols[0, size:size + m], cols[1, size:size + m], cols[2, size:size + m] = e[pick], c[pick], q[pick]
                cols[3, size:size + m], cols[4, size:size + m] = j, src[pick]
                cols[5, size:size + m] = s[pick] if interior else 0
                size += m
        off.append(size)
    return FrontStore(excess=cols[0, :size], carry=cols[1, :size], phases=cols[2, :size], position=cols[3, :size],
                      source=cols[4, :size], off=tuple(off))


def _same(a: FrontStore, b: FrontStore) -> bool:
    return a.off == b.off and all(np.array_equal(getattr(a, k), getattr(b, k))
                                  for k in ("excess", "carry", "phases", "position", "source"))


def _program(rng, n_hi):
    n = rng.randint(1, n_hi)
    sizes = [rng.choice([1, 1, 1, 2, 3]) if rng.random() < 0.85 else rng.randint(1, 8) for _ in range(n)]
    live = [rng.choice([0, 0, 1, 2, 4, 7]) for _ in range(n - 1)] + [0]
    forced = sorted(rng.sample(range(1, n), rng.randint(0, min(3, n - 1)))) if n > 1 else []
    C = rng.randint(max(sizes), max(max(sizes), sum(sizes) + 2))
    operand = [rng.choice([0, 0, 0, s, rng.randint(0, s)]) for s in sizes]
    return sizes, live, forced, C, operand


def _large(seed, n=5000):
    """A program of ``n`` units of 1 to 9 tiles with spread live sets, at C = ceil(F / 2)."""
    rng = random.Random(seed)
    sizes = [rng.choice([1] * 6 + [2, 2, 3, 4, 9]) for _ in range(n)]
    live = [rng.choice([256, 512, 1024, 2048, 4096, 8192]) * rng.randint(1, 12) for _ in range(n - 1)] + [0]
    return sizes, live, math.ceil(sum(sizes) / 2)


class TestCutWidths:
    @pytest.mark.parametrize("seed", range(4))
    def test_every_boundary_width_in_one_pass_equals_the_cut_width(self, seed):
        rng = random.Random(seed)
        for _ in range(200):
            n = rng.randint(1, 30)
            nodes = [CutNode(core_id=i, latency=i, out_width=rng.randint(1, 9),
                             sources=tuple(sorted(rng.sample(range(i), rng.randint(0, min(3, i)))))) for i in range(n)]
            cuts = sorted(rng.sample(range(1, n), rng.randint(0, n - 1))) if n > 1 else []
            bounds = [0, *cuts, n]
            cut = PassCut.over(nodes, [list(range(a, b)) for a, b in zip(bounds, bounds[1:])])
            widths = cut.cut_widths()
            assert widths == tuple(cut.cut_width(b) for b in range(cut.pass_count - 1))
            assert all(set(cut.live_at(b)) == {w for w in cut.carried if b in w.live_boundaries}
                       for b in range(cut.pass_count))

    @pytest.mark.timeout(120)
    def test_a_cut_of_thousands_of_passes_and_long_wires_states_its_widths(self):
        n = 6000
        nodes = [CutNode(core_id=i, latency=i, out_width=8, sources=((i - 1,) if i else ()) + (
            (i - 3000,) if i >= 3000 else ())) for i in range(n)]
        cut = PassCut.over(nodes, [[i] for i in range(n)])
        widths = cut.cut_widths()
        # a producer below 3000 lives until its reader 3000 later, one above it until the next unit
        assert len(widths) == n - 1 and widths[0] == 8 and widths[2999] == widths[3000] == 8 * 3000
        assert widths[-1] == 16 and widths[4500] == 8 * (6000 - 4500)


class TestTheSlidingFront:
    @pytest.mark.parametrize("seed", range(4))
    def test_the_store_equals_the_window_scans_entry_for_entry(self, seed):
        rng = random.Random(40 + seed)
        for _ in range(250):
            sizes, live, forced, C, operand = _program(rng, 16)
            qmax = q_fit(sizes, C, forced) + rng.choice([0, 1, 2, Q_SPAN])
            cap = rng.choice([None, None, 1, 2, 3])
            for held in ((), operand):
                for r in sorted({0, rng.randint(0, C), C}):
                    assert _same(front_store(sizes, live, C, r, qmax, forced, segment_cap=cap, operand=held,
                                             scan=False),
                                 _scan_store(sizes, live, C, r, qmax, forced, cap, held)), (sizes, live, forced, C, r)

    @pytest.mark.parametrize("seed", range(2))
    def test_a_program_of_hundreds_of_units_has_the_window_scans_store(self, seed):
        rng = random.Random(60 + seed)
        sizes = [rng.choice([1, 1, 1, 2, 3, 5]) for _ in range(300)]
        live = [rng.choice([0, 8, 16, 64, 128, 512]) * rng.randint(1, 5) for _ in range(299)] + [0]
        C = math.ceil(sum(sizes) / 2)
        qmax = q_fit(sizes, C) + Q_SPAN
        for r in (C // 9, C // 3, C):
            assert _same(front_store(sizes, live, C, r, qmax, scan=False), _scan_store(sizes, live, C, r, qmax))

    @pytest.mark.timeout(240)
    def test_the_floor_of_a_program_of_thousands_of_units_plans_within_minutes(self):
        sizes, live, C = _large(0)
        plan = plan_steps(sizes, live, C)
        r, carry, Q = plan.steps[0]
        assert 0 < r < C and Q >= 2 and carry > 0


def _store_cases(rng, n_hi):
    """Random programs, each at a few r, with the bounds the planner reads there (none, the full one, and one asked
    for plans below a carry), for comparing two builds of one store."""
    for _ in range(120):
        sizes, live, forced, C, operand = _program(rng, n_hi)
        qmax = q_fit(sizes, C, forced) + rng.choice([0, 1, 2, Q_SPAN])
        cap = rng.choice([None, None, 1, 2, 3])
        for held in ((), operand):
            for r in sorted({0, rng.randint(0, C), C}):
                full = front_store(sizes, live, C, r, qmax, forced, segment_cap=cap, operand=held, scan=False)
                g = full.best()
                uppers = [None] if g is None else [None, int(full.carry[g]), max(0, int(full.carry[g]) - 1)]
                bounds = [None] + [carry_bound(sizes, live, C, r, qmax, forced, operand_bound(held), cap, u)
                                   for u in uppers]
                for bound in bounds:
                    yield sizes, live, forced, C, r, qmax, cap, held, bound


class TestTheWindowScanDispatch:
    """[Fix round, review finding 6] The store is built by the window scan where the phase window is short and by the
    sliding fronts elsewhere (``residency_front_window.scans``): on short windows the scan's one Pareto pass per
    position costs less than the sliding queues' merges (MEASURED: the review's 240 programs of 20 to 400 units took
    70 s on the F3 tree against 32 s with the scan at the head, the same outputs). Both builds are one store, entry
    for entry, with or without the carry bound and with or without the lean read."""

    @pytest.mark.parametrize("seed", range(4))
    def test_the_scan_is_the_reference_scan_entry_for_entry(self, seed):
        rng = random.Random(140 + seed)
        for _ in range(250):
            sizes, live, forced, C, operand = _program(rng, 16)
            qmax = q_fit(sizes, C, forced) + rng.choice([0, 1, 2, Q_SPAN])
            cap = rng.choice([None, None, 1, 2, 3])
            for held in ((), operand):
                for r in sorted({0, rng.randint(0, C), C}):
                    assert _same(front_store(sizes, live, C, r, qmax, forced, segment_cap=cap, operand=held,
                                             scan=True),
                                 _scan_store(sizes, live, C, r, qmax, forced, cap, held)), (sizes, live, forced, C, r)

    @pytest.mark.parametrize("seed", range(4))
    def test_with_a_bound_and_a_lean_read_the_scan_and_the_sliding_fronts_build_one_store(self, seed):
        rng = random.Random(150 + seed)
        for sizes, live, forced, C, r, qmax, cap, held, bound in _store_cases(rng, 16):
            for keep in (True, False):
                a = front_store(sizes, live, C, r, qmax, forced, segment_cap=cap, operand=held, bound=bound,
                                keep=keep, scan=True)
                b = front_store(sizes, live, C, r, qmax, forced, segment_cap=cap, operand=held, bound=bound,
                                keep=keep, scan=False)
                assert _same(a, b), (sizes, live, forced, C, r, cap, held, keep)

    @pytest.mark.parametrize("seed", range(2))
    def test_on_hundreds_of_units_the_scan_and_the_sliding_fronts_build_one_store(self, seed):
        sizes, live, C = _large(170 + seed, n=400)
        qmax = q_fit(sizes, C) + Q_SPAN
        lo = max(0, math.ceil((sum(sizes) - C) / (qmax - 1)))
        for r in (lo, (lo + C) // 2, C):
            for cap in (None, 4):
                full = front_store(sizes, live, C, r, qmax, segment_cap=cap, scan=False)
                assert _same(front_store(sizes, live, C, r, qmax, segment_cap=cap, scan=True), full)
                g = full.best()
                upper = None if g is None else int(full.carry[g])
                bound = carry_bound(sizes, live, C, r, qmax, (), None, cap, upper)
                for keep in (True, False):
                    assert _same(front_store(sizes, live, C, r, qmax, segment_cap=cap, bound=bound, keep=keep,
                                             scan=True),
                                 front_store(sizes, live, C, r, qmax, segment_cap=cap, bound=bound, keep=keep,
                                             scan=False))

    @pytest.mark.parametrize("cap", [None, 3])
    def test_the_store_scans_where_the_window_is_short_and_slides_elsewhere(self, cap, monkeypatch):
        import mimarsinan.mapping.support.residency_front_store as S
        import mimarsinan.mapping.support.residency_front_window as W

        w = W.SCAN_WINDOW if cap is None else W.SCAN_WINDOW_CAPPED
        assert 0 < W.SCAN_WINDOW <= W.SCAN_WINDOW_CAPPED
        sizes = [1, 2, 1] * 200  # F = 800 over 600 units: the window holds about 0.75 C positions
        F = sum(sizes)
        short, long_ = (w * F) // len(sizes), (w * F) // len(sizes) + 1
        assert S.scans(sizes, short, cap is not None) and not S.scans(sizes, long_, cap is not None)
        seen = []
        real = S._scan
        monkeypatch.setattr(S, "_scan", lambda *a, **k: seen.append(a[2]) or real(*a, **k))
        live = [3] * (len(sizes) - 1) + [0]
        for C in (short, long_):
            qmax = q_fit(sizes, C) + Q_SPAN
            r = C // 2
            got = front_store(sizes, live, C, r, qmax, segment_cap=cap)
            assert _same(got, front_store(sizes, live, C, r, qmax, segment_cap=cap, scan=False))
        assert seen == [short]  # the default read scanned the short window only (the forced reads never dispatch)

    @pytest.mark.parametrize("seed", range(3))
    def test_the_planner_reads_one_step_function_from_either_build(self, seed, monkeypatch):
        import mimarsinan.mapping.support.residency_front_window as W

        rng = random.Random(190 + seed)
        for _ in range(40):
            n = rng.choice([20, 60, 120, 250])
            sizes = [rng.randint(1, rng.choice([1, 2, 3, 5])) for _ in range(n)]
            F = sum(sizes)
            C = max(max(sizes), int(F / rng.choice([1.5, 2, 3, 5])))
            live = [rng.choice([0, rng.randint(1, 50), rng.randint(1, 500)]) for _ in range(n)]
            forced = sorted(rng.sample(range(1, n), rng.choice([0, 1, 3])))
            cap = rng.choice([None, 2, 4, 8])
            got = []
            for window in (0, 10 ** 9):  # never scan, always scan
                monkeypatch.setattr(W, "SCAN_WINDOW", window)
                monkeypatch.setattr(W, "SCAN_WINDOW_CAPPED", window)
                plan = plan_steps(sizes, live, C, forced=forced, segment_cap=cap)
                if plan is None:
                    got.append(None)
                    continue
                parts = [RF.phase_partition(sizes, live, C, s[0], plan.qmax, forced, segment_cap=cap)
                         for s in plan.steps]
                got.append((plan.steps, plan.stateless, parts, plan.at(C)))
            assert got[0] == got[1], (sizes, live, C, forced, cap)


def _min_excess(sizes, C, r, qmax, forced, operand):
    """Brute force: the least excess of any partition within C (no forced cut inside, at most r operand tiles per
    phase, at most qmax phases), or None."""
    import itertools
    n, best = len(sizes), None
    for mask in itertools.product((0, 1), repeat=n - 1):
        cuts = [i + 1 for i, b in enumerate(mask) if b]
        if any(p not in cuts for p in forced):
            continue
        bounds = [0, *cuts, n]
        phases = list(zip(bounds, bounds[1:]))
        widths = [sum(sizes[a:b]) for a, b in phases]
        if max(widths) > C or len(phases) > qmax or any(sum(operand[a:b]) > r for a, b in phases):
            continue
        excess = sum(max(0, R - r) for R in widths)
        best = excess if best is None or excess < best else best
    return best


class TestTheCarryBound:
    @pytest.mark.parametrize("seed", range(3))
    def test_the_least_excess_of_every_completion_is_the_brute_forces(self, seed):
        rng = random.Random(80 + seed)
        for _ in range(200):
            sizes, live, forced, C, operand = _program(rng, 10)
            qmax = q_fit(sizes, C, forced) + rng.choice([0, 1, Q_SPAN])
            for r in sorted({0, rng.randint(0, C), C}):
                held = operand_bound(operand)
                table, _after = completion_tables(phase_windows(sizes, live, C, r, forced, held), r, qmax, (1,), carry=0)
                want = _min_excess(sizes, C, r, qmax, forced, operand)
                assert (int(table[0, qmax, 0]) if table[0, qmax, 0] < 2 ** 59 else None) == want

    @pytest.mark.parametrize("seed", range(4))
    def test_the_planned_store_has_the_full_stores_best_entry_and_phases(self, seed):
        rng = random.Random(90 + seed)
        for _ in range(250):
            sizes, live, forced, C, operand = _program(rng, 16)
            qmax = q_fit(sizes, C, forced) + rng.choice([0, 1, 2, Q_SPAN])
            cap = rng.choice([None, None, 1, 2, 3])
            for held in ((), operand):
                for r in sorted({0, rng.randint(0, C), C}):
                    full = front_store(sizes, live, C, r, qmax, forced, segment_cap=cap, operand=held)
                    g = full.best()
                    for read in (bounded_store, planned_store):
                        planned = read(sizes, live, C, r, qmax, forced, segment_cap=cap, operand=held)
                        if g is None:
                            assert planned is None or planned.best() is None
                            continue
                        h = planned.best()
                        assert (int(planned.carry[h]), int(planned.phases[h])) == (int(full.carry[g]),
                                                                                   int(full.phases[g]))
                        assert planned.path() == full.path() and len(planned.carry) <= len(full.carry)

    @pytest.mark.parametrize("seed", range(2))
    def test_on_hundreds_of_units_the_bound_keeps_the_phases_and_drops_most_entries(self, seed):
        sizes, live, C = _large(100 + seed, n=600)
        qmax = q_fit(sizes, C) + Q_SPAN
        lo = max(0, math.ceil((sum(sizes) - C) / (qmax - 1)))
        for r in (lo, (lo + C) // 2, C):
            full, planned = front_store(sizes, live, C, r, qmax), bounded_store(sizes, live, C, r, qmax)
            if full.best() is None:
                assert planned is None or planned.best() is None
                continue
            assert planned.path() == full.path()
            assert (planned.carry[planned.best()], planned.phases[planned.best()]) == (
                full.carry[full.best()], full.phases[full.best()])
        bound = carry_bound(sizes, live, C, lo, qmax)
        if bound is not None and front_store(sizes, live, C, lo, qmax).best() is not None:
            assert len(bounded_store(sizes, live, C, lo, qmax).carry) < len(front_store(sizes, live, C, lo, qmax).carry)

    @pytest.mark.slow
    @pytest.mark.timeout(600)
    def test_the_floor_of_a_program_of_tens_of_thousands_of_units_plans_within_minutes(self):
        sizes, live, C = _large(7, n=20000)
        plan = plan_steps(sizes, live, C)
        r, carry, Q = plan.steps[0]
        assert 0 < r < C and Q >= 2 and carry > 0



def _counting(monkeypatch):
    calls = []
    store = RF.FRONTS["store"]

    def counted(sizes, live, C, r, qmax, forced=(), **kw):
        calls.append(int(r))
        return store(sizes, live, C, r, qmax, forced, **kw)

    monkeypatch.setitem(RF.FRONTS, "store", counted)
    return calls


class TestTheStepFunctionReadsTheFrontOnlyForCarries:
    @pytest.mark.parametrize("seed", range(3))
    def test_plan_exists_is_the_fronts_feasibility(self, seed):
        rng = random.Random(110 + seed)
        for _ in range(300):
            sizes, live, forced, C, operand = _program(rng, 14)
            qmax = q_fit(sizes, C, forced) + rng.choice([0, 1, Q_SPAN])
            for r in sorted({0, rng.randint(0, C), C}):
                held = operand_bound(operand)
                assert plan_exists(sizes, C, r, qmax, forced, held) == (
                    front_store(sizes, live, C, r, qmax, forced, operand=operand).best() is not None)

    def test_the_stateless_plan_reads_the_front_at_c_alone(self, monkeypatch):
        sizes, live, C = _large(3, n=400)
        calls = _counting(monkeypatch)
        plan = residency_steps(sizes, live, C)
        assert stateless_step(plan)[0] == C and calls == [C]

    def test_the_floor_bisects_on_feasibility_and_reads_the_front_at_r_star(self, monkeypatch):
        """Twelve units of 50 tiles at C = 100: the fluid bound is 39, the floor 46 (twelve phases of 50 keep the
        budget from r = 46 on: 12 (50 - r) <= 100 - r), found by feasibility alone; the front is read at C and r*."""
        sizes, live, C = [50] * 12, [8 * (i % 5 + 1) for i in range(11)] + [0], 100
        qmax = q_fit(sizes, C) + Q_SPAN
        lo = max(0, math.ceil((sum(sizes) - C) / (qmax - 1)))
        monkeypatch.setattr(RF, "STORE_MIN_UNITS", 1)
        calls = _counting(monkeypatch)
        floor = plan_steps(sizes, live, C).step(0)
        assert (lo, floor[0]) == (39, 46) and calls == [C, floor[0]]
        monkeypatch.undo()
        assert plan_steps(sizes, live, C).steps[0] == floor


class TestTheWalksDecisions:
    @pytest.mark.parametrize("seed", range(3))
    def test_a_probe_below_a_ceiling_is_the_exact_answer_or_none(self, seed):
        rng = random.Random(130 + seed)
        for _ in range(200):
            sizes, live, forced, C, operand = _program(rng, 16)
            qmax = q_fit(sizes, C, forced) + Q_SPAN
            for r in sorted({0, rng.randint(0, C), C}):
                exact = front_mincarry_store(sizes, live, C, r, qmax, forced, operand=operand)
                for ceiling in (None, 1, 3, 9, 30, 10 ** 9):
                    got = front_mincarry_below(sizes, live, C, r, qmax, forced, ceiling, operand=operand)
                    want = exact if exact is not None and (ceiling is None or exact[0] < ceiling) else None
                    assert got == want, (sizes, live, forced, C, r, ceiling)

    @pytest.mark.parametrize("seed", range(2))
    def test_a_verdict_of_the_bound_alone_is_the_exact_answers(self, seed):
        decided = 0
        for k in range(4):
            sizes, live, C = _large(170 + 10 * seed + k, n=300)
            qmax = q_fit(sizes, C) + Q_SPAN
            lo = max(0, math.ceil((sum(sizes) - C) / (qmax - 1)))
            for r in (lo, (lo + C) // 3, (lo + C) // 2, C):
                exact = front_mincarry_store(sizes, live, C, r, qmax)
                for ceiling in ([exact[0] - 1, exact[0], exact[0] + 1, 2 * exact[0]] if exact else [1, 10 ** 9]):
                    verdict = carries_below(sizes, live, C, r, qmax, (), ceiling)
                    if verdict is not None:
                        decided += 1
                        assert verdict == (exact is not None and exact[0] < ceiling)
        assert decided > 0

    @pytest.mark.parametrize("density", (1.0, 0.5, 0.05))
    def test_the_walk_finds_the_steps_a_scan_of_every_r_finds(self, density):
        """Each next step is the smallest r above the last step's whose carry falls below it: the steps of a scan of
        every r, on non-increasing functions with a step at every r, at about every other r, and at few r."""
        rng = random.Random(int(160 + 100 * density))
        for _ in range(60):
            C = rng.randint(1, 300)
            r0, carry, table = rng.randint(0, C), rng.randint(C, 10 * C + 10), {}
            for r in range(r0, C + 1):
                if r > r0 and rng.random() < density:
                    carry = max(0, carry - rng.randint(1, 9))
                table[r] = (carry, rng.randint(1, 5))
            want, last = [], None
            for r in range(r0, C + 1):
                if last is None or table[r][0] < last:
                    want.append((r, *table[r]))
                    last = table[r][0]
            assert RF.StepWalk(table.get, C, r0).every() == tuple(want), (C, r0)

    def test_a_function_with_a_step_at_every_r_is_walked_with_one_probe_per_step(self):
        """[F3 round, planner timing] Route (b)'s lane programs step at nearly every r (C5 at 256 x 256: 919 steps over
        r 393 to 1,765; S1: a step at each of r 664 to 695): the next step is probed at the last step's r + 1 first,
        then searched outward, so a step at every r costs one probe per step and not a bisection over (r, C]."""
        C, probed = 4096, []

        def probe(r):
            probed.append(r)
            return (C - r, 1)

        steps = RF.StepWalk(probe, C, 0).every()
        assert len(steps) == C + 1 and steps[-1] == (C, 0, 1)
        assert len(probed) <= 3 * len(steps)

    def test_a_function_with_few_steps_far_apart_is_walked_in_logarithmic_probes(self):
        C, probed = 1 << 16, []

        def probe(r):
            probed.append(r)
            return (1 if r < C - 7 else 0, 1)

        assert RF.StepWalk(probe, C, 0).every() == ((0, 1, 1), (C - 7, 0, 1))
        assert len(probed) <= 2 * 16 + 8

    def test_the_whole_walk_of_hundreds_of_units_equals_the_exact_walk(self):
        rng, steps = random.Random(150), 0
        for _ in range(6):
            sizes = [rng.choice([1, 1, 1, 2, 3, 5]) for _ in range(rng.randint(120, 260))]
            live = [rng.choice([0, 8, 16, 64, 128, 512]) * rng.randint(1, 5) for _ in range(len(sizes) - 1)] + [0]
            forced = sorted(rng.sample(range(1, len(sizes)), rng.randint(0, 3)))
            C = math.ceil(sum(sizes) / rng.choice([2, 3, 5]))
            decided = plan_steps(sizes, live, C, forced=forced)
            exact = plan_steps(sizes, live, C, forced=forced, front=front_mincarry_np)
            assert decided.as_ra_dict() == exact.as_ra_dict()
            steps += len(decided.steps)
        assert steps > 12                                     # the walks hold several steps each on average
