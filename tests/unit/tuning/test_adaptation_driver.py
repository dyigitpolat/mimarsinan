"""AdaptationDriver: run order + scheduler-policy selection (P4)."""

from mimarsinan.tuning.orchestration.adaptation_driver import AdaptationDriver
from mimarsinan.tuning.orchestration.rate_scheduler import RateScheduler

EPS = 2 ** -6


def test_run_drives_scheduler_then_finalizes():
    order = []

    class _Sched:
        def run(self, committed, attempt):
            order.append(("scheduler", committed))
            attempt(1.0)

    driver = AdaptationDriver(
        scheduler=_Sched(),
        attempt=lambda a: order.append(("attempt", a)),
        finalize=lambda: order.append(("finalize",)) or "result",
        committed=0.25,
    )
    result = driver.run()
    assert result == "result"
    assert order == [("scheduler", 0.25), ("attempt", 1.0), ("finalize",)]


def _phase_recording_host(*, catastrophic=False, rolled_back=False, rate=0.5):
    calls = []
    ctx = type("C", (), {
        "is_catastrophic": catastrophic, "rolled_back": rolled_back, "rate": rate,
    })()

    class _Host:
        def _begin_cycle(self, r):
            calls.append("begin"); return ctx
        def _probe_instant(self, c):
            calls.append("probe")
        def _recover(self, c):
            calls.append("recover")
        def _measure_post(self, c):
            calls.append("measure")
        def _rollback_cycle(self, c, outcome):
            calls.append(("rollback", outcome)); return 0.0
        def _commit_cycle(self, c):
            calls.append("commit"); return c.rate

    return _Host(), calls


def test_run_cycle_commit_path_order():
    host, calls = _phase_recording_host()
    out = AdaptationDriver.run_cycle(host, 0.5)
    assert out == 0.5
    assert calls == ["begin", "probe", "recover", "measure", "commit"]


def test_run_cycle_catastrophic_short_circuits_before_recovery():
    host, calls = _phase_recording_host(catastrophic=True)
    out = AdaptationDriver.run_cycle(host, 0.5)
    assert out == 0.0
    # no recover/measure/commit after a catastrophic probe
    assert calls == ["begin", "probe", ("rollback", "catastrophic")]


def test_run_cycle_rollback_after_recovery():
    host, calls = _phase_recording_host(rolled_back=True)
    out = AdaptationDriver.run_cycle(host, 0.5)
    assert out == 0.0
    assert calls == ["begin", "probe", "recover", "measure", ("rollback", "rollback")]


def test_build_scheduler_selects_policy():
    greedy = AdaptationDriver.build_scheduler(
        epsilon=EPS, max_rounds=5, skip_one_shot=False, initial_step=0.5
    )
    assert isinstance(greedy, RateScheduler)
    assert greedy.policy == "greedy_to_one"

    ladder = AdaptationDriver.build_scheduler(
        epsilon=EPS, max_rounds=5, skip_one_shot=True, initial_step=0.125
    )
    assert ladder.policy == "uniform_ladder"
    assert ladder.initial_step == 0.125


def test_build_scheduler_fixed_ladder_policy():
    """``policy_override='fixed_ladder'`` wins and carries the explicit rate list
    (the folded fast genuine-blend path)."""
    sched = AdaptationDriver.build_scheduler(
        epsilon=EPS, max_rounds=5, skip_one_shot=True, initial_step=0.125,
        policy_override="fixed_ladder", rates=[0.5, 1.0],
    )
    assert sched.policy == "fixed_ladder"
    assert sched.rates == [0.5, 1.0]
