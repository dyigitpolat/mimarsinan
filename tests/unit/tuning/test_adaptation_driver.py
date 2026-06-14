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
