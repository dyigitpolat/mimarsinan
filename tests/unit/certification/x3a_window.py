"""[U-TGAUGE5] The X3a screen's programs as Soft Core Mapping builds them where a run reads the window.

The screen maps its drawn networks without the depth-balancing relays (``certification/randomized/program.py``), so
a per-cycle draw that holds a live edge at a latency gap other than one is refused by name by the per-cycle executor
(the window rule). A test whose subject is not that gap reads the draw through the mapping's window pass
(``depth_balancing.balance_cycle_windows``, the relays the recipe of every per-cycle cascade arms) for the runs whose
executor reads the window: the per-cycle cascades that neither run level units nor read whole windows.
"""

from __future__ import annotations

from mimarsinan.certification.randomized import executors as ex
from mimarsinan.certification.randomized import program
from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode
from spiking_points import point_of_mode
from mimarsinan.mapping.latency.ir import IRLatency


def reads_the_window(semantics: str) -> bool:
    run = ex.RUNS[semantics]
    if "spiking_mode" not in run.flow:
        return False
    policy = policy_for_spiking_mode(point_of_mode(run.flow["spiking_mode"], run.flow.get("ttfs_cycle_schedule")))
    return (policy.runs_per_cycle_cascade and not run.retimed
            and not run.flow.get("lif_execution_synchronized", False))


def map_through_the_window(monkeypatch) -> None:
    """Every IR the screen maps (``program.integer_ir``) for this test carries the mapping's window pass (dead
    in-segment reads off, then the relays), with the IR latency pass run again."""
    from mimarsinan.mapping.latency.depth_balancing import balance_cycle_windows

    original = program.integer_ir

    def mapped(*args, **kwargs):
        ir = original(*args, **kwargs)
        balance_cycle_windows(ir, cascade=True, armed=True, thresholding_mode="<=")
        IRLatency(ir).calculate()
        return ir

    monkeypatch.setattr(program, "integer_ir", mapped)
