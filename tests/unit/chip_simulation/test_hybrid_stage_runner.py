"""Hybrid stage runner callback order."""

from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import HybridStageContext, run_hybrid_stages


class _Stage:
    def __init__(self, kind: str, name: str = ""):
        self.kind = kind
        self.name = name


class _Mapping:
    def __init__(self, stages):
        self.stages = stages


def test_callback_order_with_after_hooks():
    log = []
    mapping = _Mapping([
        _Stage("neural", "n1"),
        _Stage("compute", "c1"),
        _Stage("neural", "n2"),
    ])
    buf = {}

    def on_neural(ctx):
        log.append(f"neural:{ctx.stage_index}")

    def on_compute(ctx):
        log.append(f"compute:{ctx.stage_index}")

    def after_neural(ctx):
        log.append(f"after_neural:{ctx.stage_index}")

    def after_compute(ctx):
        log.append(f"after_compute:{ctx.stage_index}")

    run_hybrid_stages(
        mapping,
        buf,
        on_neural=on_neural,
        on_compute=on_compute,
        after_neural=after_neural,
        after_compute=after_compute,
    )
    assert log == [
        "neural:0",
        "after_neural:0",
        "compute:1",
        "after_compute:1",
        "neural:2",
        "after_neural:2",
    ]


def test_legacy_triple_callback_still_works():
    seen = []

    def on_neural(idx, stage, buf):
        seen.append((idx, stage.kind))

    mapping = _Mapping([_Stage("neural")])
    run_hybrid_stages(mapping, {}, on_neural=on_neural, on_compute=lambda *a: None)
    assert seen == [(0, "neural")]
