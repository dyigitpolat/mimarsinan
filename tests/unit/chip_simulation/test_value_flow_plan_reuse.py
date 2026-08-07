"""Gather plans are static: build them once per (core, device), not per forward.

Measured (real 4,925-core ViT, 2026-08-07): the deployed census runs 157
batches and each forward rebuilt every segment's gather plans — 5,000
``SpanFillPlan`` constructions costing 18.4 s of a 26.3 s preparation, against
0.3 s of actual matmuls. A plan is a pure function of (core spans, device) and
the spans are static after packing, so reuse is semantics-free.

It must NOT resurrect the +113 GB weight-residency defect: weights stay
chain-scoped ([wsm V3]); only the small index tensors are shared.
"""

import gc

import torch

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow

from unit.chip_simulation.test_value_flow_memory import _token_bank_hybrid
from unit.chip_simulation.value_upload_probe import probe_uploads


def _flow(dtype=torch.float64):
    return ValueHybridCoreFlow(_token_bank_hybrid(), dtype=dtype)


def _x(flow, batch=3):
    width = int(flow.hybrid_mapping.stages[0].hard_core_mapping.cores[0]
                .axons_per_core)
    torch.manual_seed(5)
    return torch.randn(batch, width * 4, dtype=flow.value_dtype)


def _count_plan_builds(monkeypatch):
    from mimarsinan.models.spiking import signal_spans

    calls = []
    real = signal_spans.SpanFillPlan.__init__

    def counting(self, *a, **k):
        calls.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(signal_spans.SpanFillPlan, "__init__", counting)
    return calls


class TestPlansAreBuiltOncePerDevice:
    def test_second_forward_builds_no_new_plans(self, monkeypatch):
        flow = _flow()
        x = _x(flow)
        calls = _count_plan_builds(monkeypatch)
        with torch.no_grad():
            flow(x)
        first = len(calls)
        assert first > 0, "the vehicle must exercise gather plans"
        with torch.no_grad():
            flow(x)
        assert len(calls) == first, (
            f"the second forward rebuilt {len(calls) - first} plans; gather "
            f"plans are static and must be reused across forwards"
        )

    def test_output_plans_reuse_the_mapping_cache(self, monkeypatch):
        """The segment output plan compressed 4,925 sources per stage per
        forward; it must come from the mapping's cached spans."""
        # Count in BOTH namespaces that can compress: value_execution's
        # direct import and the mapping accessor's. Patching only one would
        # make this test unable to fail.
        import mimarsinan.chip_simulation.value_run.value_execution as ve
        import mimarsinan.mapping.packing.softcore.hard_core_mapping as hcm_mod

        calls = []

        def wrap(module, name):
            real = getattr(module, name)

            def counting(*a, **k):
                calls.append(1)
                return real(*a, **k)

            monkeypatch.setattr(module, name, counting)

        wrap(ve, "compress_spike_sources")
        wrap(hcm_mod, "compress_spike_sources")
        flow = _flow()
        x = _x(flow)
        with torch.no_grad():
            flow(x)
            flow(x)
        n_stages = sum(
            1 for st in flow.hybrid_mapping.stages
            if getattr(st, "hard_core_mapping", None) is not None
        )
        assert len(calls) <= n_stages, (
            f"{len(calls)} output-span compressions over 2 forwards of "
            f"{n_stages} neural stages — spans must be compressed once, "
            f"not per forward"
        )


class TestReuseIsBitIdentical:
    def test_repeated_forwards_agree_to_the_bit(self):
        flow = _flow()
        x = _x(flow)
        with torch.no_grad():
            a = flow(x)
            b = flow(x)
        assert [float(v).hex() for v in a.flatten()] == \
               [float(v).hex() for v in b.flatten()]

    def test_matches_a_fresh_flow(self):
        """A cold flow (empty plan cache) must produce identical values."""
        flow = _flow()
        x = _x(flow)
        with torch.no_grad():
            warm = flow(x)
            warm = flow(x)
        fresh = _flow()
        with torch.no_grad():
            cold = fresh(_x(fresh))
        assert [float(v).hex() for v in warm.flatten()] == \
               [float(v).hex() for v in cold.flatten()]


class TestWeightLifetimeIsUnchanged:
    """Plan reuse must not retain weights: the +113 GB defect stays fixed."""

    def test_weights_still_die_with_the_forward(self):
        flow = _flow()
        x = _x(flow)
        with probe_uploads(keep=False) as probe:
            with torch.no_grad():
                flow(x)
            gc.collect()
            assert probe.records, "the vehicle must upload weights"
            live = [r for r in probe.records if r[1]() is not None]
            assert not live, (
                f"{len(live)} weight tensors outlived the forward; plan reuse "
                f"must not extend weight lifetime"
            )
