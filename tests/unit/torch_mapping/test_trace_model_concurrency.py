"""Regression: FX-tracing concurrency and stale-patch recovery.

Root cause (observed in GUI): ``torch.fx`` patches ``nn.Module.__call__`` per
tracer without thread-safety. Concurrent traces (e.g. overlapping GUI
requests) can leave a stale wrapper installed — ``ShapeProp`` later hits it
and raises ``NameError: module is not installed as a submodule``.
``trace_model`` serialises with a lock and detects stale patches.
"""

from __future__ import annotations

import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import VisionTransformer

from mimarsinan.torch_mapping.torch_graph_tracer import (
    _ORIG_MODULE_CALL,
    _ORIG_MODULE_GETATTR,
    trace_model,
)


def _tiny_vit():
    return VisionTransformer(
        image_size=32, patch_size=8,
        num_layers=2, num_heads=4, hidden_dim=64, mlp_dim=128,
        num_classes=10,
    ).eval()


class TestConcurrentTracing:
    def test_parallel_traces_do_not_corrupt_module_call(self):
        errs = []

        def work(_i):
            try:
                trace_model(_tiny_vit(), (3, 32, 32), device="cpu")
            except Exception as e:
                errs.append((type(e).__name__, str(e)[:200]))

        threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errs, f"concurrent traces failed: {errs}"
        assert nn.Module.__call__ is _ORIG_MODULE_CALL
        assert nn.Module.__getattr__ is _ORIG_MODULE_GETATTR


class TestSequentialTraces:
    def test_two_sequential_traces_leave_call_restored(self):
        trace_model(_tiny_vit(), (3, 32, 32), device="cpu")
        assert nn.Module.__call__ is _ORIG_MODULE_CALL

        trace_model(_tiny_vit(), (3, 32, 32), device="cpu")
        assert nn.Module.__call__ is _ORIG_MODULE_CALL
