"""Shared construction of hybrid mappings and HCM spiking flows for pipeline steps."""

from __future__ import annotations

from typing import Any

import torch

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.mapping.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.mapping.ir import IRGraph
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow


def build_hybrid_mapping_for_pipeline(
    ir_graph: IRGraph,
    platform_constraints: dict[str, Any],
) -> Any:
    return build_hybrid_hard_core_mapping(
        ir_graph=ir_graph,
        cores_config=platform_constraints["cores"],
        allow_neuron_splitting=bool(platform_constraints.get("allow_neuron_splitting", False)),
        allow_scheduling=bool(platform_constraints.get("allow_scheduling", False)),
        allow_coalescing=bool(platform_constraints.get("allow_coalescing", False)),
    )


def build_spiking_hybrid_flow(
    pipeline,
    hybrid_mapping,
    *,
    preprocessor=None,
) -> SpikingHybridCoreFlow:
    cfg = pipeline.config
    flow = SpikingHybridCoreFlow(
        cfg["input_shape"],
        hybrid_mapping,
        int(cfg["simulation_steps"]),
        preprocessor,
        cfg["firing_mode"],
        cfg["spike_generation_mode"],
        cfg["thresholding_mode"],
        spiking_mode=cfg.get("spiking_mode", "lif"),
    )
    return flow.to(cfg["device"])


def run_hcm_spiking_test(
    pipeline,
    flow: SpikingHybridCoreFlow,
    *,
    device: str | None = None,
    max_batch_cap: int | None = None,
    retry_on_oom: bool = False,
) -> float:
    """Run soft-core / HCM metric test with subsample or batch limit from config."""
    device = device or pipeline.config["device"]
    sim_batches = pipeline.config.get("simulation_batch_count", None)
    max_samples = int(pipeline.config.get("max_simulation_samples", 0) or 0)
    attempt_cap = max_batch_cap
    last_error: Exception | None = None

    for attempt in range(2 if retry_on_oom else 1):
        trainer = BasicTrainer(
            flow,
            device,
            DataLoaderFactory(pipeline.data_provider_factory),
            None,
        )
        try:
            if attempt_cap is not None:
                trainer.set_test_batch_size(
                    min(int(trainer.test_batch_size), int(attempt_cap))
                )
            if max_samples > 0:
                return float(
                    trainer.test_on_subsample(
                        max_samples=max_samples,
                        seed=int(pipeline.config.get("seed", 0)),
                    )
                )
            return float(trainer.test(max_batches=sim_batches))
        except RuntimeError as exc:
            last_error = exc
            oom_cls = getattr(torch.cuda, "OutOfMemoryError", ())
            is_oom = isinstance(exc, oom_cls) or "out of memory" in str(exc).lower()
            if not retry_on_oom or not is_oom or attempt >= 1:
                raise
            attempt_cap = int(pipeline.config.get("simulation_batch_size", 8))
        finally:
            trainer.close()

    if last_error is not None:
        raise last_error
    raise RuntimeError("run_hcm_spiking_test failed without raising")


def record_hcm_reference(
    pipeline,
    hybrid_mapping,
    sample: torch.Tensor,
    *,
    sample_index: int = 0,
    device: str | None = None,
):
    """Build HCM flow and return ``(flow, record)`` from ``forward_with_recording``."""
    device = device or pipeline.config["device"]
    flow = build_spiking_hybrid_flow(pipeline, hybrid_mapping).to(device).eval()
    with torch.no_grad():
        _out, ref = flow.forward_with_recording(
            sample.to(device), sample_index=sample_index
        )
    return flow, ref


def run_hcm_mapping_metric(
    pipeline,
    ir_graph: IRGraph,
    platform_constraints: dict[str, Any],
    *,
    hybrid_mapping: Any | None = None,
    cache_key: str = "hybrid_mapping",
    device: str | None = None,
    max_batch_cap: int | None = None,
    retry_on_oom: bool = False,
    outer_oom_retry: bool = False,
) -> float | None:
    """Build hybrid mapping, run HCM spiking test, optionally cache mapping.

    Returns accuracy or None on non-fatal failure when ``outer_oom_retry`` handles OOM.
    """
    device = device or pipeline.config["device"]
    attempt_cap = max_batch_cap
    last_exc: Exception | None = None

    for attempt in range(2 if outer_oom_retry else 1):
        try:
            if hybrid_mapping is None:
                hybrid_mapping = build_hybrid_mapping_for_pipeline(
                    ir_graph, platform_constraints
                )
                pipeline.cache.add(cache_key, hybrid_mapping, "pickle")
            flow = build_spiking_hybrid_flow(pipeline, hybrid_mapping)
            return float(
                run_hcm_spiking_test(
                    pipeline,
                    flow,
                    device=device,
                    max_batch_cap=attempt_cap,
                    retry_on_oom=retry_on_oom,
                )
            )
        except RuntimeError as exc:
            last_exc = exc
            oom_cls = getattr(torch.cuda, "OutOfMemoryError", ())
            is_oom = isinstance(exc, oom_cls) or "out of memory" in str(exc).lower()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not outer_oom_retry or not is_oom or attempt >= 1:
                raise
            attempt_cap = int(pipeline.config.get("simulation_batch_size", 8))
        except Exception:
            raise

    if last_exc is not None:
        raise last_exc
    return None


def assert_spike_parity_or_raise(ref, actual) -> None:
    """Raise ``AssertionError`` with ``format_first_diff`` when records diverge."""
    from mimarsinan.chip_simulation.spike_recorder import compare_records, format_first_diff

    diffs = compare_records(ref, actual)
    if diffs:
        raise AssertionError(format_first_diff(diffs))
