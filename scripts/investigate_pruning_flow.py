"""
Investigation script: pruning flow from model → IR → soft cores → hard cores.

Runs a real MNIST pipeline with pruning enabled and captures:
- ir_graph after Soft Core Mapping (pruned IR)
- hard_core_mapping after Hard Core Mapping

Then reports: initial masks, IR node mask lengths/shapes, segment soft core
mask presence and shapes (via PRUNING_INVESTIGATION logging), and final hard
core dimensions plus all-zero row/column checks.

Usage (from repo root):
  PRUNING_INVESTIGATION=1 PYTHONPATH=src python scripts/investigate_pruning_flow.py [config.json]

Default config: examples/mnist_arch_search_nsga2_ttfs.json (must have pruning enabled).
"""

from __future__ import annotations

import json
import os
import sys


def _setup_path():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(repo_root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return repo_root


def _load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def _resolve_platform_constraints(deployment_config: dict) -> tuple[dict, dict]:
    deployment_parameters = deployment_config["deployment_parameters"]
    platform_constraints_raw = deployment_config["platform_constraints"]
    if isinstance(platform_constraints_raw, dict) and "mode" in platform_constraints_raw:
        mode = platform_constraints_raw.get("mode", "user")
        if mode == "user":
            platform_constraints = platform_constraints_raw.get(
                "user",
                {k: v for k, v in platform_constraints_raw.items() if k != "mode"},
            )
        elif mode == "auto":
            auto = platform_constraints_raw.get("auto", {}) or {}
            fixed = auto.get("fixed", {}) or {}
            search_space = auto.get("search_space", {}) or {}
            arch_cfg = deployment_parameters.setdefault("arch_search", {})
            for k, v in search_space.items():
                arch_cfg.setdefault(k, v)
            platform_constraints = fixed
        else:
            raise ValueError(f"Invalid platform_constraints.mode: {mode}")
    else:
        platform_constraints = platform_constraints_raw
    return deployment_parameters, platform_constraints


def _report_ir_graph(ir_graph, label: str) -> None:
    """Log neural core shapes and pruning masks after pruning."""
    try:
        from mimarsinan.mapping.ir import NeuralCore
    except ImportError:
        print(f"[{label}] Cannot import IR; skipping ir_graph report.")
        return
    neural_cores = [n for n in ir_graph.nodes if isinstance(n, NeuralCore)]
    print(f"\n--- {label} ---")
    print(f"Neural cores: {len(neural_cores)}")
    for node in neural_cores:
        try:
            mat = node.get_core_matrix(ir_graph)
            shape = mat.shape
        except Exception:
            shape = getattr(node.core_matrix, "shape", None) or "?"
        row_mask = getattr(node, "pruned_row_mask", None)
        col_mask = getattr(node, "pruned_col_mask", None)
        n_axons = shape[0] if isinstance(shape, tuple) else 0
        n_neurons = shape[1] if isinstance(shape, tuple) and len(shape) > 1 else 0
        sum_row = sum(row_mask) if row_mask else 0
        sum_col = sum(col_mask) if col_mask else 0
        len_row = len(row_mask) if row_mask else 0
        len_col = len(col_mask) if col_mask else 0
        bank = getattr(node, "weight_bank_id", None)
        print(
            f"  node_id={node.id} shape={shape} "
            f"pruned_row_mask len={len_row} sum={sum_row} "
            f"pruned_col_mask len={len_col} sum={sum_col} "
            f"weight_bank_id={bank}"
        )
        if row_mask and n_axons and len_row != n_axons:
            print(f"    WARNING row_mask length {len_row} != matrix rows {n_axons}")
        if col_mask and n_neurons and len_col != n_neurons:
            print(f"    WARNING col_mask length {len_col} != matrix cols {n_neurons}")


def _report_hard_core_mapping(hard_core_mapping) -> None:
    """Report each segment's hard cores: dimensions and all-zero row/column check."""
    import numpy as np
    try:
        segments = hard_core_mapping.get_neural_segments()
    except Exception:
        segments = getattr(hard_core_mapping, "stages", [])
        segments = [s.hard_core_mapping for s in segments if getattr(s, "kind", None) == "neural" and getattr(s, "hard_core_mapping", None)]
    print("\n--- Final Hard Core Mapping ---")
    for seg_idx, hcm in enumerate(segments):
        cores = getattr(hcm, "cores", [])
        print(f"Segment {seg_idx}: {len(cores)} hard cores")
        for cidx, hc in enumerate(cores):
            mat = np.asarray(hc.core_matrix, dtype=np.float64)
            n_axons, n_neurons = mat.shape
            zero_rows = np.where(np.all(np.abs(mat) < 1e-12, axis=1))[0]
            zero_cols = np.where(np.all(np.abs(mat) < 1e-12, axis=0))[0]
            print(
                f"  core {cidx} shape={mat.shape} "
                f"axons_per_core={getattr(hc,'axons_per_core',n_axons)} "
                f"neurons_per_core={getattr(hc,'neurons_per_core',n_neurons)} "
                f"all_zero_rows={len(zero_rows)} all_zero_cols={len(zero_cols)}"
            )
            if len(zero_rows) > 0 or len(zero_cols) > 0:
                print(f"    zero_row_indices={zero_rows.tolist()[:20]}{'...' if len(zero_rows)>20 else ''}")
                print(f"    zero_col_indices={zero_cols.tolist()[:20]}{'...' if len(zero_cols)>20 else ''}")


def main() -> None:
    repo_root = _setup_path()
    os.chdir(repo_root)

    # Enable temporary logging in pipeline steps and mapping
    os.environ["PRUNING_INVESTIGATION"] = "1"

    from init import init
    init()

    from mimarsinan.common.reporter import DefaultReporter
    from mimarsinan.pipelining.pipelines.deployment_pipeline import DeploymentPipeline
    from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory
    import mimarsinan.data_handling.data_providers  # noqa: F401

    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(repo_root, "examples", "mnist_arch_search_nsga2_ttfs.json")
    deployment_config = _load_config(config_path)
    deployment_parameters, platform_constraints = _resolve_platform_constraints(deployment_config)
    deployment_name = deployment_config["experiment_name"]
    pipeline_mode = deployment_config.get("pipeline_mode", "phased")
    working_directory = os.path.join(
        deployment_config["generated_files_path"],
        deployment_name + "_" + pipeline_mode + "_deployment_run",
    )
    os.makedirs(working_directory, exist_ok=True)
    os.makedirs(os.path.join(working_directory, "_RUN_CONFIG"), exist_ok=True)
    with open(os.path.join(working_directory, "_RUN_CONFIG", "config.json"), "w") as f:
        json.dump(deployment_config, f, indent=4)

    # Ensure pruning is enabled
    if not deployment_parameters.get("pruning", False) or float(deployment_parameters.get("pruning_fraction", 0)) <= 0:
        print("WARNING: config has pruning disabled or pruning_fraction=0; enabling for investigation.")
        deployment_parameters["pruning"] = True
        deployment_parameters.setdefault("pruning_fraction", 0.1)

    # Run from Pruning Adaptation through Hard Core Mapping
    start_step = "Pruning Adaptation"
    stop_step = "Hard Core Mapping"
    merged_params = dict(deployment_parameters)
    DeploymentPipeline.apply_preset(pipeline_mode, merged_params)

    reporter = DefaultReporter()
    pipeline = DeploymentPipeline(
        data_provider_factory=BasicDataProviderFactory(
            deployment_config["data_provider_name"],
            "./datasets",
            seed=deployment_config.get("seed", 0),
        ),
        deployment_parameters=merged_params,
        platform_constraints=platform_constraints,
        reporter=reporter,
        working_directory=working_directory,
    )

    captured = {"ir_graph_after_soft_core_mapping": None, "hard_core_mapping": None}

    def post_hook(step_name: str, step) -> None:
        pipe = step.pipeline
        if step_name == "Soft Core Mapping":
            key = "Soft Core Mapping.ir_graph"
            if key in pipe.cache.keys():
                captured["ir_graph_after_soft_core_mapping"] = pipe.cache.get(key)
        elif step_name == "Hard Core Mapping":
            key = "Hard Core Mapping.hard_core_mapping"
            if key in pipe.cache.keys():
                captured["hard_core_mapping"] = pipe.cache.get(key)

    pipeline.register_post_step_hook(post_hook)

    print("Running pipeline from Pruning Adaptation to Hard Core Mapping...")
    pipeline.run_from(step_name=start_step, stop_step=stop_step)

    try:
        reporter.finish()
    except Exception:
        pass

    # Report from captured data
    if captured["ir_graph_after_soft_core_mapping"] is not None:
        _report_ir_graph(captured["ir_graph_after_soft_core_mapping"], "IR graph after Soft Core Mapping")
    else:
        print("\n--- IR graph after Soft Core Mapping --- not captured (cache key missing?)")

    if captured["hard_core_mapping"] is not None:
        _report_hard_core_mapping(captured["hard_core_mapping"])
    else:
        print("\n--- Hard Core Mapping --- not captured (cache key missing?)")

    print("\nInvestigation run complete.")


if __name__ == "__main__":
    main()
