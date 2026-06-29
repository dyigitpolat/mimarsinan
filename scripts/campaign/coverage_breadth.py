"""Manifest of post-SSOT breadth work items."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Sequence

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_CAMPAIGN = os.path.join(_REPO, "scripts", "campaign")
if _CAMPAIGN not in sys.path:
    sys.path.insert(0, _CAMPAIGN)
_GPU = os.path.join(_REPO, "scripts", "gpu")
if _GPU not in sys.path:
    sys.path.insert(0, _GPU)

import experiment_matrix as em  # noqa: E402
import scheduler as sch  # noqa: E402
from mimarsinan.chip_simulation.ledger_schema import normalize_ledger_record
from mimarsinan.research.harness import (
    BudgetSchedule,
    DiagnosticCell,
    ExperimentContext,
    FixRecipe,
    RecipePreset,
    VehicleSpec,
    build_deep_cnn_digit_manifest,
    build_fmnist_mixer_manifest,
    build_kmnist_mixer_manifest,
    build_lenet5_digit_manifest,
    build_mnist_mixer_manifest,
    planned_mnist_mixer_ledger_row,
    recipe_preset,
    recipe_registry,
)

_DEFAULT_BASE_CONFIG = os.path.join(
    "docs", "checkpoint_research", "repro", "base_configs", "cifar10_d4_synchronized.json",
)
_IMAGENET_BASE_CONFIG = os.path.join("templates", "imgnet_resnet50_pretrained.json")
_TOP_LEVEL_OVERLAY_KEYS = {
    "data_provider_name",
    "generated_files_path",
    "pipeline_mode",
    "seed",
    "start_step",
    "stop_step",
    "target_metric_override",
}
_AXIS_COORDINATE_KEYS = (
    "model",
    "dataset",
    "spiking_mode",
    "sync",
    "backend",
    "regime",
    "quantization",
    "pruning",
    "mapping_strategy",
    "S",
    "depth",
    "hypervolume_cell_key",
)


def _success_gates() -> dict[str, bool]:
    return {
        "ledger_row_required": True,
        "nf_scm_parity_required": True,
        "cost_or_proxy_required": True,
        "validity_tier_required": True,
    }


def _run_item(
    item_id: str,
    *,
    vehicle: VehicleSpec,
    recipe: FixRecipe,
    budget: BudgetSchedule,
    base_config: str = _DEFAULT_BASE_CONFIG,
    simulation_steps: int = 4,
    rationale: str,
    priority: int = 45,
    need_mb: int = 8000,
    timeout_s: int = 2400,
) -> dict[str, Any]:
    ctx = ExperimentContext(
        vehicle=vehicle,
        recipe=recipe,
        budget=budget,
        seed=0,
        simulation_steps=simulation_steps,
    )
    item = {
        "id": item_id,
        "kind": "run",
        "vehicle": vehicle.name,
        "recipe": recipe.name,
        "budget": budget.name,
        "base_config": base_config,
        "simulation_steps": simulation_steps,
        "config_overlay": ctx.config_overlay(),
        "rationale": rationale,
        "success_gates": _success_gates(),
        "priority": priority,
        "need_mb": need_mb,
        "timeout_s": timeout_s,
    }
    item["axis_coordinates"] = _axis_coordinates(planned_ledger_row(item))
    return item


def _semantic_screen_item(axis: str) -> dict[str, Any]:
    return {
        "id": f"semantic_screen_{axis}",
        "kind": "semantic_screen",
        "axis": axis,
        "artifact": f"docs/research/findings/{axis}_semantic_screen.json",
        "success_gates": {
            "paired_cells_required": True,
            "artifact_required": True,
            "axis_flip_requires_human_review": True,
        },
    }


def _mixer_diagnostic_items(*, enabled: bool = False) -> list[dict[str, Any]]:
    """Expand mixer diagnostic backlog entries across digit workloads."""
    manifests = (
        build_mnist_mixer_manifest(seeds=(0,)),
        build_fmnist_mixer_manifest(seeds=(0,)),
        build_kmnist_mixer_manifest(seeds=(0,)),
    )
    recipes = recipe_registry()
    items: list[dict[str, Any]] = []
    priority = 5
    for manifest in manifests:
        for cell in manifest.cells:
            for recipe_id in cell.recipe_ids:
                recipe = recipes[recipe_id]
                items.append(
                    {
                        "id": f"{cell.cell_id}_{recipe.recipe_id}",
                        "kind": "mixer_diagnostic",
                        "manifest_id": manifest.manifest_id,
                        "cell_id": cell.cell_id,
                        "recipe": recipe.recipe_id,
                        "template": cell.template,
                        "dataset": cell.dataset,
                        "enabled": enabled,
                        "priority": priority,
                        "acceptance": cell.acceptance.to_dict(),
                        "planned_ledger_row": planned_mnist_mixer_ledger_row(
                            run_id=f"{cell.cell_id}_{recipe.recipe_id}_s0",
                            cell=cell,
                            recipe=recipe,
                            seed=0,
                        ),
                    }
                )
                priority += 1
    return items


def _workload_diagnostic_items() -> list[dict[str, Any]]:
    manifests = (
        build_lenet5_digit_manifest(),
        build_deep_cnn_digit_manifest(),
    )
    recipes = recipe_registry()
    items: list[dict[str, Any]] = []
    priority = 200
    for manifest in manifests:
        for cell in manifest.cells:
            for recipe_id in cell.recipe_ids:
                recipe = recipes[recipe_id]
                items.append(
                    {
                        "id": f"{cell.cell_id}_{recipe.recipe_id}",
                        "kind": "workload_diagnostic",
                        "manifest_id": manifest.manifest_id,
                        "cell_id": cell.cell_id,
                        "recipe": recipe.recipe_id,
                        "template": cell.template,
                        "dataset": cell.dataset,
                        "model_type": cell.model_type,
                        "depth": cell.depth,
                        "priority": priority,
                        "acceptance": manifest.acceptance.to_dict(),
                    }
                )
                priority += 1
    return items


def coverage_breadth_manifest() -> list[dict[str, Any]]:
    """Explicit breadth work after the SSOT/ledger path is in place."""
    sync_recipe = recipe_preset("sync_qat_fast_bn")
    lif_preset = recipe_preset("lif_qat_fast_bn")
    imagenet_lif_recipe = FixRecipe(
        lif_preset.name,
        lif_preset.mechanism,
        {**lif_preset.config_overlay(), "preload_weights": True},
        owner=lif_preset.owner,
        rationale=lif_preset.rationale,
    )
    fast_budget = BudgetSchedule(
        "fast_timing_bounded",
        max_tuning_wall_s=20 * 60,
        max_ft_pass_wall_s=300,
        scale_ramp_steps=False,
    )
    return [
        _run_item(
            "residual_deep_cnn_cifar10_sync_d4",
            vehicle=VehicleSpec("residual_deep_cnn", "residual_deep_cnn", "cifar10", depth=4),
            recipe=sync_recipe,
            budget=fast_budget,
            rationale="Compare residual vs plain deep_cnn on the failing CIFAR sync cell.",
        ),
        _run_item(
            "residual_deep_cnn_cifar10_sync_d6",
            vehicle=VehicleSpec("residual_deep_cnn", "residual_deep_cnn", "cifar10", depth=6),
            recipe=sync_recipe,
            budget=fast_budget,
            rationale="Check whether residual protection survives at d6.",
        ),
        _run_item(
            "svhn_deep_cnn_sync_d4",
            vehicle=VehicleSpec("deep_cnn", "deep_cnn", "svhn", depth=4),
            recipe=sync_recipe,
            budget=fast_budget,
            rationale="Close the named SVHN frontier for the narrow breadth claim.",
        ),
        _run_item(
            "cifar100_deep_cnn_sync_d4",
            vehicle=VehicleSpec("deep_cnn", "deep_cnn", "cifar100", depth=4),
            recipe=sync_recipe,
            budget=fast_budget,
            rationale="Extend synchronized deep_cnn breadth to CIFAR-100.",
        ),
        *_mixer_diagnostic_items(enabled=False),
        *_workload_diagnostic_items(),
        _semantic_screen_item("pruning"),
        _semantic_screen_item("regime"),
        _run_item(
            "imagenet_resnet50_adapted_lif",
            vehicle=VehicleSpec("resnet50", "torch_resnet50", "imagenet", depth=50),
            recipe=imagenet_lif_recipe,
            budget=BudgetSchedule(
                "imagenet_capstone_budget",
                max_tuning_wall_s=None,
                max_ft_pass_wall_s=300,
                scale_ramp_steps=True,
            ),
            base_config=_IMAGENET_BASE_CONFIG,
            simulation_steps=32,
            rationale="Measure adapted deployed-SNN accuracy for the ImageNet capstone.",
            priority=70,
            need_mb=24000,
            timeout_s=14400,
        ),
    ]


def _deep_update(target: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _load_base_config(path: str, *, repo: str = _REPO) -> dict[str, Any]:
    full_path = path if os.path.isabs(path) else os.path.join(repo, path)
    with open(full_path) as fh:
        return json.load(fh)


def _merge_overlay(config: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(config)
    deployment = out.setdefault("deployment_parameters", {})
    platform = out.setdefault("platform_constraints", {})
    for key, value in overlay.items():
        if key == "model_config":
            model_config = deployment.setdefault("model_config", {})
            if not isinstance(model_config, dict):
                model_config = {}
                deployment["model_config"] = model_config
            _deep_update(model_config, dict(value))
        elif key == "platform_constraints":
            _deep_update(platform, dict(value))
        elif key == "simulation_steps":
            steps = int(value)
            deployment["simulation_steps"] = steps
            platform["simulation_steps"] = steps
            platform["target_tq"] = steps
        elif key in _TOP_LEVEL_OVERLAY_KEYS:
            out[key] = copy.deepcopy(value)
        else:
            deployment[key] = copy.deepcopy(value)
    return out


def deployment_config_for_item(item: dict[str, Any], *, repo: str = _REPO) -> dict[str, Any]:
    """Return the concrete ``run.py --headless`` config for one run item."""
    if item.get("kind") != "run":
        raise ValueError(f"only run items can become deployment configs: {item.get('id')!r}")
    config = _load_base_config(item["base_config"], repo=repo)
    config = _merge_overlay(config, item["config_overlay"])
    config["experiment_name"] = item["id"]
    config.setdefault("generated_files_path", "./generated")
    config.setdefault("pipeline_mode", "phased")
    config.setdefault("target_metric_override", None)
    config.setdefault("start_step", None)
    config.setdefault("stop_step", None)
    return config


def planned_ledger_row(item: dict[str, Any], *, repo: str = _REPO) -> dict[str, Any]:
    """Normalize the ledger axes a run item is expected to produce once measured."""
    config = deployment_config_for_item(item, repo=repo)
    deployment = config["deployment_parameters"]
    platform = config.get("platform_constraints", {})
    model_config = deployment.get("model_config") or {}
    row: dict[str, Any] = {
        "model": deployment["model_type"],
        "dataset": config["data_provider_name"],
        "spiking_mode": deployment.get("spiking_mode", "lif"),
        "deployment_validity": "VALID_FLAGGED_manifest_planned",
        "S": platform.get("simulation_steps", deployment.get("simulation_steps")),
        "depth": model_config.get("depth", item.get("simulation_steps")),
        "source": "coverage_breadth_manifest",
        "run_id": item["id"],
        "weight_quantization": deployment.get("weight_quantization", False),
        "activation_quantization": deployment.get("activation_quantization", False),
        "preload_weights": deployment.get("preload_weights", False),
        "weight_source": deployment.get("weight_source"),
        "prune_sparsity": deployment.get("prune_sparsity", 0.0),
    }
    if row["spiking_mode"] == "ttfs_cycle_based":
        row["schedule"] = deployment.get("ttfs_cycle_schedule", "synchronized")
    return normalize_ledger_record(row)


def _axis_coordinates(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in _AXIS_COORDINATE_KEYS}


def _expect_artifact(config: dict[str, Any]) -> str:
    generated = config.get("generated_files_path", "./generated")
    pipeline_mode = config.get("pipeline_mode", "phased")
    return os.path.join(
        generated,
        f"{config['experiment_name']}_{pipeline_mode}_deployment_run",
        "__target_metric.json",
    )


def coverage_breadth_queue_manifest(
    *, config_dir: str | None = None, repo: str = _REPO,
) -> list[dict[str, Any]]:
    """Write run configs and return ``research_loop enqueue`` job specs."""
    config_dir = config_dir or os.path.join(repo, "experiments", "campaign")
    os.makedirs(config_dir, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    for item in coverage_breadth_manifest():
        if item["kind"] != "run":
            continue
        config = deployment_config_for_item(item, repo=repo)
        config_path = os.path.join(config_dir, f"{item['id']}.json")
        with open(config_path, "w") as fh:
            json.dump(config, fh, indent=2)
            fh.write("\n")
        config_arg = os.path.relpath(config_path, repo)
        tags = {
            "batch_id": "coverage_breadth",
            "work_item_id": item["id"],
            "recipe": item["recipe"],
            "budget": item["budget"],
        }
        tags.update(item["axis_coordinates"])
        tags.update(item["success_gates"])
        jobs.append({
            "id": item["id"],
            "mode": "fit",
            "need_mb": item["need_mb"],
            "priority": item["priority"],
            "timeout_s": item["timeout_s"],
            "cmd": ["env/bin/python", "run.py", "--headless", config_arg],
            "cwd": repo,
            "expect_artifact": _expect_artifact(config),
            "tags": tags,
        })
    return jobs


def _mixer_batch_id(cell: DiagnosticCell, recipe: RecipePreset) -> str:
    return f"mnist_mixer_diag_{cell.cell_id}_{recipe.recipe_id}"


def _mixer_id_template(cell: DiagnosticCell, recipe: RecipePreset) -> str:
    return f"{_mixer_batch_id(cell, recipe)}_s{{seed}}"


def _mixer_batch_for_cell_recipe(
    cell: DiagnosticCell,
    recipe: RecipePreset,
    *,
    seeds: Sequence[int],
    enabled: bool,
    priority: int,
) -> dict[str, Any]:
    batch = {
        "id": _mixer_batch_id(cell, recipe),
        "template": cell.template,
        "base": dict(recipe.base_overrides),
        "grid": {"seed": [int(s) for s in seeds]},
        "id_template": _mixer_id_template(cell, recipe),
        "priority": int(priority),
        "mode": "fit",
        "need_mb": 8000,
        "timeout_s": 3600,
        "enabled": bool(enabled),
        "acceptance": cell.acceptance.to_dict(),
        "planned_ledger_rows": [
            planned_mnist_mixer_ledger_row(
                run_id=_mixer_id_template(cell, recipe).format(seed=int(seed)),
                cell=cell,
                recipe=recipe,
                seed=int(seed),
            )
            for seed in seeds
        ],
        "tags": {
            "study": "MNIST_MIXER_DIAGNOSTICS",
            "cluster": "MNIST_MIXER_CLOSURE",
            "cell_id": cell.cell_id,
            "diagnostic_role": cell.role,
            "vehicle": "mlp_mixer_core",
            "dataset": "MNIST_DataProvider",
            "firing": cell.firing,
            "sync": cell.sync,
            "backend": cell.backend,
            "acceptance_min_deployed_acc": cell.acceptance.min_deployed_acc,
            "acceptance_max_relative_time": cell.acceptance.max_relative_time,
            **recipe.to_tags(),
        },
    }
    em.validate_batch(batch)
    return batch


def gen_mnist_mixer_diagnostic_batches(
    *,
    seeds: Sequence[int] = (0, 1, 2),
    enabled: bool = False,
) -> list[dict[str, Any]]:
    manifest = build_mnist_mixer_manifest(seeds)
    recipes = recipe_registry()
    batches: list[dict[str, Any]] = []
    priority = 10
    for cell in manifest.cells:
        for recipe_id in cell.recipe_ids:
            batches.append(
                _mixer_batch_for_cell_recipe(
                    cell,
                    recipes[recipe_id],
                    seeds=manifest.seeds,
                    enabled=enabled,
                    priority=priority,
                )
            )
            priority += 1
    return batches


def emit_mnist_mixer_diagnostic_backlog(
    out_path: str,
    *,
    seeds: Sequence[int] = (0, 1, 2),
    enabled: bool = False,
) -> int:
    batches = gen_mnist_mixer_diagnostic_batches(seeds=seeds, enabled=enabled)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(batches, fh, indent=2)
        fh.write("\n")
    return len(batches)


def _mixer_job_for_config(
    job_id: str,
    cfg_path: str,
    cfg: dict[str, Any],
    batch: dict[str, Any],
) -> dict[str, Any]:
    rel_cfg = os.path.relpath(cfg_path, _REPO)
    pmode = cfg.get("pipeline_mode", "phased")
    return {
        "id": job_id,
        "mode": batch.get("mode", "fit"),
        "need_mb": batch.get("need_mb", 8000),
        "priority": batch.get("priority", 50),
        "timeout_s": batch.get("timeout_s", 3600),
        "cmd": ["env/bin/python", "run.py", "--headless", rel_cfg],
        "cwd": _REPO,
        "expect_artifact": f"generated/{job_id}_{pmode}_deployment_run/__target_metric.json",
        "tags": dict(batch.get("tags", {}), batch_id=batch["id"]),
    }


def emit_mnist_mixer_queue_manifest(
    out_path: str,
    *,
    seeds: Sequence[int] = (0, 1, 2),
    config_dir: str | None = None,
) -> int:
    """Write runnable MNIST mixer queue jobs and their config files."""
    config_root = config_dir or os.path.join(_REPO, "experiments", "campaign")
    os.makedirs(config_root, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    for batch in gen_mnist_mixer_diagnostic_batches(seeds=seeds, enabled=True):
        for job_id, cfg in sch.instantiate(batch):
            cfg_path = os.path.join(config_root, f"{job_id}.json")
            with open(cfg_path, "w") as fh:
                json.dump(cfg, fh, indent=2)
                fh.write("\n")
            jobs.append(_mixer_job_for_config(job_id, cfg_path, cfg, batch))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(jobs, fh, indent=2)
        fh.write("\n")
    return len(jobs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    parser.add_argument("--format", choices=("work", "queue"), default="work")
    parser.add_argument(
        "--config-dir",
        default=os.path.join(_REPO, "experiments", "campaign"),
        help="Where --format queue writes generated run.py configs",
    )
    sub = parser.add_subparsers(dest="cmd")

    breadth = sub.add_parser("coverage-breadth")
    breadth.add_argument("--out", default=None)
    breadth.add_argument("--format", choices=("work", "queue"), default="work")
    breadth.add_argument(
        "--config-dir",
        default=os.path.join(_REPO, "experiments", "campaign"),
        help="Where --format queue writes generated run.py configs",
    )

    backlog = sub.add_parser("generate-mnist-mixer")
    backlog.add_argument("--out", default=os.path.join(_REPO, "runs", "campaign", "backlog_mnist_mixer_diagnostics.json"))
    backlog.add_argument("--seeds", default="0,1,2")
    backlog.add_argument("--enabled", action="store_true")

    queue = sub.add_parser("queue-manifest-mnist-mixer")
    queue.add_argument("--out", default=os.path.join(_REPO, "runs", "campaign", "mnist_mixer_queue_manifest.json"))
    queue.add_argument("--config-dir", default=os.path.join(_REPO, "experiments", "campaign"))
    queue.add_argument("--seeds", default="0,1,2")

    args = parser.parse_args(argv)
    if args.cmd in (None, "coverage-breadth"):
        out_path = getattr(args, "out", None)
        fmt = getattr(args, "format", "work")
        config_dir = getattr(args, "config_dir", os.path.join(_REPO, "experiments", "campaign"))
        manifest = (
            coverage_breadth_queue_manifest(config_dir=config_dir)
            if fmt == "queue"
            else coverage_breadth_manifest()
        )
        text = json.dumps(manifest, indent=2)
        if out_path:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            with open(out_path, "w") as fh:
                fh.write(text + "\n")
        else:
            print(text)
        return 0

    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    if args.cmd == "generate-mnist-mixer":
        count = emit_mnist_mixer_diagnostic_backlog(
            args.out,
            seeds=seeds,
            enabled=args.enabled,
        )
    elif args.cmd == "queue-manifest-mnist-mixer":
        count = emit_mnist_mixer_queue_manifest(
            args.out,
            seeds=seeds,
            config_dir=args.config_dir,
        )
    else:
        parser.error(f"unknown command {args.cmd!r}")
    print(json.dumps({"written": count, "out": args.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
