"""SSOT for the tiered integration-run matrices: (re)generates all configs + manifests."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TRAINING_RECIPE = {
    "optimizer": "adamw",
    "scheduler": "cosine",
    "weight_decay": 0.0001,
    "warmup_ratio": 0.05,
    "grad_clip_norm": 1,
    "layer_wise_lr_decay": 1,
    "label_smoothing": 0,
    "betas": [0.9, 0.999],
}
TUNING_RECIPE = {**TRAINING_RECIPE, "warmup_ratio": 0}

PLATFORMS = {
    "A": {"cores": [{"max_axons": 256, "max_neurons": 512, "count": 60, "has_bias": True},
                    {"max_axons": 512, "max_neurons": 256, "count": 60, "has_bias": True}],
          "max_axons": 512, "max_neurons": 512},
    "B": {"cores": [{"max_axons": 784, "max_neurons": 512, "count": 60, "has_bias": True},
                    {"max_axons": 512, "max_neurons": 256, "count": 60, "has_bias": True}],
          "max_axons": 784, "max_neurons": 512},
    "C": {"cores": [{"max_axons": 1024, "max_neurons": 512, "count": 180, "has_bias": True},
                    {"max_axons": 512, "max_neurons": 256, "count": 180, "has_bias": True}],
          "max_axons": 1024, "max_neurons": 512},
    "D": {"cores": [{"max_axons": 576, "max_neurons": 256, "count": 512, "has_bias": True},
                    {"max_axons": 256, "max_neurons": 576, "count": 512, "has_bias": True}],
          "max_axons": 576, "max_neurons": 576},
    "E": {"cores": [{"max_axons": 3072, "max_neurons": 768, "count": 69, "has_bias": True},
                    {"max_axons": 768, "max_neurons": 3072, "count": 69, "has_bias": True}],
          "max_axons": 3072, "max_neurons": 3072},
    "F": {"cores": [{"max_axons": 2304, "max_neurons": 512, "count": 256, "has_bias": True},
                    {"max_axons": 512, "max_neurons": 256, "count": 256, "has_bias": True}],
          "max_axons": 2304, "max_neurons": 512},
    "G": {"cores": [{"max_axons": 4608, "max_neurons": 2048, "count": 512, "has_bias": True}],
          "max_axons": 4608, "max_neurons": 2048},
}

VEHICLES = {
    "mmixcore": {"model_type": "mlp_mixer_core", "platform": "A", "axis": "mlp_mixer_core",
                 "model_config": {"base_activation": "ReLU", "patch_n_1": 4, "patch_m_1": 4,
                                  "patch_c_1": 32, "fc_w_1": 64, "fc_w_2": 64}},
    "lenet5": {"model_type": "lenet5", "platform": "B", "axis": "lenet5",
               "model_config": {"variant": "lenet5"}},
    "deepcnn": {"model_type": "deep_cnn", "platform": "C", "axis": "deep_cnn",
                "model_config": {"depth": 8, "width": 16}},
    "deepmlp": {"model_type": "deep_mlp", "platform": "B", "axis": "deep_mlp",
                "model_config": {"depth": 8, "width": 64}},
    "simplemlp": {"model_type": "simple_mlp", "platform": "B", "axis": "deep_mlp",
                  "model_config": {"mlp_width_1": 256, "mlp_width_2": 128}},
}

MODES = {
    "lif": {"spiking_mode": "lif", "firing_mode": "Default", "spike_generation_mode": "Uniform",
            "thresholding_mode": "<", "axis": ("lif", "none")},
    "ttfs": {"spiking_mode": "ttfs", "firing_mode": "TTFS", "spike_generation_mode": "TTFS",
             "thresholding_mode": "<=", "axis": ("ttfs", "none")},
    "ttfsq": {"spiking_mode": "ttfs_quantized", "firing_mode": "TTFS", "spike_generation_mode": "TTFS",
              "thresholding_mode": "<=", "axis": ("ttfs_quantized", "none")},
    "casc": {"spiking_mode": "ttfs_cycle_based", "ttfs_cycle_schedule": "cascaded",
             "firing_mode": "TTFS", "spike_generation_mode": "TTFS", "thresholding_mode": "<=",
             "axis": ("ttfs_cycle_based", "cascaded")},
    "sync": {"spiking_mode": "ttfs_cycle_based", "ttfs_cycle_schedule": "synchronized",
             "firing_mode": "TTFS", "spike_generation_mode": "TTFS", "thresholding_mode": "<=",
             "axis": ("ttfs_cycle_based", "synchronized")},
}

# Quant axis reflects RUNTIME truth (SSOT: config_schema/deployment_derivation.py):
# activation quantization is derived from the mode (ON for lif/casc/sync/ttfsq,
# OFF for analytical ttfs), so configs carry only the WQ declaration and never pin
# activation_quantization. fp = the vanilla float assembly (pipeline_mode vanilla).
QUANT = {
    "fp": {"weight_quantization": False},
    "wq": {"weight_quantization": True},
}
AQ_DERIVED_MODES = {"lif", "ttfsq", "casc", "sync"}


def _quant_axis(row):
    """Resolved hypervolume quantization coordinate (runtime truth, not config fiction)."""
    if not QUANT[row["quant"]]["weight_quantization"]:
        return "none"
    return "wq_aq" if row["mode"] in AQ_DERIVED_MODES else "wq"

T0 = [
    dict(n=1, mode="lif", quant="wq", wb=5, s=4, vehicle="mmixcore"),
    dict(n=2, mode="lif", quant="fp", wb=5, s=8, vehicle="lenet5", firing="Novena",
         encoding="offload", pruned=0.5, tags=["novena", "offload", "pruned"]),
    # W2: the 360-core pool packs t0_03 only scheduled (111/360 peak over 4 phases).
    dict(n=3, mode="lif", quant="wq", wb=4, s=16, vehicle="deepcnn", depth=8,
         scheduling=True, tags=["sched"]),
    # W3c respec: was the fictional aq form (wq=False + weight_bits ran as a de-facto
    # float deployment, X4 passed that form); now a real WQ deployment.
    dict(n=4, mode="lif", quant="wq", wb=5, s=32, vehicle="deepmlp", depth=8,
         note="W3c respec 2026-07-06: fictional aq form (weight_quantization=false + "
              "weight_bits ran float) -> real WQ deployment; X4 passed the old form."),
    dict(n=5, mode="lif", quant="wq", wb=5, s=4, vehicle="simplemlp", seed=1),
    dict(n=6, mode="ttfs", quant="wq", wb=5, s=8, vehicle="mmixcore"),
    # W3c respec: same fictional-aq class as t0_04.
    dict(n=7, mode="ttfs", quant="wq", wb=5, s=16, vehicle="lenet5",
         note="W3c respec 2026-07-06: fictional aq form (weight_quantization=false + "
              "weight_bits ran float) -> real WQ deployment; X4 passed the old form."),
    dict(n=8, mode="ttfs", quant="fp", wb=5, s=32, vehicle="deepcnn", depth=8,
         scheduling=True, sim_samples=25, tags=["wall_risk", "sched"],
         note="Sim-sample respec 2026-07-07 (user-directed): the analytic "
              "per-core GEMM nevresim step is the one sample-bound sim wall "
              "(~7 s/sample, 712 s at N=100, X4); accuracy read 1.00."),
    dict(n=9, mode="ttfs", quant="wq", wb=5, s=4, vehicle="deepmlp", depth=4, width=128, pruned=0.5, tags=["pruned"]),
    dict(n=10, mode="ttfs", quant="fp", wb=5, s=16, vehicle="simplemlp",
         coalescing=False, splitting=False, tags=["identity"]),
    dict(n=11, mode="ttfsq", quant="wq", wb=5, s=16, vehicle="mmixcore",
         encoding="offload", tags=["offload"]),
    dict(n=12, mode="ttfsq", quant="wq", wb=8, s=32, vehicle="lenet5", tags=["wall_risk"]),
    dict(n=13, mode="ttfsq", quant="wq", wb=5, s=4, vehicle="deepcnn", depth=4),
    dict(n=14, mode="ttfsq", quant="wq", wb=5, s=8, vehicle="deepmlp", depth=8),
    dict(n=15, mode="ttfsq", quant="wq", wb=5, s=8, vehicle="simplemlp", pruned=0.10,
         tags=["pruned10"],
         note="W3c respec 2026-07-06: pruning 0.5 -> 0.10 (user-directed; 50% is "
              "too strong for this cell)."),
    dict(n=16, mode="casc", quant="wq", wb=5, s=8, vehicle="mmixcore", encoding="offload",
         scheduling=True, has_bias=False, tags=["offload", "sched", "nobias"]),
    dict(n=17, mode="casc", quant="wq", wb=5, s=32, vehicle="lenet5", tags=["wall_risk"]),
    dict(n=18, mode="casc", quant="wq", wb=5, s=4, vehicle="deepcnn", depth=4, pruned=0.5,
         tags=["pruned", "known_collapse_candidate"]),
    # W2: plain d16 is recipe-unreachable (5/5 runs at chance); residual is the
    # trainable deep backbone (USER DECISION 2026-07-06: residual, depth kept).
    dict(n=19, mode="casc", quant="wq", wb=4, s=16, vehicle="deepmlp", depth=16,
         residual=True, tags=["wall_risk", "known_collapse_candidate", "residual"]),
    dict(n=20, mode="casc", quant="wq", wb=5, s=4, vehicle="simplemlp"),
    dict(n=21, mode="sync", quant="wq", wb=5, s=8, vehicle="mmixcore", pruned=0.10,
         tags=["pruned10"],
         note="W3c respec 2026-07-06: pruning 0.5 -> 0.10 (user-directed; 50% is "
              "too strong for this cell)."),
    dict(n=22, mode="sync", quant="wq", wb=5, s=4, vehicle="lenet5", scheduling=True, tags=["sched"]),
    dict(n=23, mode="sync", quant="wq", wb=8, s=16, vehicle="deepcnn", depth=4),
    dict(n=24, mode="sync", quant="wq", wb=5, s=8, vehicle="deepmlp", depth=4, width=128),
    dict(n=25, mode="sync", quant="wq", wb=5, s=32, vehicle="simplemlp"),
]

T1 = [
    dict(n=1, mode="lif", quant="wq", wb=8, s=16, vehicle="squeezenet", regime="pretrained"),
    dict(n=2, mode="ttfs", quant="wq", wb=8, s=32, vehicle="vit", regime="pretrained", tags=["wall_risk"]),
    dict(n=3, mode="ttfsq", quant="wq", wb=8, s=32, vehicle="vit", regime="pretrained",
         pruned=0.05, tags=["wall_risk", "pruned"]),
    dict(n=4, mode="casc", quant="wq", wb=5, s=8, vehicle="deepcnn32", depth=8, regime="from_scratch"),
    dict(n=5, mode="sync", quant="wq", wb=5, s=8, vehicle="deepcnn32", depth=4, regime="from_scratch"),
    dict(n=6, mode="lif", quant="wq", wb=8, s=32, vehicle="deepcnn32", depth=8, regime="from_scratch"),
    dict(n=7, mode="casc", quant="wq", wb=8, s=16, vehicle="squeezenet", regime="pretrained",
         scheduling=True, tags=["sched"]),
    dict(n=8, mode="ttfs", quant="fp", wb=8, s=16, vehicle="mixerc10", regime="from_scratch"),
]

T1_VEHICLES = {
    "squeezenet": {"model_type": "torch_squeezenet11", "platform": "D", "axis": "vit_b",
                   "model_config": {}, "coalescing": False},
    "vit": {"model_type": "torch_vit", "platform": "E", "axis": "vit_b",
            "model_config": {}, "coalescing": False,
            "preprocessing": {"interpolation": "bicubic", "resize_to": 224, "normalize": "imagenet"},
            "batch_size": 512, "tuning_batch_size": 128},
    "deepcnn32": {"model_type": "deep_cnn", "platform": "F", "axis": "deep_cnn",
                  "model_config": {"depth": 8, "width": 32}},
    "mixerc10": {"model_type": "mlp_mixer_core", "platform": "F", "axis": "mlp_mixer_core",
                 "model_config": {"base_activation": "ReLU", "patch_n_1": 4, "patch_m_1": 4,
                                  "patch_c_1": 256, "fc_w_1": 128, "fc_w_2": 256}},
}

T2 = [
    dict(n=1, mode="lif", quant="wq", wb=8, s=32, vehicle="resnet50", dataset="ImageNet",
         regime="pretrained", scheduling=True, lr=0.0001, finetune_epochs=0, budget=0.5, tags=["sched"]),
    dict(n=2, mode="ttfsq", quant="wq", wb=8, s=32, vehicle="vit", dataset="CIFAR100",
         regime="pretrained", scheduling=True, pruned=0.05, tags=["sched", "pruned"]),
    dict(n=3, mode="casc", quant="wq", wb=8, s=32, vehicle="squeezenet", dataset="CIFAR100",
         regime="pretrained", scheduling=True, tags=["sched", "wall_risk"]),
]

T2_VEHICLES = {
    "resnet50": {"model_type": "torch_resnet50", "platform": "G", "axis": "vit_b",
                 "model_config": {}, "coalescing": False},
    "vit": T1_VEHICLES["vit"],
    "squeezenet": T1_VEHICLES["squeezenet"],
}

DATASET_AXIS = {"MNIST": "mnist", "CIFAR10": "cifar10", "CIFAR100": "cifar100", "ImageNet": "imagenet"}


def _name(tier, row, vehicles):
    v = row["vehicle"]
    depth = f"_d{row['depth']}" if "depth" in row else ""
    tags = "".join(f"_{t}" for t in row.get("tags", []) if t in
                   ("offload", "sched", "nobias", "pruned", "pruned10", "novena",
                    "identity", "residual"))
    return f"t{tier}_{row['n']:02d}_{row['mode']}_{v}{depth}_{row['quant']}_s{row['s']}{tags}"


def _platform(row, vehicles):
    v = vehicles[row["vehicle"]]
    plat = json.loads(json.dumps(PLATFORMS[v["platform"]]))
    has_bias = row.get("has_bias", True)
    for core in plat["cores"]:
        core["has_bias"] = has_bias
    plat["has_bias"] = has_bias
    plat["target_tq"] = row["s"]
    plat["simulation_steps"] = row["s"]
    plat["weight_bits"] = row["wb"]
    plat["allow_coalescing"] = row.get("coalescing", v.get("coalescing", True))
    plat["allow_neuron_splitting"] = row.get("splitting", True)
    return plat


def _deployment(tier, row, vehicles, dataset):
    v = vehicles[row["vehicle"]]
    mode = MODES[row["mode"]]
    quant = QUANT[row["quant"]]
    model_config = dict(v["model_config"])
    if "depth" in row:
        model_config["depth"] = row["depth"]
    if "width" in row:
        model_config["width"] = row["width"]
    if "residual" in row:
        model_config["residual"] = row["residual"]

    dp = {
        "lr": row.get("lr", 0.003),
        "tuning_budget_scale": row.get(
            "budget", 0.25 if tier == 0 and row["mode"] in ("lif", "ttfs") else 0.5 if tier == 0 else 1,
        ),
        "degradation_tolerance": 0.15 if tier == 0 else 0.1,
        "model_config_mode": "user",
        "hw_config_mode": "fixed",
        "model_type": v["model_type"],
        "model_config": model_config,
        "batch_size": v.get("batch_size", 128),
        "max_simulation_samples": row.get("sim_samples", 100),
        "sanafe_arch_preset": "loihi",
        "sanafe_sample_count": 1,
        "allow_scheduling": row.get("scheduling", False),
        "spiking_mode": mode["spiking_mode"],
        "firing_mode": row.get("firing", mode["firing_mode"]),
        "spike_generation_mode": mode["spike_generation_mode"],
        "thresholding_mode": mode["thresholding_mode"],
        "encoding_layer_placement": row.get("encoding", "subsume"),
        "weight_quantization": quant["weight_quantization"],
    }
    if "ttfs_cycle_schedule" in mode:
        dp["ttfs_cycle_schedule"] = mode["ttfs_cycle_schedule"]
    if "pruned" in row:
        dp["pruning"] = True
        dp["pruning_fraction"] = row["pruned"]
    if "tuning_batch_size" in v:
        dp["tuning_batch_size"] = v["tuning_batch_size"]
    if "preprocessing" in v:
        dp["preprocessing"] = v["preprocessing"]

    regime = row.get("regime", "from_scratch")
    if regime == "pretrained":
        dp["weight_source"] = "torchvision"
        dp["finetune_epochs"] = row.get("finetune_epochs", 2)
    else:
        dp["training_epochs"] = 2 if tier == 0 else 20
        if tier == 0:
            dp["training_recipe"] = TRAINING_RECIPE
            dp["tuning_recipe"] = TUNING_RECIPE
    return dp


def _cell(tier, row, vehicles, dataset):
    v = vehicles[row["vehicle"]]
    firing, sync = MODES[row["mode"]]["axis"]
    return {
        "firing": firing,
        "sync": sync,
        "quantization": _quant_axis(row),
        "S": str(row["s"]),
        "depth": str(row["depth"]) if "depth" in row else "any",
        "vehicle": v["axis"],
        "dataset": DATASET_AXIS[dataset],
        "regime": row.get("regime", "from_scratch"),
        "pruning": "pruned" if "pruned" in row else "dense",
        "encoding_placement": row.get("encoding", "subsume"),
    }


def _wall_budget(tier, row, vehicles, default_min):
    if tier != 0:
        return default_min
    # Measured locally: LIF's Loihi leg and conv-model cells exceed 5 min.
    if row["mode"] == "lif" or vehicles[row["vehicle"]]["model_type"] == "deep_cnn":
        return 12
    return 6


COVERAGE_NOTES = {
    0: [
        "Quantization axis is RUNTIME truth (SSOT: config_schema/"
        "deployment_derivation.py): activation quantization is derived from the "
        "mode (ON for lif/casc/sync/ttfsq, OFF for analytical ttfs); configs "
        "never pin activation_quantization. Names use wq (bits-quantized) or fp "
        "(float/vanilla) only.",
        "W3c respec 2026-07-06: t0_04/t0_07 were the fictional aq class "
        "(weight_quantization=false + weight_bits ran as de-facto float; X4 "
        "passed those forms) -> respecced to real WQ deployments.",
        "W3c respec 2026-07-06: t0_15/t0_21 pruning 0.5 -> 0.10 (user-directed). "
        "t0_02/t0_09/t0_18 stay at 0.5: they pass and keep the heavy-pruning "
        "stressor coverage.",
    ],
}


def _emit_tier(tier, rows, vehicles, dataset, wall_budget_min):
    out_dir = ROOT / f"tier{tier}"
    out_dir.mkdir(exist_ok=True)
    for stale in out_dir.glob("t*.json"):
        stale.unlink()
    manifest = {"tier": tier, "dataset": dataset,
                "wall_budget_minutes_per_run": wall_budget_min, "runs": []}
    if tier in COVERAGE_NOTES:
        manifest["coverage_notes"] = COVERAGE_NOTES[tier]
    for row in rows:
        ds = row.get("dataset", dataset)
        name = _name(tier, row, vehicles)
        config = {
            "seed": row.get("seed", 0),
            "pipeline_mode": "vanilla" if row["quant"] == "fp" else "phased",
            "experiment_name": name,
            "generated_files_path": "./generated",
            "data_provider_name": f"{ds}_DataProvider",
            "platform_constraints": _platform(row, vehicles),
            "deployment_parameters": _deployment(tier, row, vehicles, ds),
            "target_metric_override": None,
            "start_step": None,
            "stop_step": None,
        }
        (out_dir / f"{name}.json").write_text(json.dumps(config, indent=2) + "\n")
        entry = {
            "name": name,
            "config": f"{name}.json",
            "model_type": vehicles[row["vehicle"]]["model_type"],
            "cell": _cell(tier, row, vehicles, ds),
            "tags": row.get("tags", []),
            "expected_wall_min": _wall_budget(tier, row, vehicles, wall_budget_min),
        }
        if "note" in row:
            entry["note"] = row["note"]
        manifest["runs"].append(entry)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return len(rows)


def main():
    n0 = _emit_tier(0, T0, VEHICLES, "MNIST", 5)
    n1 = _emit_tier(1, T1, T1_VEHICLES, "CIFAR10", 120)
    n2 = _emit_tier(2, T2, T2_VEHICLES, "ImageNet", 360)
    print(f"tier0={n0} tier1={n1} tier2={n2}")


if __name__ == "__main__":
    main()
