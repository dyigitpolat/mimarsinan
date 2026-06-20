"""Generate 9 MMIXCORE + MNIST templates with full config-matrix coverage.

Each is wizard-constructible: validated by config_schema.validate_deployment_config
and resolved by build_flat_pipeline_config (the wizard's build path). Run from repo root.
"""
import copy
import json
import os

MMIXCORE_CORES = [
    {"max_axons": 256, "max_neurons": 512, "count": 60, "has_bias": True},
    {"max_axons": 512, "max_neurons": 256, "count": 60, "has_bias": True},
]
MODEL_CONFIG = {
    "base_activation": "ReLU",
    "patch_n_1": 4, "patch_m_1": 4, "patch_c_1": 128, "fc_w_1": 64, "fc_w_2": 128,
}
TRAIN_RECIPE = {
    "optimizer": "adamw", "scheduler": "cosine", "weight_decay": 0.0001,
    "warmup_ratio": 0.05, "grad_clip_norm": 1, "layer_wise_lr_decay": 1,
    "label_smoothing": 0, "betas": [0.9, 0.999],
}
TUNE_RECIPE = {**TRAIN_RECIPE, "warmup_ratio": 0}


def platform(has_bias=True, allow_scheduling_unused=None):
    cores = copy.deepcopy(MMIXCORE_CORES)
    if not has_bias:
        for c in cores:
            c["has_bias"] = False
    return {
        "cores": cores,
        "max_axons": 512, "max_neurons": 512,
        "target_tq": 4, "simulation_steps": 4, "weight_bits": 5,
        "has_bias": has_bias,
        "allow_coalescing": True, "allow_neuron_splitting": True,
    }


def base_dp():
    return {
        "lr": 0.003, "training_epochs": 10, "tuning_budget_scale": 1,
        "degradation_tolerance": 0.15,
        "model_config_mode": "user", "hw_config_mode": "fixed",
        "model_type": "mlp_mixer_core", "model_config": copy.deepcopy(MODEL_CONFIG),
        "batch_size": 128,
        "training_recipe": copy.deepcopy(TRAIN_RECIPE),
        "tuning_recipe": copy.deepcopy(TUNE_RECIPE),
        "max_simulation_samples": 500,
        "allow_scheduling": False,
        "enable_nevresim_simulation": True,
        "enable_loihi_simulation": False,
        "enable_sanafe_simulation": True,
        "sanafe_arch_preset": "loihi",
        "sanafe_sample_count": 1,
        "sanafe_custom_arch_path": None,
    }


def lif_dp(firing="Default", cycle_accurate=True):
    dp = base_dp()
    dp.update({
        "spiking_mode": "lif", "firing_mode": firing,
        "spike_generation_mode": "Uniform", "thresholding_mode": "<",
        "encoding_layer_placement": "subsume",
        "cycle_accurate_lif_forward": cycle_accurate,
        "activation_quantization": False, "weight_quantization": True,
    })
    return dp


def ttfs_dp(spiking, schedule=None, encoding="subsume", weight_quant=True):
    dp = base_dp()
    dp.update({
        "spiking_mode": spiking, "firing_mode": "TTFS",
        "spike_generation_mode": "TTFS", "thresholding_mode": "<=",
        "encoding_layer_placement": encoding,
        "activation_quantization": (spiking == "ttfs_quantized"),
        "weight_quantization": weight_quant,
    })
    if schedule is not None:
        dp["ttfs_cycle_schedule"] = schedule
    return dp


def doc(name, dp, has_bias=True, pipeline_mode="phased"):
    return {
        "seed": 0,
        "pipeline_mode": pipeline_mode,
        "experiment_name": name,
        "generated_files_path": "./generated",
        "data_provider_name": "MNIST_DataProvider",
        "platform_constraints": platform(has_bias=has_bias),
        "deployment_parameters": dp,
        "target_metric_override": None,
        "start_step": None, "stop_step": None,
    }


# ── the 9 templates (full-matrix coverage) ──────────────────────────────────
TEMPLATES = {}

# T1 — LIF rate baseline: subsume, <, nevresim+sanafe, on_chip bias
TEMPLATES["mnist_mmixcore_matrix_1_lif_rate"] = doc(
    "mnist_mmixcore_matrix_1_lif_rate", lif_dp(firing="Default"))

# T2 — LIF Novena firing + OFFLOAD + Loihi(Lava, LIF-only) backend (+ nevresim + sanafe)
_t2 = lif_dp(firing="Novena", cycle_accurate=False)
_t2["encoding_layer_placement"] = "offload"
_t2["enable_loihi_simulation"] = True
TEMPLATES["mnist_mmixcore_matrix_2_lif_novena_offload_loihi"] = doc(
    "mnist_mmixcore_matrix_2_lif_novena_offload_loihi", _t2)

# T3 — LIF + PRUNING on + SYNC-POINTS (allow_scheduling)
_t3 = lif_dp(firing="Default")
_t3.update({"pruning": True, "pruning_fraction": 0.5, "allow_scheduling": True})
TEMPLATES["mnist_mmixcore_matrix_3_lif_pruned_scheduled"] = doc(
    "mnist_mmixcore_matrix_3_lif_pruned_scheduled", _t3)

# T4 — TTFS analytical: subsume, <=, weight-quant on, act-quant off
TEMPLATES["mnist_mmixcore_matrix_4_ttfs_analytical"] = doc(
    "mnist_mmixcore_matrix_4_ttfs_analytical", ttfs_dp("ttfs"))

# T5 — TTFS quantized + OFFLOAD (activation_quantization forced on)
TEMPLATES["mnist_mmixcore_matrix_5_ttfs_quantized_offload"] = doc(
    "mnist_mmixcore_matrix_5_ttfs_quantized_offload",
    ttfs_dp("ttfs_quantized", encoding="offload"))

# T6 — TTFS cycle CASCADED (genuine single-spike): nevresim+sanafe
TEMPLATES["mnist_mmixcore_matrix_6_ttfs_cycle_cascaded"] = doc(
    "mnist_mmixcore_matrix_6_ttfs_cycle_cascaded",
    ttfs_dp("ttfs_cycle_based", schedule="cascaded"))

# T7 — TTFS cycle SYNCHRONIZED (nevresim auto-disabled → sanafe+HCM)
_t7 = ttfs_dp("ttfs_cycle_based", schedule="synchronized")
_t7["enable_nevresim_simulation"] = False
TEMPLATES["mnist_mmixcore_matrix_7_ttfs_cycle_synchronized"] = doc(
    "mnist_mmixcore_matrix_7_ttfs_cycle_synchronized", _t7)

# T8 — TTFS cycle cascaded + OFFLOAD + SYNC-POINTS + has_bias=False (param_encoded bias)
_t8 = ttfs_dp("ttfs_cycle_based", schedule="cascaded", encoding="offload")
_t8["allow_scheduling"] = True
TEMPLATES["mnist_mmixcore_matrix_8_ttfs_cycle_offload_scheduled_nobias"] = doc(
    "mnist_mmixcore_matrix_8_ttfs_cycle_offload_scheduled_nobias", _t8,
    has_bias=False)

# T9 — TTFS analytical, weight_quantization OFF → VANILLA pipeline
TEMPLATES["mnist_mmixcore_matrix_9_ttfs_vanilla_noWQ"] = doc(
    "mnist_mmixcore_matrix_9_ttfs_vanilla_noWQ",
    ttfs_dp("ttfs", weight_quant=False), pipeline_mode="vanilla")


def main():
    out_dir = "templates"
    for name, d in TEMPLATES.items():
        path = os.path.join(out_dir, name + ".json")
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        print("wrote", path)


if __name__ == "__main__":
    main()
