"""Registry entries: model architecture, dataset-side workload, and training keys."""

from __future__ import annotations

from mimarsinan.config_schema.registry.relevance import Relevance as R
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
)

_SEARCH_ACTIVE = R.any_of(
    R.when("model_config_mode", in_=("search",)),
    R.when("hw_config_mode", in_=("search",)),
)

ENTRIES = (
    _E("model_config_mode", group="model", owner="deployment_specs/search_mode",
       type=T.ENUM, options=("user", "search"), category=Category.BASIC, exposure="user",
       label="Model Config Mode",
       doc="user: declare model_config by hand; search: the co-search discovers "
           "the architecture (configure it in the Co-search panel)."),
    _E("hw_config_mode", group="hardware", owner="deployment_specs/search_mode",
       type=T.ENUM, options=("fixed", "search"), category=Category.BASIC, exposure="user",
       label="Hardware Config Mode",
       doc="fixed: declare the core grid by hand; search: the co-search discovers "
           "it (configure it in the Co-search panel)."),
    _E("model_type", group="model", owner="model_registry/builders",
       type=T.STR, category=Category.BASIC, exposure="user", label="Model Type", important=True,
       doc="Registered model builder id (see /api/model_types); in search mode, "
           "the builder family whose config space the co-search explores."),
    _E("model_config", group="model", owner="model_registry/builders",
       type=T.JSON, category=Category.BASIC, exposure="user", label="Model Config",
       doc="Per-builder architecture hyperparameters; the field schema comes from "
           "the builder (see /api/model_config_schema/{model_type}).",
       relevant=R.when("model_config_mode", in_=("user",)), provided_by="co_search"),
    _E("model_factory", group="model", owner="model_registry",
       type=T.STR, category=Category.ADVANCED, label="Model Factory",
       doc="Explicit model factory override (research escape; normally absent).",
       empty_means="the registered builder for model_type is used"),
    _E("arch_search", group="co_search", owner="search/architecture_search",
       type=T.JSON, category=Category.ADVANCED, exposure="user", label="Search Strategy",
       doc="Co-search optimizer, budget, and objective declaration (see "
           "/api/wizard/schema); drives model and/or hardware discovery.",
       relevant=_SEARCH_ACTIVE, promote_when=_SEARCH_ACTIVE,
       empty_means="the NAS defaults (NSGA-II, default budget, all objectives)"),
    _E("preprocessing", group="workload", owner="data_handling/preprocessing",
       type=T.JSON, category=Category.ADVANCED, exposure="user", label="Preprocessing",
       doc="Input preprocessing spec: resize_to, normalize (imagenet/cifar/...), interpolation."),
    _E("weight_source", group="model", owner="weight_preloading",
       type=T.STR, category=Category.ADVANCED, exposure="user", label="Weight Source",
       doc="Pretrained weight source id/path; presence selects the fine-tune path "
           "(finetune_epochs/finetune_lr) instead of from-scratch pretraining.",
       empty_means="train from scratch (the pretraining path)"),
    _E("preload_weights", group="model", owner="weight_preloading",
       type=T.BOOL, category=Category.ADVANCED, label="Preload Weights",
       doc="Load cached weights when available instead of pretraining."),
    _E("spike_encoding_seed", group="workload", owner="spike_generation",
       type=T.INT, category=Category.ADVANCED, label="Spike Encoding Seed",
       doc="Seed for stochastic spike-train encoding; null follows the run seed.",
       empty_means="follows the run seed"),
    _E("lr", group="training", owner="training_loop",
       type=T.FLOAT, category=Category.BASIC, exposure="user", label="Learning Rate",
       doc="Base learning rate for pretraining and adaptation recovery.",
       bounds=(0.0, 1.0)),
    _E("lr_range_min", group="training", owner="lr_finder",
       type=T.FLOAT, category=Category.ADVANCED, label="LR Range Min",
       doc="Lower bound of the LR-finder sweep.", bounds=(0.0, None)),
    _E("lr_range_max", group="training", owner="lr_finder",
       type=T.FLOAT, category=Category.ADVANCED, label="LR Range Max",
       doc="Upper bound of the LR-finder sweep.", bounds=(0.0, None)),
    _E("training_epochs", group="training", owner="training_loop",
       type=T.INT, category=Category.BASIC, exposure="user", label="Training Epochs",
       doc="From-scratch pretraining epochs.", unit="epochs", bounds=(0, None)),
    _E("finetune_epochs", group="training", owner="weight_preloading",
       type=T.INT, category=Category.ADVANCED, exposure="user", label="Fine-tune Epochs",
       doc="Epochs of fine-tuning after preloading pretrained weights.",
       unit="epochs", bounds=(0, None), relevant=R.when_set("weight_source")),
    _E("finetune_lr", group="training", owner="weight_preloading",
       type=T.FLOAT, category=Category.ADVANCED, exposure="user", label="Fine-tune LR",
       doc="Learning rate for the fine-tune path (defaults to lr when absent).",
       bounds=(0.0, 1.0), relevant=R.when_set("weight_source"),
       empty_means="falls back to the base learning rate (lr)"),
    _E("batch_size", group="training", owner="DataLoaderFactory",
       type=T.INT, category=Category.BASIC, exposure="user", label="Batch Size",
       doc="Training/eval dataloader batch size.", bounds=(1, None)),
    _E("tuning_batch_size", group="training", owner="DataLoaderFactory",
       type=T.INT, category=Category.ADVANCED, label="Tuning Batch Size",
       doc="Adaptation-tuner dataloader batch size override (large-model cells "
           "shrink it to fit tuning-time memory).", bounds=(1, None),
       empty_means="falls back to batch_size"),
    _E("training_recipe", group="training", owner="training_loop",
       type=T.RECIPE, category=Category.ADVANCED, exposure="user", label="Training Recipe",
       doc="Optimizer/scheduler recipe for pretraining (AdamW + cosine + LLRD default)."),
    _E("tuning_recipe", group="training", owner="AdaptationManager",
       type=T.RECIPE, category=Category.ADVANCED, exposure="user", label="Tuning Recipe",
       doc="Optimizer/scheduler recipe for adaptation tuners (no warmup/LLRD default)."),
    _E("kd_ce_alpha", group="training", owner="training_loop/distillation",
       type=T.FLOAT, category=Category.ADVANCED, label="KD CE Alpha",
       doc="Cross-entropy weight in the knowledge-distillation blend loss.",
       bounds=(0.0, 1.0)),
    _E("kd_temperature", group="training", owner="training_loop/distillation",
       type=T.FLOAT, category=Category.ADVANCED, label="KD Temperature",
       doc="Softmax temperature for distillation targets.", bounds=(0.0, None)),
    _E("input_data_scale", group="workload", owner="workload_profile/spike_encoding",
       type=T.FLOAT, category=Category.ADVANCED, label="Input Data Scale",
       doc="Deployed input-boundary scale (post-transform upper value bound). "
           "Providers register it via DataWorkloadProfile.input_value_range; "
           "explicit value wins; absent = 1.0 (unit-range data).",
       bounds=(0.0, None),
       empty_means="the provider's registered range, else 1.0 (unit-range data)"),
    _E("pretrained_weight_source", group="model", owner="workload_profile/weight_preloading",
       type=T.STR, category=Category.ADVANCED, label="Pretrained Weight Source",
       doc="Weight source the preload_weights regime resolves to. Builders "
           "register it via ModelWorkloadProfile; explicit value wins.",
       empty_means="the builder's registered source (if any)"),
    _E("clamp_cuda_assert_prone", group="model", owner="workload_profile/deployment_specs",
       type=T.BOOL, category=Category.ADVANCED, label="Clamp CUDA-assert Prone",
       doc="Architecture is known to trip CUDA asserts under clamp adaptation "
           "(warns to enable cuda_debug). Builders register it via "
           "ModelWorkloadProfile; explicit value wins."),
)
