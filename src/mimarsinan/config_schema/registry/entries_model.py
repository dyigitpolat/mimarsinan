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

# The pretrained-weight regime is active while the regime flag is on OR a
# document pins an explicit source (the config-data escape).
_PRETRAINED_REGIME = R.any_of(
    R.when_true("preload_weights"),
    R.when_set("weight_source"),
)


def _why_weight_source(cfg: dict) -> str:
    """WHY against the RESOLVED config: the resolution already folded the
    builder registration in, so credit whichever input actually won."""
    resolved = cfg.get("weight_source")
    registered = cfg.get("pretrained_weight_source")
    if cfg.get("preload_weights"):
        if registered is not None and (resolved is None or resolved == registered):
            return f"the model builder's registration ({registered!r})"
        if registered is None and resolved is None:
            return "unresolved — the model builder registers no pretrained source"
        return f"explicit declaration ({resolved!r}) — wins over the registration"
    if resolved:
        return f"explicit declaration ({resolved!r})"
    return "none — from-scratch pretraining"

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
       type=T.JSON, category=Category.BASIC, exposure="user", label="Preprocessing",
       doc="Input preprocessing spec: resize_to, normalize (imagenet/cifar/...), interpolation."),
    _E("weight_source", group="training", owner="weight_preloading",
       type=T.STR, category=Category.DERIVED, derivation="derived",
       exposure="user", label="Weight Source",
       doc="THE pretrained-weight-source concept (one key, round-5 dedupe). "
           "Builder-registration-provided: preload_weights=true resolves it to "
           "the builder's ModelWorkloadProfile registration (no registration "
           "fails loud). A document may still pin an id/path/URL explicitly "
           "(config-data escape) — explicit wins; presence selects the "
           "fine-tune path instead of from-scratch pretraining.",
       derived_from=("preload_weights", "model_type"),
       why=_why_weight_source, declarable=True, provenance="builder profile",
       empty_means="preload on: the builder's registered source; off: train from scratch"),
    _E("preload_weights", group="training", owner="weight_preloading",
       type=T.BOOL, category=Category.BASIC, exposure="user", label="Preload Weights",
       doc="Use pretrained weights: the weight source resolves from the model "
           "builder's registration and the fine-tune path replaces from-scratch "
           "pretraining."),
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
       unit="epochs", bounds=(0, None), relevant=_PRETRAINED_REGIME,
       promote_when=_PRETRAINED_REGIME),
    _E("finetune_lr", group="training", owner="weight_preloading",
       type=T.FLOAT, category=Category.ADVANCED, exposure="user", label="Fine-tune LR",
       doc="Learning rate for the fine-tune path (defaults to lr when absent).",
       bounds=(0.0, 1.0), relevant=_PRETRAINED_REGIME,
       promote_when=_PRETRAINED_REGIME,
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
       type=T.RECIPE, category=Category.BASIC, exposure="user", label="Training Recipe",
       doc="Optimizer/scheduler recipe for pretraining (AdamW + cosine + LLRD default)."),
    _E("mirror_training_recipe", group="tuning", owner="deployment_derivation/recipe_mirror",
       type=T.BOOL, category=Category.BASIC, exposure="user",
       label="Mirror Training Recipe",
       effect="The tuning recipe reflects the training recipe as-is",
       doc="Reflect the training recipe into the tuning recipe (default off). "
           "While on, the training concern owns the tuning recipe; an explicit "
           "tuning_recipe declaration conflicts and is rejected."),
    _E("tuning_recipe", group="tuning", owner="AdaptationManager",
       type=T.RECIPE, category=Category.BASIC, exposure="user", label="Tuning Recipe",
       doc="Optimizer/scheduler recipe for adaptation tuners (no warmup/LLRD default).",
       relevant=R.when("mirror_training_recipe", in_=(False,)),
       provided_by="training"),
    _E("kd_ce_alpha", group="training", owner="training_loop/distillation",
       type=T.FLOAT, category=Category.ADVANCED, label="KD CE Alpha",
       doc="Cross-entropy weight in the knowledge-distillation blend loss "
           "(the mode recipe may override the schema default).",
       bounds=(0.0, 1.0), provenance="ConversionPolicy recipe"),
    _E("kd_temperature", group="training", owner="training_loop/distillation",
       type=T.FLOAT, category=Category.ADVANCED, label="KD Temperature",
       doc="Softmax temperature for distillation targets (the mode recipe "
           "may override the schema default).", bounds=(0.0, None),
       provenance="ConversionPolicy recipe"),
    _E("input_data_scale", group="workload", owner="workload_profile/spike_encoding",
       type=T.FLOAT, category=Category.ADVANCED, label="Input Data Scale",
       doc="Deployed input-boundary scale (post-transform upper value bound). "
           "Providers register it via DataWorkloadProfile.input_value_range; "
           "explicit value wins; absent = 1.0 (unit-range data).",
       bounds=(0.0, None), provenance="provider registration",
       empty_means="the provider's registered range, else 1.0 (unit-range data)"),
    _E("pretrained_weight_source", group="training",
       owner="workload_profile/weight_preloading",
       type=T.STR, category=Category.DERIVED, derivation="derived",
       exposure="derived", label="Pretrained Weight Source (registration)",
       doc="The builder ModelWorkloadProfile registration the preload_weights "
           "regime resolves to — an injection contract, never authorable "
           "(round-5 dedupe: weight_source is the one declarable concept).",
       derived_from=("model_type",),
       why=lambda cfg: "the model builder's ModelWorkloadProfile registration",
       declarable=False, hidden=True, provenance="builder profile"),
    _E("clamp_cuda_assert_prone", group="model", owner="workload_profile/deployment_specs",
       type=T.BOOL, category=Category.ADVANCED, label="Clamp CUDA-assert Prone",
       doc="Architecture is known to trip CUDA asserts under clamp adaptation "
           "(warns to enable cuda_debug). Builders register it via "
           "ModelWorkloadProfile; explicit value wins.",
       provenance="builder profile"),
)
