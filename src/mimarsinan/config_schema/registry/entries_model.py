"""Registry entries: model architecture, dataset-side workload, and training keys."""

from __future__ import annotations

from mimarsinan.common.pretrained import (
    derived_weight_set_id,
    legal_preload_values,
    legal_weight_set_ids,
    select_weight_set,
    selected_source,
)
from mimarsinan.config_schema.registry.relevance import Relevance as R
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
    frozen_default as _frozen,
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
    chosen = select_weight_set(cfg)
    if cfg.get("preload_weights"):
        if resolved and (chosen is None or resolved != chosen["source"]):
            return f"explicit declaration ({resolved!r}) — wins over the weight set"
        if chosen is not None:
            return f"the chosen weight set {chosen['id']!r} ({chosen['source']!r})"
        return "unresolved — no applicable pretrained weight set is registered"
    if resolved:
        return f"explicit declaration ({resolved!r})"
    return "none — from-scratch pretraining"


def _why_weight_set(cfg: dict) -> str:
    """WHY the chosen weight set: an explicit id wins, else the builder default."""
    chosen = select_weight_set(cfg)
    if chosen is None:
        return "none — the preload regime is off or nothing is applicable"
    declared = cfg.get("pretrained_weight_set")
    if declared:
        return f"declared weight set {chosen['id']!r}"
    return f"the builder's default weight set {chosen['id']!r}"

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
       doc="THE loader-facing pretrained-weight source (one key). Derived from "
           "the CHOSEN registered weight set's own source under preload_weights; "
           "a document may still pin an id/path/URL explicitly (config-data "
           "escape) — explicit wins. Presence selects the fine-tune path instead "
           "of from-scratch pretraining.",
       derived_from=("preload_weights", "pretrained_weight_set", "model_type"),
       why=_why_weight_source, declarable=True, provenance="builder profile",
       derived_default=lambda cfg: selected_source(cfg),
       empty_means="preload on: the chosen weight set's source; off: train from scratch"),
    _E("preload_weights", group="training", owner="weight_preloading",
       type=T.BOOL, category=Category.BASIC, exposure="user", label="Preload Weights",
       doc="Use pretrained weights: the source resolves from the CHOSEN weight "
           "set the model builder registers and the fine-tune path replaces "
           "from-scratch pretraining. Disabled when the builder registers no "
           "applicable pretrained weight set (the legal set admits only off).",
       legal_values=lambda cfg: legal_preload_values(
           cfg, source_declared=bool(cfg.get("weight_source"))
       )),
    _E("pretrained_weight_set", group="training", owner="weight_preloading",
       type=T.STR, category=Category.DERIVED, derivation="derived",
       exposure="user", label="Pretrained Weight Set",
       doc="WHICH of the model builder's registered pretrained weight sets to "
           "load (task/dataset/geometry/classes/source). The legal set is the "
           "ids the current builder + model_config register; a single "
           "registration LOCKS the choice, several let the user select. "
           "Builder-declared — the framework never enumerates workloads.",
       derived_from=("preload_weights", "model_type", "model_config"),
       why=_why_weight_set, declarable=True, provenance="builder profile",
       derived_default=lambda cfg: derived_weight_set_id(cfg),
       legal_values=lambda cfg: legal_weight_set_ids(cfg),
       relevant=_PRETRAINED_REGIME, empty_means="the builder's default weight set"),
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
       promote_when=_PRETRAINED_REGIME,
       empty_means="0 — load the weights without fine-tuning"),
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
       derived_default=_frozen(1.0),
       empty_means="the provider's registered range, else 1.0 (unit-range data)"),
    _E("pretrained_weight_sets", group="training",
       owner="workload_profile/weight_preloading",
       type=T.JSON, category=Category.DERIVED, derivation="derived",
       exposure="derived", label="Pretrained Weight Sets (registration)",
       doc="The SET of pretrained weight sets the builder's ModelWorkloadProfile "
           "registers — the injection contract the wizard reads, never authorable. "
           "Each record: id/label/task/dataset/input_shape/num_classes/source plus "
           "expected accuracy, licence, preprocessing and applicability facts.",
       derived_from=("model_type",),
       why=lambda cfg: "the model builder's ModelWorkloadProfile registration",
       declarable=False, hidden=True, provenance="builder profile",
       derived_default=lambda cfg: list(cfg.get("pretrained_weight_sets") or [])),
    _E("clamp_cuda_assert_prone", group="model", owner="workload_profile/deployment_specs",
       type=T.BOOL, category=Category.ADVANCED, label="Clamp CUDA-assert Prone",
       doc="Architecture is known to trip CUDA asserts under clamp adaptation "
           "(warns to enable cuda_debug). Builders register it via "
           "ModelWorkloadProfile; explicit value wins.",
       provenance="builder profile", derived_default=_frozen(False)),
)
