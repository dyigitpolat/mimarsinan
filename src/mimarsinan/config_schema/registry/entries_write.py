"""Registry entries: the write-law policy keys, the core-type field schema the write fields extend, the
declarations the builder injects into entries declared beside other concerns (consumer readings included), and
the platform bias rule both doors apply."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple

from mimarsinan.config_schema.registry.decision_classes import COST_ONLY, FUNCTION_BEARING
from mimarsinan.config_schema.registry.types import Category, ConfigKeySchema as _E, FieldType as T
from mimarsinan.config_schema.registry.value_schema import ValueSchema, enum_legal_set, schema_of, value_errors
from mimarsinan.data_handling.preprocessing import PreprocessingSpec, interpolation_mode_names
from mimarsinan.config_schema.registry.core_type_doors import platform_sign_errors as platform_sign_errors
from mimarsinan.mapping.platform.platform_constraints import PER_SYNAPSE, WEIGHT_SIGN_GRANULARITIES
from mimarsinan.mapping.support.period_census import (
    OPERAND_WRITE_SOURCES,
    WRITE_GRANULARITIES,
    WRITE_OVERLAP_MODES,
)
from mimarsinan.tuning.orchestration.temporal_allocation import (
    S_ALLOCATION_EXPLICIT,
    S_ALLOCATION_MODES,
    S_ALLOCATION_SUPPORTED_MODES,
    S_ALLOCATION_UNIFORM,
    TemporalAllocationResolver,
    resolve_s_allocation_mode,
)

_PC = "platform_constraints"
_OWNER = "deployment_record.cost"

#: The fields of one ``cores`` entry (geometry, bias, weight sign, write); open: the parser flags the rest.
CORE_TYPE_SCHEMA = ValueSchema(
    kind=T.JSON,
    fields={
        "max_axons": ValueSchema(kind=T.INT, low=1, decision_class=COST_ONLY,
                                 doc="axons per core: tiling and packing only",
                                 absent_means="undeclared; mapping needs it"),
        "max_neurons": ValueSchema(kind=T.INT, low=1, decision_class=COST_ONLY,
                                   doc="neurons per core: tiling and packing only",
                                   absent_means="undeclared; mapping needs it"),
        "count": ValueSchema(kind=T.INT, low=1, decision_class=COST_ONLY,
                             doc="cores of this type: packing and passes only",
                             absent_means="one core"),
        "has_bias": ValueSchema(kind=T.BOOL, decision_class=FUNCTION_BEARING,
                                doc="the bias mode changes the weight-quantization grid",
                                absent_means="the platform bias mode: the flat has_bias, else the nested "
                                             "platform_constraints.has_bias, else true (a bias lane)"),
        "weight_sign_granularity": ValueSchema(
            kind=T.ENUM, options=WEIGHT_SIGN_GRANULARITIES, decision_class=COST_ONLY,
            doc="one sign per synapse, or one per axon row: a mixed-sign input then takes two rows (W+ then W-, "
                "exact on the integer grid under every integer semantics; analytical TTFS, outside the integer "
                "scope, equal up to float rounding), so rows, packing and cost only",
            absent_means=f"{PER_SYNAPSE} (every synapse carries its own sign)"),
        "weight_bits": ValueSchema(kind=T.INT, low=2, high=32, decision_class=FUNCTION_BEARING,
                                   doc="the cells' weight width: the grid of every layer the layer-to-type rule "
                                       "(layer_core_types) puts on this type, and the only type its cores pack onto",
                                   absent_means="the platform weight_bits"),
        "write_granularity": ValueSchema(kind=T.ENUM, options=WRITE_GRANULARITIES,
                                         decision_class=COST_ONLY,
                                         doc="the unit one write commits: write time only",
                                         absent_means="undeclared; a term needing it refuses by name"),
        "write_units_in_parallel": ValueSchema(kind=T.INT, low=1, decision_class=COST_ONLY,
                                               doc="units one write commits at once: write time only",
                                               absent_means="1"),
        "cells_per_weight": ValueSchema(kind=T.INT, low=1, decision_class=COST_ONLY, doc=(
            "[U-C4] array cells one stored weight occupies (a differential pair 2, a weight bit-sliced over narrower "
            "cells its slices): the cells a weight write writes and wears, and the cells area scales with; "
            "max_axons x max_neurons stay the weights the mapping packs, so cost only"),
            absent_means="1: one cell stores one weight"),
        "weight_cell_rows": ValueSchema(kind=T.INT, low=1, decision_class=COST_ONLY, doc=(
            "[U-C4] of a weight's cells_per_weight cells, those on adjacent rows of its column (a differential pair 2); "
            "the rest on adjacent columns of its row: cost only"), absent_means="required where cells_per_weight exceeds 1"),
        "write_physics": ValueSchema(kind=T.STR, min_length=1, decision_class=COST_ONLY,
                                     doc="the write technology of this core type's cells (N2b): a shipped physics "
                                         "profile whose write rows price their write energy, write time and "
                                         "endurance, per core type; a name no profile ships is refused by name at "
                                         "resolution",
                                     absent_means="the chip's one physics (platform_physics_profile and its "
                                                  "overrides)"),
        "compute_physics": ValueSchema(kind=T.STR, min_length=1, decision_class=COST_ONLY,
                                       doc="the compute technology of this core type's cells (E-6, N2 F5): a shipped "
                                           "physics profile whose compute row (e_mac) prices this core type's MAC "
                                           "events, per core type; a name no profile ships is refused by name at "
                                           "resolution",
                                       absent_means="the chip's one physics (platform_physics_profile and its "
                                                    "overrides)"),
        "operand": ValueSchema(
            kind=T.JSON, decision_class=FUNCTION_BEARING,
            doc="an operand core type (N2): products of two activations leave the host for its cores, each "
                "bank's payload written per inference on its grid",
            absent_means="no operand core type: every product of two activations stays a host op, byte-identical",
            fields={
                "payload_bits": ValueSchema(kind=T.INT, low=2, decision_class=FUNCTION_BEARING,
                                            doc="the payload grid width: rounds every written operand",
                                            absent_means="required when operand is declared"),
                "write_source": ValueSchema(kind=T.ENUM, options=OPERAND_WRITE_SOURCES, decision_class=COST_ONLY,
                                            doc="who writes the payloads (a host op, or the producer's on-chip "
                                                "write-back): the same codes, priced apart, where every payload "
                                                "write copies its producer's codes (readout_ranges "
                                                "per_consumer_group, each stationary operand core-fed at "
                                                "payload_bits); on_chip on a program whose banks re-snap is "
                                                "refused by name at Operand Lowering",
                                            absent_means="host"),
            },
            required=("payload_bits",),
        ),
    },
    closed=False,
)


def s_allocation_as_read(config: Mapping[str, Any]) -> str:
    """The allocation mode as its consumer reads it: ``resolve_s_allocation_mode`` (which lowercases), and
    an explicit per-depth map whose consumer allocation is uniform (``TemporalAllocation.is_uniform``: every
    entry equals simulation_steps) is the uniform declaration it states. No door knows the cascade depth, so
    the map is read at its own length; a map the consumer cannot read leaves the explicit mode. The consumer
    casts each entry with int(); the map's own value schema (``INJECTED_DECLARATIONS``) refuses at both doors
    every map that is no non-empty list of JSON ints >= 1 (the registry's INT rule, as for simulation_steps), and
    so every map the cast would change: no admitted document is read through the cast."""
    mode = resolve_s_allocation_mode(config)
    if mode != S_ALLOCATION_EXPLICIT:
        return mode
    try:
        resolver = TemporalAllocationResolver.from_config(config)
        if resolver.explicit is None:
            return mode
        allocation = resolver.resolve(depth=len(resolver.explicit))
    except (ArithmeticError, TypeError, ValueError):
        return mode
    return S_ALLOCATION_UNIFORM if allocation.is_uniform else mode


#: The one field of ``preprocessing`` whose consumers read a non-canonical spelling differently: the FFCV writer
#: lowercases ``Bilinear`` to bilinear, the torchvision transform falls back to bicubic
#: (``data_handling/preprocessing.py``), so its legal set is the spellings both read alike, judged as written. The
#: other fields stay the parser's (``PREPROCESSING_FIELDS``).
PREPROCESSING_SCHEMA = ValueSchema(
    kind=T.JSON,
    fields={"interpolation": ValueSchema(
        kind=T.ENUM, options=interpolation_mode_names(), decision_class=FUNCTION_BEARING,
        doc="the resize filter of the inputs the network trains and computes on",
        absent_means=PreprocessingSpec().interpolation)},
    closed=False,
)


def _recipe_field(kind: str, *, listed: bool = False, low: Optional[float] = None, high: Optional[float] = None,
                  high_exclusive: bool = False, length: Optional[int] = None, nullable: bool = False) -> ValueSchema:
    """One recipe field as ``TrainingRecipe`` reads it: ``build_recipe`` passes a null field to the dataclass as
    None, which only an Optional field (``nullable``) reads as its own value; every other None fails at training,
    so it is judged as a value. ``low``/``high`` (of each item, ``listed``) and ``length`` are the bounds the
    consumer applies (torch's optimizers, ``_betas_pair``, the loss)."""
    scalar = ValueSchema(kind=kind, low=low, high=high, high_exclusive=high_exclusive)
    return ValueSchema(kind=T.JSON if listed else kind, item=scalar if listed else None,
                       low=None if listed else low, high=None if listed else high,
                       high_exclusive=False if listed else high_exclusive,
                       min_length=length or 0, max_length=length, null_unset=nullable,
                       decision_class=FUNCTION_BEARING, doc="a training-recipe field the trainer reads",
                       absent_means="the TrainingRecipe default")


#: The fields a recipe's consumer reads (``model_training.training_recipe.TrainingRecipe``, which ``build_recipe``
#: takes them into, silently dropping every other one), with their types and the bounds the consumer applies (torch's
#: Adam/AdamW betas in [0, 1), exactly two (``_betas_pair``); a weight decay and an SGD momentum >= 0; a gradient-clip
#: norm >= 0 (``clip_grad_norm_`` reverses the gradients below 0); a label smoothing in [0, 1] (the loss); a warmup
#: ratio in [0, 1] of the steps); closed, so both doors refuse a field it
#: does not read and a recipe that is no object by name, and a null field but the Optional ``label_smoothing``. A
#: test pins the fields to the consumer's.
RECIPE_SCHEMA = ValueSchema(kind=T.RECIPE, fields={
    "optimizer": _recipe_field(T.STR), "weight_decay": _recipe_field(T.FLOAT, low=0.0),
    "betas": _recipe_field(T.FLOAT, listed=True, low=0.0, high=1.0, high_exclusive=True, length=2),
    "momentum": _recipe_field(T.FLOAT, low=0.0),
    "scheduler": _recipe_field(T.STR), "warmup_ratio": _recipe_field(T.FLOAT, low=0.0, high=1.0),
    "grad_clip_norm": _recipe_field(T.FLOAT, low=0.0), "layer_wise_lr_decay": _recipe_field(T.FLOAT),
    "no_decay_keywords": _recipe_field(T.STR, listed=True),
    "label_smoothing": _recipe_field(T.FLOAT, low=0.0, high=1.0, nullable=True),
})

#: Declared attributes of keys whose entries live beside other concerns, injected by the builder: the core-type
#: schema of ``cores``, the absent meaning of the platform bias mode (``platform_bias_mode``), the
#: interpolation legal set of ``preprocessing``, the recipes' fields, and the legal sets taken from their consumers: a value the
#: consumer refuses is no option, a value it names but no run implements is reserved, and both doors refuse
#: either by name (``enum_legal_set``; the per-depth S map takes the list its consumer parses). A consumer that
#: normalizes what it reads gives the key its reading (``consumer_reading``), by reference. An entry never
#: declares a different value of an attribute injected here (a value nothing would read).
INJECTED_DECLARATIONS: Mapping[str, Mapping[str, Any]] = {
    # sanafe/arch_synth/floorplan.resolve_floorplan refuses a platform of no cores (a capacity below 1), and
    # every core type counts at least one core: a non-empty list.
    "cores": {"value_schema": ValueSchema(kind=T.CORES, item=CORE_TYPE_SCHEMA, min_length=1)},
    "has_bias": {"empty_means": "true — every core type declaring no has_bias gets a bias lane; a flat "
                                "config's nested platform_constraints.has_bias, when present, is taken "
                                "instead"},
    "preprocessing": {"value_schema": PREPROCESSING_SCHEMA},
    "training_recipe": {"value_schema": RECIPE_SCHEMA},
    "tuning_recipe": {"value_schema": RECIPE_SCHEMA},
    # tuning/orchestration/temporal_allocation.py: three modes are named, read lowercased; only uniform
    # is threaded.
    "s_allocation": {**enum_legal_set(S_ALLOCATION_SUPPORTED_MODES, reserved=tuple(
        mode for mode in S_ALLOCATION_MODES if mode not in S_ALLOCATION_SUPPORTED_MODES)),
        "consumer_reading": s_allocation_as_read},
    # temporal_allocation._parse_explicit: a non-empty per-depth list of positive ints, judged as written
    # (the consumer's int() cast reads 32.9 and "32" as 32; 32.0, equal in value, is refused by the INT rule,
    # as simulation_steps: 32.0 is).
    "s_allocation_explicit": {"value_schema": ValueSchema(
        kind=T.INT_LIST, item=ValueSchema(kind=T.INT, low=1), min_length=1)},
}


#: Where a flat config declares the platform bias mode: the flat key, then the nested object's field.
PLATFORM_BIAS_PATHS = ("has_bias", "platform_constraints.has_bias")


def platform_bias_declarations(config: Mapping[str, Any]) -> Tuple[Any, Any]:
    """The platform bias mode as declared: the flat ``has_bias`` and the nested
    ``platform_constraints.has_bias`` a flat config may carry (None where absent)."""
    nested = config.get("platform_constraints")
    return config.get("has_bias"), (nested.get("has_bias") if isinstance(nested, Mapping) else None)


def platform_bias_mode(config: Mapping[str, Any]) -> Any:
    """The bias mode of every core type that declares no ``has_bias`` of its own: the flat ``has_bias``
    first, then the nested ``platform_constraints.has_bias``, else true (a bias lane)."""
    flat, nested = platform_bias_declarations(config)
    return flat if flat is not None else nested if nested is not None else True


def conflicting_platform_bias_error(config: Mapping[str, Any]) -> Optional[str]:
    """Why the platform bias mode is ambiguous, or None: a flat and a nested boolean that disagree. A
    non-boolean is the BOOL schema's to refuse (``platform_bias_errors`` judges it first)."""
    flat, nested = platform_bias_declarations(config)
    if not (isinstance(flat, bool) and isinstance(nested, bool)) or flat == nested:
        return None
    return (f"has_bias={flat!r} and platform_constraints.has_bias={nested!r} declare the platform bias "
            f"mode twice and disagree: declare it once")


def platform_bias_errors(config: Mapping[str, Any], bias_entry: _E) -> List[str]:
    """Why the platform bias declarations of ``config`` refuse, else [] (the one rule both doors call: the run
    door over its flat config, the document door over the layout a run gives its two sections): each
    declaration, flat and nested, judged by ``bias_entry``'s value schema (the registry's BOOL) at its own
    path, then a flat and a nested boolean that disagree (``conflicting_platform_bias_error``)."""
    schema = schema_of(bias_entry)
    errors = [message for path, value in zip(PLATFORM_BIAS_PATHS, platform_bias_declarations(config))
              if value is not None and schema is not None for message in value_errors(schema, value, path)]
    conflict = conflicting_platform_bias_error(config)
    return errors if conflict is None else [*errors, conflict]


WRITE_OVERLAP_SCHEMA = ValueSchema(
    kind=T.JSON,
    fields={
        "mode": ValueSchema(kind=T.ENUM, options=WRITE_OVERLAP_MODES, decision_class=COST_ONLY,
                            doc="writes exposed or hidden behind compute: write time only"),
        "shadow_capacity_fraction": ValueSchema(
            kind=T.FLOAT, low=0.0, high=1.0, high_exclusive=True, decision_class=COST_ONLY,
            doc="array share lost to shadow buffering: write time only", absent_means="0"),
    },
    required=("mode",),
)

ENTRIES = (
    _E("inference_batch_K", section=_PC, group="mapping_strategy", owner=_OWNER, type=T.INT,
       category=Category.ADVANCED, exposure="user", label="Inference Batch K",
       effect="Inferences served per weight-programming period (write terms at batch K)",
       doc="Inferences each programmed period serves: backing-store writes are amortized over "
           "K, producer writes are charged per inference. Absent: no batch is declared and no "
           "write-at-K term is emitted.",
       bounds=(1, None), passthrough=True,
       empty_means="no batch declared — no write-at-K term is emitted"),
    _E("write_overlap", section=_PC, group="hardware", owner=_OWNER, type=T.JSON,
       category=Category.ADVANCED, exposure="user", label="Write Overlap",
       effect="Whether array writes are exposed after compute or hidden behind it",
       doc="The write-compute overlap discipline: serial exposes every write, overlapped hides "
           "writes behind the previous pass's compute after the declared shadow-capacity loss.",
       value_schema=WRITE_OVERLAP_SCHEMA, passthrough=True,
       empty_means="serial, shadow capacity 0"),
    _E("write_ports", section=_PC, group="hardware", owner=_OWNER, type=T.INT,
       category=Category.ADVANCED, exposure="user", label="Write Ports",
       effect="Concurrent write ports of the chip",
       doc="Write ports that commit to the arrays concurrently; the exposed write time divides "
           "by them.",
       bounds=(1, None), passthrough=True, empty_means="1 — one write port"),
    _E("inference_rate_hz", section=_PC, group="mapping_strategy", owner=_OWNER, type=T.FLOAT,
       category=Category.ADVANCED, exposure="user", label="Inference Rate", unit="Hz",
       effect="Converts the write-endurance lifetime from inferences to hours",
       doc="The declared inference rate: the lifetime the endurance rows price in inferences "
           "is also reported in hours.",
       bounds=(0.0, None), exclusive_bounds=(True, False), passthrough=True,
       empty_means="lifetime reported in inferences only"),
    _E("min_lifetime_hours", section=_PC, group="hardware", owner=_OWNER, type=T.FLOAT,
       category=Category.ADVANCED, exposure="user", label="Minimum Lifetime", unit="h",
       effect="A program whose write-endurance lifetime falls below this is inadmissible",
       doc="The lifetime floor: a program whose priced endurance lifetime is shorter is "
           "inadmissible by name, with every priced term still reported.",
       bounds=(0.0, None), exclusive_bounds=(True, False), passthrough=True,
       empty_means="no lifetime floor"),
)
