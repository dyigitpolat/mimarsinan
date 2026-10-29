"""Registry entries: the executed residency plan selection, position multiplexing and its pipelined form."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mimarsinan.chip_simulation.activation_semantics import effective_legacy_spiking_mode, resolve_activation_semantics
from mimarsinan.chip_simulation.core_semantics import is_mvm_core_semantics, resolve_core_semantics
from mimarsinan.config_schema.registry.decision_classes import COST_ONLY
from mimarsinan.config_schema.registry.relevance import Relevance as R
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
)
from mimarsinan.config_schema.registry.value_schema import RequiredIff, ValueSchema
from mimarsinan.mapping.support.period_census import RESIDENCY_STEP_RULES, STEP_INDEX
from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy

RESIDENCY_PLAN_SCHEMA = ValueSchema(
    kind=T.JSON,
    fields={
        "step": ValueSchema(kind=T.ENUM, options=RESIDENCY_STEP_RULES, decision_class=COST_ONLY,
                            doc="which residency step executes: passes and pinned set only"),
        "index": ValueSchema(kind=T.INT, low=0, decision_class=COST_ONLY,
                             doc="index of the executed step: passes and pinned set only"),
    },
    required=("step",),
    required_iff=(RequiredIff("index", "step", (STEP_INDEX,)),),
)

ENTRIES = (
    _E("residency_plan", section="platform_constraints", group="mapping_strategy",
       owner="mapping/support/schedule/residency_composition", type=T.JSON,
       category=Category.ADVANCED, exposure="user", label="Residency Plan",
       effect="Executes a planned residency step: pinned cores stay programmed across periods",
       doc="Selects the residency plan the scheduled build executes: floor runs the minimum-r "
           "step, stateless the stateless plan, index the step of that index. The plan pins "
           "cores across periods and rewrites the rest; outputs are those of the U1 schedule.",
       value_schema=RESIDENCY_PLAN_SCHEMA, passthrough=True,
       relevant=R.when_true("allow_scheduling"),
       empty_means="U1's pass planner, byte for byte"),
    _E("position_multiplexing", section="platform_constraints", group="mapping_strategy",
       owner="mapping/support/position_multiplexing", type=T.BOOL,
       category=Category.ADVANCED, exposure="user", label="Position Multiplexing",
       effect="One copy of each shared weight bank serves its output positions in turn, pass by pass",
       doc="Mapping decision: true holds one copy of every shared weight bank (a conv's kernel, a token-wise "
           "layer's weights) and serves its output positions in turn over passes on that copy, so the chip "
           "commits each weight once per run at the cost of passes, latency and carry; false or absent "
           "gives every position its own copy wherever the chip holds them. Outputs are unchanged.",
       passthrough=True, relevant=R.when_true("allow_scheduling"),
       legal_values=lambda cfg: _multiplexing_legal(cfg),
       empty_means="every position its own copy, the planner byte for byte"),
    _E("pipelined_multiplexing", section="platform_constraints", group="mapping_strategy",
       owner="mapping/support/pipelined_multiplexing", type=T.BOOL,
       category=Category.ADVANCED, exposure="user", label="Pipelined Position Multiplexing",
       effect="Layers share passes as a wavefront over the resident copies of their banks",
       doc="Mapping decision under position multiplexing: true lets consecutive layers whose bank copies fit the "
           "declared cores together share passes as a wavefront (a position runs in the first pass after the "
           "positions it reads), each group's first pass loading every copy of the group and every later pass "
           "running on them, so the program costs fewer passes for the same programmed cells and carry; false or "
           "absent runs one level per pass. Outputs are unchanged.",
       passthrough=True, relevant=R.when_true("position_multiplexing"),
       legal_values=lambda cfg: _pipelined_legal(cfg),
       empty_means="one level per pass under position multiplexing, the planner byte for byte"),
    _E("carry_buffer_bytes", section="platform_constraints", group="hardware",
       owner="deployment_record/quantities/value_domain", type=T.INT,
       category=Category.ADVANCED, exposure="user", label="On-Chip Carry Buffer (bytes)", unit="B",
       effect="An on-chip buffer holds the values a wavefront group produces and reads itself",
       doc="Chip component: an on-chip carry buffer of this many bytes. A value a pipelined wavefront group produces "
           "and reads itself stays in it (written once, read once) instead of crossing the chip boundary, priced at "
           "the target's e_carry_buffer_per_byte, which the physics must declare with its source or the energy is "
           "refused by name; the buffer leaks and occupies silicon on its declared bytes, filled or not "
           "(p_carry_buffer_static_per_byte and area_carry_buffer_per_byte, each declared with its source or the "
           "energy or the area refused by name); a group whose live carry, sized in the run's domain (values at "
           "activation_bits, or the pass transfer's rasters or counts), exceeds the buffer is refused by name. "
           "Undeclared, every carried value crosses the boundary (off chip), as the chip without one runs it.",
       passthrough=True, relevant=R.when_true("pipelined_multiplexing"), bounds=(1, None),
       empty_means="no on-chip carry buffer: the carry crosses the chip boundary"),
)

#: Declarations whose cores the shape-only plane cannot see: IR elimination compacts bank instances into owned cores
#: (no shared copy on the executed plane), and depth balancing inserts relay cores the layout walk does not record.
_UNSEEN_BY_THE_SHAPE_ONLY_PLANE = ("pruning", "lif_depth_balancing_relays")


def _run_value(cfg, key: str) -> Any:
    """``key`` as the run reads it: the declared value, else the conversion recipe's for the run's mode (the
    ConversionPolicy SSOT that ``recipe_fold`` folds beneath the declarations: the LIF recipe arms the depth-balancing
    relays); a value-domain run folds no spiking recipe, and an unknown mode (its own schema's keyed error) none."""
    declared = cfg.get(key)
    if declared is not None or is_mvm_core_semantics(resolve_core_semantics(cfg)):
        return declared
    try:
        recipe = ConversionPolicy.derive(effective_legacy_spiking_mode(cfg), cfg.get("ttfs_cycle_schedule"),
                                         spiking_variant=cfg.get("spiking_variant"))
    except ValueError:
        return declared
    return recipe.knobs.get(key)


def _multiplexing_legal(cfg):
    """Only false where the run changes the executed cores the shape-only plane plans: both planes refuse it. Each
    such key is judged as the run reads it (``_run_value``; carried minor N1 of the pack verify: a LIF document that
    leaves the relays undeclared runs them). An operand core type (N2) is one too: the shape-only plane keeps a product
    as a host op that the executed plane lowers onto operand cores, whose payload is rewritten every pass, so no run of
    passes serves one copy of it. [U-L] Executed pool lanes (WI-5e, class P) are admitted: every core the lane fusion
    inserts carries a layout record of its own (``lanes/ir/layout_records.py``), which the multiplexing planner orders
    by its level in the fused program, and the shape-only plane walks the same fused program."""
    operand = any(isinstance(c, Mapping) and c.get("operand") is not None for c in cfg.get("cores") or ())
    unseen = any(_run_value(cfg, key) for key in _UNSEEN_BY_THE_SHAPE_ONLY_PLANE)
    return (False,) if operand or unseen else (False, True)


def _pipelined_legal(cfg):
    """True only where position multiplexing may be true and the run's executors compose a wavefront pass: the value
    domain, streamed LIF and every TTFS variant; windowed LIF (the synchronized variant, whose recipe arms per-hop
    re-timing: each level its own unit) refuses it by name. An unresolvable semantics axis rules nothing out here: its
    own keyed error is the single truth."""
    if True not in _multiplexing_legal(cfg):
        return (False,)
    try:
        windowed = (not is_mvm_core_semantics(resolve_core_semantics(cfg))
                    and resolve_activation_semantics(cfg).mode_id == "lif_sync")
    except ValueError:
        windowed = False
    return (False,) if windowed else (False, True)
