"""Registry entries: the executed residency plan selection, position multiplexing, its pipelined form with its layer
copies and its batch pipeline, and its banked reduction cores."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mimarsinan.chip_simulation.activation_semantics import resolve_activation_semantics
from mimarsinan.chip_simulation.spiking_point import resolve_spiking_point
from mimarsinan.chip_simulation.core_semantics import is_mvm_core_semantics, resolve_core_semantics
from mimarsinan.config_schema.registry.decision_classes import COST_ONLY
from mimarsinan.config_schema.registry.relevance import Relevance as R
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
)
from mimarsinan.config_schema.registry.value_schema import RequiredIff, ValueSchema
from mimarsinan.mapping.lanes.ir.reduction_banks import KEY as REDUCTION_BANKS
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
    _E("layer_replication", section="platform_constraints", group="mapping_strategy",
       owner="mapping/support/pipelined_multiplexing", type=T.BOOL,
       category=Category.ADVANCED, exposure="user", label="Layer Replication",
       effect="A wavefront group's free cores hold more copies of its layers, so each pass serves more positions",
       doc="Mapping decision under pipelined multiplexing: true fills the cores a wavefront group's layout leaves free "
           "with more copies of the group's banks, one at a time to the bank with the most instances per copy while "
           "the layout packs, and each instance runs on its bank's copy that is free earliest, so the program costs "
           "fewer passes for more programmed cells; false or absent keeps one copy per bank. Outputs are unchanged.",
       passthrough=True, relevant=R.when_true("pipelined_multiplexing"),
       legal_values=lambda cfg: _replication_legal(cfg),
       empty_means="one copy per bank, the planner byte for byte"),
    _E("wavefront_group_cores", section="platform_constraints", group="mapping_strategy",
       owner="mapping/support/pipelined_multiplexing", type=T.INT,
       category=Category.ADVANCED, exposure="user", label="Wavefront Group Cores",
       effect="One copy of a wavefront group's layers takes at most this many cores, the rest hold copies",
       doc="Mapping decision under layer replication: a wavefront group closes when one copy of its banks no longer "
           "packs on this many cores, so groups end earlier and more of the declared cores hold copies. Outputs are "
           "unchanged.",
       passthrough=True, relevant=R.when_true("layer_replication"), bounds=(1, None),
       empty_means="every declared core"),
    _E("batch_pipeline_slots", section="platform_constraints", group="mapping_strategy",
       owner="mapping/support/pipelined_multiplexing", type=T.INT,
       category=Category.ADVANCED, exposure="user", label="Batch Pipeline Slots",
       effect="A run of the program serves this many inputs, which share the passes of every wavefront group",
       doc="Mapping decision under pipelined multiplexing: a run of the program serves B inputs. The instances of "
           "input 1, then input 2, and so on share each wavefront group's passes, each in the first pass after its own "
           "input's producers on its bank's copy that is free earliest, so a group's start-up passes are paid once "
           "per run; a pass outside every wavefront group runs once per input. The chip must state how many inputs' "
           "carried values it holds at once (carry_input_slots), else B is refused by name, as is a B above it. "
           "Absent, one input per run. Outputs are unchanged.",
       passthrough=True, relevant=R.when_true("pipelined_multiplexing"), bounds=(2, None),
       empty_means="one input per run, the planner byte for byte"),
    _E("carry_input_slots", section="platform_constraints", group="hardware",
       owner="mapping/support/pipelined_multiplexing", type=T.INT,
       category=Category.ADVANCED, exposure="user", label="Carried Input Slots",
       effect="How many inputs' carried values the chip holds at once",
       doc="Chip component: the number of inputs whose values carried between passes the chip holds apart at once. "
           "It admits a batch pipeline of at most this many inputs per run (batch_pipeline_slots); a chip that states "
           "none refuses a declared batch_pipeline_slots by name, and a larger one is refused by name. Stated alone, "
           "it changes nothing.",
       passthrough=True, relevant=R.when_set("batch_pipeline_slots"), bounds=(1, None),
       empty_means="unstated: a declared batch_pipeline_slots is refused by name"),
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
    _E(REDUCTION_BANKS, domain="value", section="platform_constraints", group="mapping_strategy",
       owner="mapping/lanes/ir/reduction_banks", type=T.BOOL,
       category=Category.ADVANCED, exposure="user", label="Shared Reduction Banks",
       effect="A pool lane's reduction cores of one matrix share one bank, served in turn on one copy",
       doc="Mapping decision under position multiplexing: true makes the window groups of each pool lane that hold "
           "one 0/1 matrix and read one producer grid the instances of one shared bank, so the one copy of each bank "
           "serves them in turn over passes (one bank per distinct matrix and grid, not one core per window group): "
           "the chip commits fewer cells and keeps a resident program on fewer cores, at the cost of passes; false or "
           "absent gives every window group its own core. A reduction's reduced cores keep their own weights. "
           "Outputs are unchanged.",
       passthrough=True, relevant=R.when_true("position_multiplexing"),
       legal_values=lambda cfg: _reduction_banks_legal(cfg),
       empty_means="every window group of a pool lane its own core, the lane fusion byte for byte"),
)

#: Declarations whose cores the shape-only plane cannot see: IR elimination compacts bank instances into owned cores
#: (no shared copy on the executed plane), and depth balancing inserts relay cores the layout walk does not record.
_UNSEEN_BY_THE_SHAPE_ONLY_PLANE = ("pruning", "lif_depth_balancing_relays")


def _run_value(cfg, key: str) -> Any:
    """``key`` as the run reads it: the declared value, else the conversion recipe's for the run's mode (the
    ConversionPolicy SSOT that ``recipe_fold`` folds beneath the declarations: the recipe of every per-cycle cascade
    arms the depth-balancing relays), its schedule resolved from the declared variant as the plan resolves it
    ([U-TGAUGE5]: a synchronized timing declaration is no cascade); a value-domain run folds no spiking recipe, and an
    unknown mode (its own schema's keyed error) none."""
    declared = cfg.get(key)
    if declared is not None or is_mvm_core_semantics(resolve_core_semantics(cfg)):
        return declared
    try:
        recipe = ConversionPolicy.derive(resolve_spiking_point(cfg))
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
    domain, streamed LIF and every TTFS variant; the synchronized LIF variant refuses it by name under every
    lif_window_firing, for two reasons: its recipe arms per-hop re-timing (each level its own unit), and under
    integrate_then_emit one membrane holds a window's charge through its emission window, so a neuron is busy 2T
    cycles an inference and no executor declares a window-end count emission. An unresolvable semantics axis rules
    nothing out here: its own keyed error is the single truth."""
    if True not in _multiplexing_legal(cfg):
        return (False,)
    try:
        windowed = (not is_mvm_core_semantics(resolve_core_semantics(cfg))
                    and resolve_activation_semantics(cfg).mode_id == "lif_sync")
    except ValueError:
        windowed = False
    return (False,) if windowed else (False, True)


def _replication_legal(cfg):
    """True exactly where pipelined multiplexing may be true: the copies are copies of its wavefront groups."""
    return _pipelined_legal(cfg)


def _reduction_banks_legal(cfg):
    """True exactly where position multiplexing may be true: the decision shapes what its one copy serves."""
    return (False,) if True not in _multiplexing_legal(cfg) else (False, True)
