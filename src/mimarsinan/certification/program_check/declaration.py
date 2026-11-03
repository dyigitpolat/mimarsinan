"""[W15] What the static check reads of a deployment's declaration: the semantics, its window, the run's pass transfer,
the encoder and its cycle-accurate trains, the per-event granularity, the half-step fold, the declared core types and
the executors' accumulation and upload dtypes.

One frozen record, built from a resolved pipeline config (``declaration_from_config``) or directly (tests, the fuzz
oracle), so the check never reads a raw config key itself."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE, VERBATIM
from mimarsinan.chip_simulation.activation_semantics import (
    ANALYTICAL_VARIANT, CASCADED_VARIANT, LIF_FAMILY, QUANTIZED_VARIANT, STREAMED_VARIANT,
    resolve_activation_semantics,
)
from mimarsinan.chip_simulation.spiking_semantics import (
    cycle_accurate_lif_trains, lif_execution_synchronized, lif_membrane_init, lif_per_hop_retiming_enabled,
    spike_phase_dither_enabled, ttfs_half_step_folded,
)
from mimarsinan.models.spiking.hybrid.carry import run_pass_transfer

#: The semantics Theorem 1 speaks for (``W1/thm/proofs.md`` section 3), and the named ones it leaves outside.
STREAMED_LIF, WINDOWED_LIF, SYNCHRONIZED_LIF = "streamed_lif", "windowed_lif", "synchronized_lif"
CASCADED_TTFS, SYNCHRONIZED_TTFS = "cascaded_ttfs", "synchronized_ttfs"
VALUE_INTEGER = "value_integer"
ANALYTICAL_TTFS, QUANTIZED_TTFS, VALUE_FLOAT = "analytical_ttfs", "quantized_ttfs", "value_float"
IN_SCOPE: Tuple[str, ...] = (STREAMED_LIF, WINDOWED_LIF, SYNCHRONIZED_LIF, CASCADED_TTFS, SYNCHRONIZED_TTFS,
                             VALUE_INTEGER)
OUTSIDE_SCOPE: Dict[str, str] = {
    ANALYTICAL_TTFS: "analytical TTFS evaluates real-valued activations, so its accumulation is not integer",
    QUANTIZED_TTFS: "quantized TTFS is the analytical closed form on a level grid, not a cycle executor",
    VALUE_FLOAT: "the value domain without a readout law runs the fp64 float path, not the integer law",
}
#: Semantics whose observable depends on spike timing: pinned latencies and delivery delays bind (Lemma 5).
TIMED: Tuple[str, ...] = (STREAMED_LIF, CASCADED_TTFS)
#: The pass transfer each semantics owes a cut (H5-collapse): the raster under streamed LIF, a count (level, value)
#: re-encode otherwise; cascaded TTFS re-emits the latched count at its fused delivery time (a delayed COLLAPSE).
REQUIRED_TRANSFER: Dict[str, str] = {STREAMED_LIF: VERBATIM, WINDOWED_LIF: COLLAPSE, SYNCHRONIZED_LIF: COLLAPSE,
                                     CASCADED_TTFS: COLLAPSE, SYNCHRONIZED_TTFS: COLLAPSE, VALUE_INTEGER: COLLAPSE}
#: The semantics whose stage loop replays carried rasters and applies delivery delays (``replay_carried_slices``, the
#: rate stage loop); the synchronized TTFS and value stage loops read no delay.
REPLAYING: Tuple[str, ...] = (STREAMED_LIF, WINDOWED_LIF, SYNCHRONIZED_LIF, CASCADED_TTFS)
#: The semantics whose host ops may leave a cycle-accurate train a stage reads verbatim (``encode_compute_boundary``).
LIF_SEMANTICS: Tuple[str, ...] = (STREAMED_LIF, WINDOWED_LIF, SYNCHRONIZED_LIF)
#: Mantissa reach of an accumulation dtype: every integer below it is exact (H8); ``int32`` is nevresim's C++ ``int``.
EXACT_INTEGER_REACH: Dict[str, int] = {"float64": 2 ** 53, "float32": 2 ** 24, "int64": 2 ** 63, "int32": 2 ** 31}
#: The executors whose runs Theorem 1 relates, per family: the reference executor (HCM, the value census) and the
#: int64 golden; ``COMPUTE_DTYPE`` (``models/spiking/spiking_config.py``) is float64.
DEFAULT_ACCUMULATION: Dict[str, str] = {"hcm": "float64"}
VALUE_ACCUMULATION: Dict[str, str] = {"value_census": "float64", "int64_golden": "int64"}
#: The dtype an executor uploads weights and input codes in, where the check must bound it: the value census runs
#: ``ValueHybridCoreFlow`` at float64, the deployed read's arithmetic (``pipelining/core/census/arithmetic.py``,
#: ``VALUE_CENSUS_DTYPE``), so every weight and code must be exact in float64 (H8).
VALUE_UPLOAD: Dict[str, str] = {"value_census": "float64"}
PER_EVENT = "per_event"


@dataclass(frozen=True)
class CheckDeclaration:
    """The declaration facts the conditions read. ``core_types``: the declared cores (``max_axons``,
    ``max_neurons``, ``count``) the residency fit reads; ``accumulation``: executor name to dtype name; ``upload``:
    executor name to the dtype it uploads weights and codes in; ``cycle_accurate_trains``: host ops leave the trains
    a stage reads verbatim (LIF only); ``half_step_folded`` and ``comparator_half_step``: Proposition 2's P3 and P4
    bind (a half-step folded into the bias or carried by the compare ladder)."""

    semantics: str
    window: int
    pass_transfer: str
    dithered_encode: bool = False
    firing_granularity: str = "per_cycle"
    membrane_init: float = 0.0
    core_types: Tuple[Mapping[str, Any], ...] = ()
    accumulation: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_ACCUMULATION))
    upload: Mapping[str, str] = field(default_factory=dict)
    cycle_accurate_trains: bool = True
    half_step_folded: bool = False
    comparator_half_step: bool = False

    def __post_init__(self) -> None:
        known = IN_SCOPE + tuple(OUTSIDE_SCOPE)
        if self.semantics not in known:
            raise ValueError(f"CheckDeclaration: semantics {self.semantics!r} is none of {known}")
        if self.pass_transfer not in (VERBATIM, COLLAPSE):
            raise ValueError(f"CheckDeclaration: pass transfer {self.pass_transfer!r}")
        unknown = sorted((set(self.accumulation.values()) | set(self.upload.values())) - set(EXACT_INTEGER_REACH))
        if unknown:
            raise ValueError(f"CheckDeclaration: accumulation dtypes {unknown} have no exact integer reach")

    @property
    def in_scope(self) -> bool:
        return self.semantics in IN_SCOPE

    @property
    def timed(self) -> bool:
        return self.semantics in TIMED

    @property
    def windowed(self) -> bool:
        return self.semantics == WINDOWED_LIF

    @property
    def value(self) -> bool:
        return self.semantics in (VALUE_INTEGER, VALUE_FLOAT)

    @property
    def replays(self) -> bool:
        """The stage loop replays rasters and applies delivery delays (``REPLAYING``)."""
        return self.semantics in REPLAYING

    @property
    def caches_trains(self) -> bool:
        return self.cycle_accurate_trains and self.semantics in LIF_SEMANTICS


def semantics_of_config(config: Mapping[str, Any], *, has_laws: Optional[bool] = None) -> str:
    """The check's semantics id of a resolved config: the value domain by ``core_semantics`` (integer when a
    readout law is declared or carried), then the spiking (family, variant) point, LIF synchronized split by its
    executor: the count executor, per-hop re-timed windowed LIF, or (neither armed) per-cycle segments."""
    if str(config.get("core_semantics", "spiking")) == "mvm":
        integer = bool(has_laws) if has_laws is not None else config.get("readout_law") is not None
        return VALUE_INTEGER if integer else VALUE_FLOAT
    semantics = resolve_activation_semantics(config)
    if semantics.family == LIF_FAMILY:
        if semantics.variant == STREAMED_VARIANT:
            return STREAMED_LIF
        if lif_execution_synchronized(config):
            return SYNCHRONIZED_LIF
        # Without per-hop re-timing the synchronized variant runs every segment on per-cycle cores (host boundaries
        # re-encode counts as under streamed LIF), so it is held to streamed LIF's conditions with the run's transfer.
        return WINDOWED_LIF if lif_per_hop_retiming_enabled(config) else STREAMED_LIF
    return {ANALYTICAL_VARIANT: ANALYTICAL_TTFS, QUANTIZED_VARIANT: QUANTIZED_TTFS,
            CASCADED_VARIANT: CASCADED_TTFS}.get(semantics.variant, SYNCHRONIZED_TTFS)


def declaration_from_config(config: Mapping[str, Any], *, cores: Optional[Sequence[Mapping[str, Any]]] = None,
                            has_laws: Optional[bool] = None,
                            backends: Optional[Mapping[str, str]] = None) -> CheckDeclaration:
    """The declaration of a resolved pipeline config (its ``cores`` unless given), the transfer as the run executes
    it (``run_pass_transfer``); the half-step is folded under synchronized TTFS with activation quantization (the
    floor-ceil convention's bias compensation, ``ttfs_half_step_folded``). ``backends``: the enabled backends whose
    charges must be exact too, each with its accumulation dtype (the gate reads them off the plan)."""
    semantics = semantics_of_config(config, has_laws=has_laws)
    value = semantics in (VALUE_INTEGER, VALUE_FLOAT)
    accumulation = dict(VALUE_ACCUMULATION if value else DEFAULT_ACCUMULATION)
    accumulation.update(backends or {})
    return CheckDeclaration(
        semantics=semantics, window=int(config.get("simulation_steps", 1) or 1),
        pass_transfer=run_pass_transfer(config),
        dithered_encode=spike_phase_dither_enabled(config),
        firing_granularity=str(config.get("firing_granularity", "per_cycle") or "per_cycle"),
        membrane_init=lif_membrane_init(config),
        core_types=tuple(dict(c) for c in (cores if cores is not None else config.get("cores") or ())),
        accumulation=accumulation, upload=dict(VALUE_UPLOAD) if value else {},
        cycle_accurate_trains=cycle_accurate_lif_trains(config),
        half_step_folded=ttfs_half_step_folded(config, floor_ceil_convention=semantics == SYNCHRONIZED_TTFS),
        comparator_half_step=bool(config.get("comparator_half_step", False)))
