"""X3a arms: the hypothesis-on arm and one hypothesis-off arm per exercisable hypothesis, one field each."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Tuple

from mimarsinan.certification.randomized.instance import (
    ANALYTICAL_TTFS,
    CASCADED_TTFS,
    IN_SCOPE_SEMANTICS,
    STREAMED_LIF,
    SYNCHRONIZED_LIF,
    SYNCHRONIZED_TTFS,
    INPUT,
    NEURAL,
    VALUE,
    WINDOWED_LIF,
    Instance,
)
import mimarsinan.chip_simulation.ttfs.ttfs_segment as ttfs_segment
from mimarsinan.chip_simulation.value_run import value_execution
from mimarsinan.mapping.ir import legacy_convert
from mimarsinan.mapping.layout import segmentation
from mimarsinan.mapping.packing import canonical, hybrid_build_scheduled, pass_timing, retimed_levels
from mimarsinan.mapping.platform import core_residency
from mimarsinan.mapping.platform.core_residency import ALL_SINGLETON_NAMES, SingletonProperty, residency_key
from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE, VERBATIM
from mimarsinan.models.spiking.hybrid.executors import sync_counts

ON = "on"
#: The hypothesis-on arm restricted to scheduled declarations of small core budgets, so passes are exercised.
ON_SCHEDULED = "on_scheduled"
BOUNDARY_ANALYTICAL = "boundary_analytical_ttfs"
#: The hypothesis-on arm under value-keyed runtime residency classes (layers may share cores): a harness
#: setting (``legacy_convert._residency_class_of`` replaced), outside the toolchain-as-shipped line.
ON_VALUES = "on_values_classes"
#: The hypothesis-on arm beyond the declared window range (T in 9-64), scheduled synchronized deployments: a targeted
#: stratum for defects that bind only at long windows (H8's count rebuild), reported apart from the declared range.
ON_LONG_WINDOW = "on_long_window"
#: CL-SG: the hypothesis-on arm with every core type declaring per-axon weight sign (mixed-sign rows split into
#: W+ then W- pairs), declared draws otherwise, and its scheduled twin (small core budgets, so passes are cut).
ON_SIGN_SPLIT = "on_sign_split"
ON_SIGN_SPLIT_SCHEDULED = "on_sign_split_scheduled"
#: Closure fix round (X1): TTFS, the encoder on chip (the first segment reads the raw input), T in 3-12, half per-axon.
ON_TTFS_OFFLOAD = "on_ttfs_offload"


@dataclass(frozen=True)
class Arm:
    """One arm. ``off_field`` names the single thing an off arm changes; the rest targets the generator.

    ``semantics`` restricts the instances to those the hypothesis speaks about; ``motifs`` and
    ``declaration`` force the structure that exercises it (targeted motifs, never a different field);
    ``transfer`` overrides the run's pass-boundary discipline; ``scale`` multiplies every weight, bias
    and threshold (the same function in exact arithmetic); ``switch`` disables one toolchain guard;
    ``setting`` is a harness setting the paired control keeps (outside the toolchain as shipped);
    ``phase_dither`` declares the phase-dithered Uniform encode on both programs (a function-bearing axis the
    paired control keeps); ``stream`` keys the arm's own generator stream, so strata over one seed range draw
    disjoint networks; ``timesteps`` replaces the generator's window range (a targeted stratum);
    ``encoding_placement`` places the input encoder on the host or on chip (``mark_encoding_layers``)."""

    id: str
    hypothesis: str
    off_field: str
    semantics: Tuple[str, ...] = IN_SCOPE_SEMANTICS
    motifs: Tuple[str, ...] = ()
    declaration: Mapping[str, Any] = field(default_factory=dict)
    transfer: Optional[str] = None
    scale: int = 1
    value_scale: int = 1
    value_dtype: str = "float64"
    encoder: str = "Uniform"
    phase_dither: bool = False
    setting: Optional[Callable[[], contextlib.AbstractContextManager]] = None
    switch: Optional[Callable[[], contextlib.AbstractContextManager]] = None
    exercisable: bool = True
    mechanism: str = ""
    stream: str = ""
    timesteps: Optional[Tuple[int, int]] = None
    encoding_placement: str = "subsume"

    @property
    def harness_setting(self) -> bool:
        """Whether the arm runs under a harness setting (a replaced toolchain function), not as shipped."""
        return self.setting is not None

    @property
    def changes_a_field(self) -> bool:
        """Whether the arm switches something off itself (else the base toolchain violates the hypothesis)."""
        return self.control() != self

    @contextlib.contextmanager
    def engaged(self) -> Iterator[None]:
        """The arm's setting (kept by its control) and, inside it, its switch (the one field off)."""
        with contextlib.ExitStack() as stack:
            for factory in (self.setting, self.switch):
                if factory is not None:
                    stack.enter_context(factory())
            yield

    def control(self) -> "Arm":
        """The paired control: the same targeting with the one field left at its hypothesis-on value."""
        return replace(self, transfer=None, scale=1, value_scale=1, value_dtype="float64", switch=None,
                       encoder="Uniform")


@contextlib.contextmanager
def _patched(owner: Any, name: str, value: Any) -> Iterator[None]:
    saved = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, saved)


#: H1 off: the hardware core still holds ONE threshold (its first resident's), but packing no longer requires
#: its residents to agree: residency classes ignore the threshold and the residency check accepts any.
UNCONSTRAINED_THRESHOLD = frozenset(ALL_SINGLETON_NAMES - {"threshold"})
_ANY_THRESHOLD = tuple(
    SingletonProperty(p.name, (lambda _a, _b: True) if p.name == "threshold" else p.compare)
    for p in core_residency.CORE_SINGLETON_PROPERTIES)


def _values_classes(constrained=ALL_SINGLETON_NAMES) -> contextlib.AbstractContextManager:
    """Runtime residency classes keyed on the stored values (the layout finalizer's rule), not on the
    source perceptron: the runtime plane's segment subgraphs carry no layout record, so its classes fall
    back to the perceptron (``legacy_convert._residency_class_of``) and no two layers ever share a core."""
    interned: Dict[Any, int] = {}

    def class_of(neural_core, graph, _pi):
        banks = getattr(graph, "weight_banks", None)
        return interned.setdefault(residency_key(neural_core, constrained=constrained, weight_banks=banks),
                                   len(interned))

    return _patched(legacy_convert, "_residency_class_of", class_of)


@contextlib.contextmanager
def _threshold_unconstrained() -> Iterator[None]:
    with _values_classes(UNCONSTRAINED_THRESHOLD), \
            _patched(core_residency, "CORE_SINGLETON_PROPERTIES", _ANY_THRESHOLD):
        yield


def _latency_unchecked() -> contextlib.AbstractContextManager:
    return _patched(canonical, "_read_latency", lambda _obj: None)


def _latest_resident_latency(pl, fused, _label):
    """A hard core whose residents the fused segment runs at several latencies starts at the latest of them
    (ChipLatency's ALAP raise, the timing the base tree ran) instead of being refused."""
    return [max((fused[n] for n in pl.nodes(h)), default=0) for h in range(len(pl.cores))]


def one_latency_unrefused() -> contextlib.AbstractContextManager:
    """G5's one-latency refusal off (``pass_timing._core_latencies``), the H1-latency assertion site on the merged
    tree: its necessity witness runs the co-residency it refuses (``test_witnesses.TestLatentDefects``)."""
    return _patched(pass_timing, "_core_latencies", _latest_resident_latency)


@contextlib.contextmanager
def _own_pass_timing() -> Iterator[None]:
    """H5-skew off: the pass projection keeps its refusals but pins no pass at the fused latencies (R1) and replays
    no carried slice at its fused delivery delay (R2), so each pass runs at its own timing (U1 before G5)."""
    def refusals_only(stage, pl, fused, *_produced):
        pass_timing._core_latencies(pl, fused, stage.name)

    with _patched(pass_timing, "_pin", refusals_only), _patched(pass_timing, "_delay", refusals_only):
        yield


def _latency_order(_sources, latencies):
    """A segment's cores by stored latency, then index: the walk order before R6 (no dependency check)."""
    return sorted(range(len(latencies)), key=lambda i: (int(latencies[i]), i))


@contextlib.contextmanager
def _latency_order_walks() -> Iterator[None]:
    """R6 off: the value walk, the synchronized LIF count walk and the TTFS segment walk order a segment's cores by
    stored latency (the base tree's walks) instead of in dependency order."""
    with _patched(value_execution, "dependency_order", _latency_order), \
            _patched(sync_counts, "dependency_order", _latency_order), \
            _patched(ttfs_segment, "dependency_order", lambda seg: _latency_order(None, seg.latencies)):
        yield


def _base_depth_groups(original):
    def depth_groups(nodes):
        """The base's rule: no levels for a segment holding any coalescing group (it then runs streamed)."""
        if any(getattr(n, "coalescing_group_id", None) is not None for n in nodes):
            return None
        return original(nodes)
    return depth_groups


@contextlib.contextmanager
def _coalescing_unleveled() -> Iterator[None]:
    """H0 off: a segment holding a coalescing group gets no depth levels (the base tree's ``segment_depth_groups``),
    so a windowed segment holding one runs streamed, at every reader of the rule."""
    groups = _base_depth_groups(segmentation.segment_depth_groups)
    with _patched(segmentation, "segment_depth_groups", groups), \
            _patched(hybrid_build_scheduled, "segment_depth_groups", groups), \
            _patched(retimed_levels, "segment_depth_groups", groups):
        yield


_SCHEDULED = {"allow_scheduling": True, "max_schedule_passes": 8, "count": (1, 4)}

_CO_RESIDENT = {"axons": (32, 64), "neurons": (32, 64), "count": (2, 6), "allow_scheduling": True,
                "max_schedule_passes": 8}
_MODEL_LEVEL = "model level only: "

ARMS: Dict[str, Arm] = {arm.id: arm for arm in (
    Arm(ON, "", ""),
    Arm(ON_SCHEDULED, "", "", declaration=_SCHEDULED, stream=ON_SCHEDULED),
    Arm("H1-theta", "H1-theta", "the singleton set co-residency admits on (the threshold no longer)",
        motifs=("siblings",), declaration=_CO_RESIDENT, setting=_values_classes, switch=_threshold_unconstrained),
    Arm("H1-latency", "H1-latency", "the latency clause of the packing predicate; retired at CL1: the pass "
                                    "projection refuses by name (PassTimingError) every co-residency of two fused "
                                    "latencies the predicate admits, so the off arm runs no deployment that violates "
                                    "it (the hypothesis is asserted at run time)",
        semantics=(STREAMED_LIF, CASCADED_TTFS, SYNCHRONIZED_LIF, WINDOWED_LIF), motifs=("narrow_chain",),
        declaration=_CO_RESIDENT, setting=_values_classes, switch=_latency_unchecked, exercisable=False),
    Arm(ON_VALUES, "", "", declaration=_CO_RESIDENT, motifs=("siblings", "narrow_chain"), setting=_values_classes,
        stream=ON_VALUES),
    Arm(ON_LONG_WINDOW, "", "", semantics=(SYNCHRONIZED_LIF, SYNCHRONIZED_TTFS), declaration=_SCHEDULED,
        stream=ON_LONG_WINDOW, timesteps=(9, 64)),
    Arm(ON_SIGN_SPLIT, "", "", declaration={"sign_split_rate": 1.0}, stream=ON_SIGN_SPLIT),
    Arm(ON_SIGN_SPLIT_SCHEDULED, "", "", declaration={**_SCHEDULED, "sign_split_rate": 1.0},
        stream=ON_SIGN_SPLIT_SCHEDULED),
    Arm(ON_TTFS_OFFLOAD, "", "", semantics=(SYNCHRONIZED_TTFS, CASCADED_TTFS), declaration={"sign_split_rate": 0.5},
        stream=ON_TTFS_OFFLOAD, timesteps=(3, 12), encoding_placement="offload"),
    Arm("H2", "H2", _MODEL_LEVEL + "window coverage relates the identity program to N", exercisable=False),
    Arm("H3", "H3", _MODEL_LEVEL + "terminal overshoot relates the identity program to N", exercisable=False),
    Arm("H4", "H4", _MODEL_LEVEL + "chase completion relates the identity program to N", exercisable=False),
    Arm("H5-collapse", "H5-collapse", "the pass transfer of a streamed run (collapse instead of verbatim)",
        semantics=(STREAMED_LIF,), motifs=("chain",), declaration=_SCHEDULED, transfer=COLLAPSE),
    Arm("H5-collapse/windowed", "H5-collapse", "the pass transfer of a windowed run (verbatim instead of "
                                               "collapse); retired at CL1: every carried wire of a windowed run is "
                                               "read by a level unit that re-encodes its count (G5's R5), and the "
                                               "TTFS and count executors read no transfer, so it changes no output",
        semantics=(WINDOWED_LIF, SYNCHRONIZED_LIF, CASCADED_TTFS, SYNCHRONIZED_TTFS),
        motifs=("chain",), declaration=_SCHEDULED, transfer=VERBATIM, exercisable=False),
    Arm("H5-skew", "H5-skew", "the pass projection's timing (G5's R1 pins and R2 delivery delays): each pass of a "
                              "timing-moving cut at its own timing", semantics=(STREAMED_LIF, CASCADED_TTFS),
        motifs=("skip",), declaration=_SCHEDULED, switch=_own_pass_timing),
    Arm("H6", "H6", _MODEL_LEVEL + "the toolchain's fused core is one exact dot product; no field declares "
                    "a partial-sum width", exercisable=False),
    Arm("H7", "H7", "fire checks at the neuron's own synaptic events in canonical rank order on per-event cores "
                    "(FPGA branch only)", exercisable=False),
    Arm("H8", "H8", "the accumulator bound: partial sums past the executor's exact integer range (2^53 for "
                    "the float64 spiking executors, 2^24 for the float32 value executor)",
        motifs=("wide_fan_in",), declaration={"allow_coalescing": True}, scale=(1 << 49) + 1,
        value_scale=(1 << 20) + 1, value_dtype="float32"),
    Arm("H9", "H9", "a stage before its producer reads a missing node and crashes (no silent counterexample "
                    "exists at code level); the stage loop's assertion names it", exercisable=False),
    Arm("H10", "H10", _MODEL_LEVEL + "the code level draws residency_plan from CL1 (its step changes the pass "
                                     "partition and the pinned set), but no executor holds a core's programmed "
                                     "contents across passes, so a broken residency is not observable there",
        exercisable=False),
    Arm("R6", "R6", "the walk order: the value, synchronized LIF count and TTFS segment walks by stored latency "
                    "instead of dependency order; the arm targets a producer and its consumer at one latency",
        semantics=(SYNCHRONIZED_LIF, SYNCHRONIZED_TTFS, VALUE), motifs=("skip",), switch=_latency_order_walks),
    Arm("H0", "H0", "the leveling of a coalescing group as one unit (segment_depth_groups, CL1 fix round): off, a "
                    "windowed segment holding a coalescing group gets no level stages and runs streamed (the base "
                    "tree's rule); the arm targets coalescing under windowed LIF", semantics=(WINDOWED_LIF,),
        motifs=("wide_fan_in",), declaration={"allow_coalescing": True, "axons": (8, 24)},
        switch=_coalescing_unleveled),
    Arm("R4", "R4", "none on this base: a phase-dithered Uniform encode rotates each slice's comb by its channel "
                    "index in the deployed stage input, not in the fused one; the arm declares the dithered encode on "
                    "both programs under windowed LIF (no coalescing)", semantics=(WINDOWED_LIF,),
        motifs=("chain",), declaration={**_SCHEDULED, "allow_coalescing": False}, phase_dither=True,
        mechanism="position_dependent_encode"),
    Arm(BOUNDARY_ANALYTICAL, "", "boundary row: analytical TTFS evaluates real-valued activations (outside the "
                                 "integer scope by name), measured under the hypothesis-on draws",
        semantics=(ANALYTICAL_TTFS,)),
)}
#: The arms that probe the box's hypotheses (the on arms and the boundary row excluded).
HYPOTHESIS_ARMS = tuple(a for a, arm in ARMS.items() if arm.hypothesis)


def scaled(instance: Instance, arm: Arm) -> Instance:
    """``instance`` with its magnitudes raised past the exact range (the H8 off arm), else unchanged.

    A spiking layer's weights, bias and threshold scale together (the same integer function in exact
    arithmetic); under the value semantics the layers reading the input scale their weights and bias.
    """
    factor = arm.value_scale if instance.semantics == VALUE else arm.scale
    if factor == 1:
        return instance
    layers = tuple(
        replace(layer, weights=tuple(tuple(w * factor for w in row) for row in layer.weights),
                bias=tuple(b * factor for b in layer.bias),
                theta=layer.theta if instance.semantics == VALUE else layer.theta * factor)
        if layer.kind == NEURAL and (instance.semantics != VALUE or INPUT in layer.sources) else layer
        for layer in instance.network.layers)
    return replace(instance, network=replace(instance.network, layers=layers))
