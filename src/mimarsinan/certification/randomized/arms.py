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
from mimarsinan.mapping.ir import legacy_convert
from mimarsinan.mapping.packing import canonical
from mimarsinan.mapping.platform import core_residency
from mimarsinan.mapping.platform.core_residency import ALL_SINGLETON_NAMES, SingletonProperty, residency_key
from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE, VERBATIM

ON = "on"
#: The hypothesis-on arm restricted to scheduled declarations of small core budgets, so passes are exercised.
ON_SCHEDULED = "on_scheduled"
BOUNDARY_ANALYTICAL = "boundary_analytical_ttfs"
#: The hypothesis-on arm under value-keyed runtime residency classes (layers may share cores): a harness
#: setting (``legacy_convert._residency_class_of`` replaced), outside the toolchain-as-shipped line.
ON_VALUES = "on_values_classes"
#: The hypothesis-on arm beyond the generator's declared window range (T in 9-64), scheduled synchronized
#: deployments: a targeted stratum for defects that bind only at long windows (the synchronized LIF count
#: rebuild, H8), reported apart from the declared-range line.
ON_LONG_WINDOW = "on_long_window"


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
    disjoint networks; ``timesteps`` replaces the generator's window range (a targeted stratum).
    """

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


_SCHEDULED = {"allow_scheduling": True, "max_schedule_passes": 8, "count": (1, 4)}

_CO_RESIDENT = {"axons": (32, 64), "neurons": (32, 64), "count": (2, 6), "allow_scheduling": True,
                "max_schedule_passes": 8}
_MODEL_LEVEL = "model level only: "

ARMS: Dict[str, Arm] = {arm.id: arm for arm in (
    Arm(ON, "", ""),
    Arm(ON_SCHEDULED, "", "", declaration=_SCHEDULED, stream=ON_SCHEDULED),
    Arm("H1-theta", "H1-theta", "the singleton set co-residency admits on (the threshold no longer)",
        motifs=("siblings",), declaration=_CO_RESIDENT, setting=_values_classes, switch=_threshold_unconstrained),
    Arm("H1-latency", "H1-latency", "the latency clause of the packing predicate",
        semantics=(STREAMED_LIF, CASCADED_TTFS, SYNCHRONIZED_LIF, WINDOWED_LIF), motifs=("narrow_chain",),
        declaration=_CO_RESIDENT, setting=_values_classes, switch=_latency_unchecked),
    Arm(ON_VALUES, "", "", declaration=_CO_RESIDENT, motifs=("siblings", "narrow_chain"), setting=_values_classes,
        stream=ON_VALUES),
    Arm(ON_LONG_WINDOW, "", "", semantics=(SYNCHRONIZED_LIF, SYNCHRONIZED_TTFS), declaration=_SCHEDULED,
        stream=ON_LONG_WINDOW, timesteps=(9, 64)),
    Arm("H2", "H2", _MODEL_LEVEL + "window coverage relates the identity program to N", exercisable=False),
    Arm("H3", "H3", _MODEL_LEVEL + "terminal overshoot relates the identity program to N", exercisable=False),
    Arm("H4", "H4", _MODEL_LEVEL + "chase completion relates the identity program to N", exercisable=False),
    Arm("H5-collapse", "H5-collapse", "the pass transfer of a streamed run (collapse instead of verbatim)",
        semantics=(STREAMED_LIF,), motifs=("chain",), declaration=_SCHEDULED, transfer=COLLAPSE),
    Arm("H5-collapse/windowed", "H5-collapse", "the pass transfer of a windowed run (verbatim instead of "
                                               "collapse)",
        semantics=(WINDOWED_LIF, SYNCHRONIZED_LIF, CASCADED_TTFS, SYNCHRONIZED_TTFS),
        motifs=("chain",), declaration=_SCHEDULED, transfer=VERBATIM),
    Arm("H5-skew", "H5-skew", "none on this base: U1 cuts without pass projection (R1 pins, R2 delivery "
                              "delays); the arm targets timing-moving cuts", semantics=(STREAMED_LIF, CASCADED_TTFS),
        motifs=("skip",), declaration=_SCHEDULED, mechanism="u1_cut"),
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
    Arm("H10", "H10", "residency integrity needs G5's residency_plan (from CL1)", exercisable=False),
    Arm("R6", "R6", "none on this base: the count and value walks order a segment's cores by stored latency; the "
                    "arm targets a producer and its consumer at one latency",
        semantics=(SYNCHRONIZED_LIF, SYNCHRONIZED_TTFS, VALUE), motifs=("skip",), mechanism="latency_order_walk"),
    Arm("H0", "H0", "none on this base: a windowed segment holding a coalescing group gets no level stages and "
                    "runs streamed (segment_depth_groups returns None); the arm targets coalescing under windowed "
                    "LIF", semantics=(WINDOWED_LIF,), motifs=("wide_fan_in",),
        declaration={"allow_coalescing": True, "axons": (8, 24)}, mechanism="retiming_dropped"),
    Arm("R4", "R4", "none on this base: a phase-dithered Uniform encode rotates each slice's comb by its channel "
                    "index in the deployed stage input, not in the fused one; the arm declares the dithered encode on "
                    "both programs under windowed LIF (no coalescing, so H0 stays on)", semantics=(WINDOWED_LIF,),
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
