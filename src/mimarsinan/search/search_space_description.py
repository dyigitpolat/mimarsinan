"""Single source of truth for describing the joint NAS + HW search space.

The architecture-search step has historically rendered the same
`(arch_options, core_*_bounds, num_core_types, target_tq, ...)` tuple into
several slightly-different shapes:

* a pseudo-JSON schema for the AgentEvolve LLM prompt,
* an example configuration the LLM can imitate,
* a free-text constraint description for the same LLM,

and a fourth view is needed for the compilagent integration: a tuple of
`compilagent.Lever`s carrying the same bounds but typed for the
`SearchSpace` returned by `Backend.derive_search_space(...)`.

To avoid four copies of the same arithmetic, this module exposes one
`SearchSpaceDescription` dataclass and several `to_*` renderers. Every
caller (AgentEvolve prompt, compilagent backend, future optimizers) reads
from the same object, so adding a new dimension to the search space is a
one-place change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple


# The optimizer step rounds core dimensions to multiples of this factor when
# decoding decision vectors (see `_decode_hw` in `joint_arch_hw_problem.py`).
# It is the same value the wizard's HTML hint advertises to users.
CORE_DIM_GRANULARITY = 8


@dataclass(frozen=True)
class SearchSpaceDescription:
    """Declarative snapshot of the full joint search space for one run.

    Fields mirror the inputs `_create_optimizer` already collects from
    `arch_search` config and the deployment `platform_constraints`. All
    renderer methods are pure functions of these fields — no I/O, no global
    state — so the same instance can be passed to every optimizer at
    construction time.
    """

    search_mode: str  # "model" | "hardware" | "joint"

    # Architecture variables: ordered (key, allowed values) pairs. Same shape
    # `JointArchHwProblem.arch_options` consumes today.
    arch_options: Tuple[Tuple[str, Tuple[Any, ...]], ...] = ()

    # Hardware variables — same bounds JointArchHwProblem._decode_hw clamps to.
    num_core_types: int = 1
    core_axons_bounds: Tuple[int, int] = (64, 2048)
    core_neurons_bounds: Tuple[int, int] = (64, 2048)
    core_count_bounds: Tuple[int, int] = (50, 500)

    target_tq: int = 32
    weight_bits: int = 8

    # Optional reference example for each core (used by the AgentEvolve
    # example renderer). When empty, a synthetic default is generated.
    example_core_dims: Tuple[Tuple[int, int], ...] = ()
    example_core_count: int = 200

    @property
    def searches_model(self) -> bool:
        return self.search_mode in ("model", "joint")

    @property
    def searches_hw(self) -> bool:
        return self.search_mode in ("hardware", "joint")

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def from_arch_search(
        cls,
        *,
        search_mode: str,
        arch_options: Sequence[Tuple[str, Sequence[Any]]],
        arch_cfg: Dict[str, Any],
        target_tq: int,
        weight_bits: int = 8,
    ) -> "SearchSpaceDescription":
        """Build a description from the loose dicts the search step uses.

        ``arch_cfg`` is the ``arch_search`` block of the deployment config
        (see ``_parse_deployment_config``); we read the same keys
        ``_create_optimizer`` reads (`num_core_types`, `core_*_bounds`).
        """

        normalised_arch = tuple(
            (str(key), tuple(values)) for key, values in arch_options
        )
        num_core_types = int(arch_cfg.get("num_core_types", 1))
        ax_bounds = tuple(arch_cfg.get("core_axons_bounds", [64, 1024]))
        neu_bounds = tuple(arch_cfg.get("core_neurons_bounds", [64, 1024]))
        cnt_bounds = tuple(arch_cfg.get("core_count_bounds", [50, 500]))
        return cls(
            search_mode=str(search_mode),
            arch_options=normalised_arch,
            num_core_types=num_core_types,
            core_axons_bounds=(int(ax_bounds[0]), int(ax_bounds[1])),
            core_neurons_bounds=(int(neu_bounds[0]), int(neu_bounds[1])),
            core_count_bounds=(int(cnt_bounds[0]), int(cnt_bounds[1])),
            target_tq=int(target_tq),
            weight_bits=int(weight_bits),
        )

    # ------------------------------------------------------------------ #
    # Renderer 1: AgentEvolve LLM JSON-ish schema
    # ------------------------------------------------------------------ #

    def to_agent_evolve_schema(self) -> Dict[str, Any]:
        """Configuration schema description for AgentEvolve LLM prompts."""
        schema: Dict[str, Any] = {}
        if self.searches_model:
            schema["model_config"] = {
                key: f"one of {list(values)}" for key, values in self.arch_options
            }
        if self.searches_hw:
            schema["platform_constraints"] = {
                "cores": (
                    f"list of {self.num_core_types} objects, each with max_axons "
                    f"(int {self.core_axons_bounds[0]}-{self.core_axons_bounds[1]}, "
                    f"multiple of {CORE_DIM_GRANULARITY}), "
                    f"max_neurons (int {self.core_neurons_bounds[0]}-{self.core_neurons_bounds[1]}, "
                    f"multiple of {CORE_DIM_GRANULARITY}), "
                    f"count (int {self.core_count_bounds[0]}-{self.core_count_bounds[1]}). "
                    f"Different core types may have different sizes (heterogeneous)."
                ),
                "target_tq": f"{self.target_tq} (fixed)",
                "weight_bits": f"{self.weight_bits} (fixed)",
            }
        return schema

    # ------------------------------------------------------------------ #
    # Renderer 2: AgentEvolve example configuration
    # ------------------------------------------------------------------ #

    def to_agent_evolve_example(self) -> Dict[str, Any]:
        """Example configuration matching ``to_agent_evolve_schema``."""
        example: Dict[str, Any] = {}
        if self.searches_model:
            example["model_config"] = {
                key: values[len(values) // 2] for key, values in self.arch_options
            }
        if self.searches_hw:
            dims = self.example_core_dims or ((512, 512), (1024, 1024))
            cores: List[Dict[str, int]] = []
            for i in range(self.num_core_types):
                ax, neu = dims[i % len(dims)]
                cores.append(
                    {
                        "max_axons": int(ax),
                        "max_neurons": int(neu),
                        "count": int(self.example_core_count),
                    }
                )
            example["platform_constraints"] = {
                "cores": cores,
                "target_tq": self.target_tq,
                "weight_bits": self.weight_bits,
            }
        return example

    # ------------------------------------------------------------------ #
    # Renderer 3: AgentEvolve constraints free-text
    # ------------------------------------------------------------------ #

    def to_agent_evolve_constraints(self) -> str:
        """Constraint description for AgentEvolve LLM prompts."""
        parts: List[str] = ["\nCRITICAL CONSTRAINTS:\n"]
        if self.searches_model and self.arch_options:
            option_lines = "\n".join(
                f"   - {key}: must be one of {list(values)}"
                for key, values in self.arch_options
            )
            parts.append(f"1. MODEL CONFIGURATION OPTIONS:\n{option_lines}\n")
        if self.searches_hw:
            parts.append(
                "2. HETEROGENEOUS CORES:\n"
                "   Different core types may have different max_axons/max_neurons.\n"
                "   Softcores are tiled for the LARGEST core type and packed by the bin-packer.\n"
            )
            parts.append(
                f"3. Core dimensions: max_axons and max_neurons must be multiples "
                f"of {CORE_DIM_GRANULARITY},\n"
                f"   axons in [{self.core_axons_bounds[0]}, {self.core_axons_bounds[1]}],\n"
                f"   neurons in [{self.core_neurons_bounds[0]}, {self.core_neurons_bounds[1]}].\n"
                f"   Core count in [{self.core_count_bounds[0]}, {self.core_count_bounds[1]}].\n"
            )
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # Renderer 4: compilagent Lever tuple
    # ------------------------------------------------------------------ #

    # Per-core dimension axis kinds; the order matches ``_decode_hw``.
    HW_DIM_KINDS: Tuple[Tuple[str, str], ...] = (
        ("max_axons", "core_axons_bounds"),
        ("max_neurons", "core_neurons_bounds"),
        ("count", "core_count_bounds"),
    )

    # Compilagent's lever surface is intentionally advisory — the agent
    # picks any integer in [min, max] and the backend validates. JSON
    # bounds (`core_axons_bounds` etc., honoured by NSGA2/AgentEvolve)
    # are deliberately ignored here so the agent is free to right-size
    # the chip to the workload. The bounds below cover every realistic
    # neuromorphic crossbar; anything outside this range is almost
    # certainly a typo.
    COMPILAGENT_AXON_BOUNDS: Tuple[int, int] = (8, 8192)
    COMPILAGENT_NEURON_BOUNDS: Tuple[int, int] = (8, 8192)
    COMPILAGENT_COUNT_BOUNDS: Tuple[int, int] = (1, 4096)

    def to_compilagent_levers(self, *, workload_id: str, backend_id: str) -> Tuple[Any, ...]:
        """Render the search space as a tuple of compilagent ``Lever``s.

        Architecture options use ``EnumChoice`` (real categorical sets);
        hardware dimensions use ``IntFreeform`` — an open integer range
        with min/max bounds — so the agent freely chooses scale based on
        the workload's footprint rather than picking from a curated
        menu. Import is lazy because mimarsinan core does not depend on
        compilagent at module-load time.
        """
        from compilagent import (
            DerivationEvidence,
            EnumChoice,
            IntFreeform,
            Lever,
        )

        levers: List[Any] = []

        if self.searches_model:
            for key, values in self.arch_options:
                lever = Lever(
                    id=f"arch.{key}",
                    target_kind="arch",
                    target_selector=key,
                    range=EnumChoice(candidates=tuple(str(v) for v in values)),
                    default=values[len(values) // 2] if values else None,
                    description=(
                        f"Architecture choice `{key}`: enum over the values "
                        f"declared by the model builder's NAS schema."
                    ),
                    evidence=DerivationEvidence(
                        rule="mimarsinan.arch_options",
                        signal="JointArchHwProblem.arch_options",
                        citations=("search/search_space_description.py",),
                    ),
                    backend_id=backend_id,
                )
                levers.append(lever)

        if self.searches_hw:
            hw_bound_map = {
                "max_axons": self.COMPILAGENT_AXON_BOUNDS,
                "max_neurons": self.COMPILAGENT_NEURON_BOUNDS,
                "count": self.COMPILAGENT_COUNT_BOUNDS,
            }
            for core_idx in range(self.num_core_types):
                for dim_name, _bounds_attr in self.HW_DIM_KINDS:
                    lo, hi = hw_bound_map[dim_name]
                    step = (
                        CORE_DIM_GRANULARITY
                        if dim_name in ("max_axons", "max_neurons")
                        else 1
                    )
                    units = "" if dim_name == "count" else "wires"
                    default = max(lo, min(hi, 128 if dim_name != "count" else 16))
                    if step > 1:
                        default = max(step, (default // step) * step)
                    lever = Lever(
                        id=f"hw.core.{core_idx}.{dim_name}",
                        target_kind="hw.core",
                        target_selector=f"{core_idx}.{dim_name}",
                        range=IntFreeform(min=int(lo), max=int(hi), step=int(step), units=units),
                        default=int(default),
                        description=(
                            f"Hardware core type {core_idx} `{dim_name}` — "
                            f"clamped to [{lo}, {hi}] and "
                            + (
                                f"snapped to multiples of {CORE_DIM_GRANULARITY}."
                                if dim_name != "count"
                                else "passed straight through."
                            )
                        ),
                        evidence=DerivationEvidence(
                            rule="mimarsinan.compilagent.open_range",
                            signal=f"compilagent_{dim_name}_bounds",
                            citations=("search/search_space_description.py",),
                        ),
                        backend_id=backend_id,
                    )
                    levers.append(lever)

        return tuple(levers)

    @staticmethod
    def _derive_int_candidates(lo: int, hi: int, dim_name: str) -> Tuple[int, ...]:
        """Generate a derived sample of candidate values for one HW dimension.

        For ``max_axons``/``max_neurons`` we step by ``CORE_DIM_GRANULARITY``
        but cap the candidate count so a 64..2048 axis stays manageable.
        For ``count`` we sample a coarse arithmetic ladder. The agent is
        free to propose any integer in the bounds (validated by the
        backend); the candidates are a hint, not a constraint.
        """
        if hi <= lo:
            return (int(lo),)
        if dim_name in ("max_axons", "max_neurons"):
            step = CORE_DIM_GRANULARITY
            target_count = 8
            span = hi - lo
            stride = max(step, (span // target_count // step) * step or step)
            values: List[int] = []
            v = lo
            while v <= hi:
                values.append(int(v))
                v += stride
            if values[-1] != hi:
                values.append(int(hi))
            return tuple(sorted(set(values)))
        # count axis
        target_count = 6
        if hi - lo + 1 <= target_count:
            return tuple(range(int(lo), int(hi) + 1))
        stride = max(1, (hi - lo) // (target_count - 1))
        values = []
        v = lo
        while v <= hi:
            values.append(int(v))
            v += stride
        if values[-1] != hi:
            values.append(int(hi))
        return tuple(sorted(set(values)))


__all__ = ["SearchSpaceDescription", "CORE_DIM_GRANULARITY"]
