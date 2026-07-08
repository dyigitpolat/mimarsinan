"""Single source of truth for describing the joint NAS + HW search space, with `to_*` renderers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple


# Must match the rounding factor `_decode_hw` in search/problems/joint/problem.py applies.
CORE_DIM_GRANULARITY = 8

# Platform-search bounds defaults (deployment-config seeds; arch_search.* overrides).
DEFAULT_CORE_AXONS_BOUNDS = (64, 2048)
DEFAULT_CORE_NEURONS_BOUNDS = (64, 2048)
DEFAULT_CORE_COUNT_BOUNDS = (50, 500)


@dataclass(frozen=True)
class SearchSpaceDescription:
    """Declarative, pure-function snapshot of the full joint search space for one run."""

    search_mode: str

    arch_options: Tuple[Tuple[str, Tuple[Any, ...]], ...] = ()

    num_core_types: int = 1
    core_axons_bounds: Tuple[int, int] = DEFAULT_CORE_AXONS_BOUNDS
    core_neurons_bounds: Tuple[int, int] = DEFAULT_CORE_NEURONS_BOUNDS
    core_count_bounds: Tuple[int, int] = DEFAULT_CORE_COUNT_BOUNDS

    target_tq: int = 32
    weight_bits: int = 8

    example_core_dims: Tuple[Tuple[int, int], ...] = ()
    example_core_count: int = 200

    @property
    def searches_model(self) -> bool:
        return self.search_mode in ("model", "joint")

    @property
    def searches_hw(self) -> bool:
        return self.search_mode in ("hardware", "joint")

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
        """Build a description from the loose ``arch_search`` config dicts the search step uses."""

        normalised_arch = tuple(
            (str(key), tuple(values)) for key, values in arch_options
        )
        num_core_types = int(arch_cfg.get("num_core_types", 1))
        ax_bounds = tuple(arch_cfg.get("core_axons_bounds", DEFAULT_CORE_AXONS_BOUNDS))
        neu_bounds = tuple(arch_cfg.get("core_neurons_bounds", DEFAULT_CORE_NEURONS_BOUNDS))
        cnt_bounds = tuple(arch_cfg.get("core_count_bounds", DEFAULT_CORE_COUNT_BOUNDS))
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

    HW_DIM_KINDS: Tuple[Tuple[str, str], ...] = (
        ("max_axons", "core_axons_bounds"),
        ("max_neurons", "core_neurons_bounds"),
        ("count", "core_count_bounds"),
    )

    COMPILAGENT_AXON_BOUNDS: Tuple[int, int] = (8, 8192)
    COMPILAGENT_NEURON_BOUNDS: Tuple[int, int] = (8, 8192)
    COMPILAGENT_COUNT_BOUNDS: Tuple[int, int] = (1, 4096)

    def to_compilagent_levers(self, *, workload_id: str, backend_id: str) -> Tuple[Any, ...]:
        from mimarsinan.search.search_space_compilagent import to_compilagent_levers

        return to_compilagent_levers(
            self, workload_id=workload_id, backend_id=backend_id,
        )

    @staticmethod
    def _derive_int_candidates(lo: int, hi: int, dim_name: str) -> Tuple[int, ...]:
        from mimarsinan.search.search_space_compilagent import derive_int_candidates

        return derive_int_candidates(lo, hi, dim_name)


__all__ = ["SearchSpaceDescription", "CORE_DIM_GRANULARITY"]
