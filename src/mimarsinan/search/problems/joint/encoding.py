"""The joint problem's encoding: one real box, every discretization inside decode.

Dimension order is CONTRACT — arch, then hardware, then deployment options — so a
promoted option appends and never shifts a dimension an existing population means
something by.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
from mimarsinan.search.option_axes import decode_option_value
from mimarsinan.search.problems.joint.types import JointHostContract, clip_int
from mimarsinan.search.search_space_description import CORE_DIM_GRANULARITY


class JointEncodingMixin(JointHostContract):
    """``n_var``/``xl``/``xu``/``decode`` — the encoded box and what it means."""

    @property
    def _n_arch_vars(self) -> int:
        return len(self.arch_options) if self._searches_model else 0

    @property
    def _n_hw_vars(self) -> int:
        return (3 * int(self.num_core_types)) if self._searches_hw else 0

    @property
    def _n_option_vars(self) -> int:
        return len(self.option_axes)

    @property
    def n_var(self) -> int:
        return self._n_arch_vars + self._n_hw_vars + self._n_option_vars

    @property
    def xl(self) -> np.ndarray:
        xl: List[float] = []
        if self._searches_model:
            xl.extend([0.0] * len(self.arch_options))
        if self._searches_hw:
            for _ in range(int(self.num_core_types)):
                xl.extend([
                    float(self.core_axons_bounds[0]),
                    float(self.core_neurons_bounds[0]),
                    float(self.core_count_bounds[0]),
                ])
        xl.extend(axis.lower for axis in self.option_axes)
        return np.array(xl, dtype=float)

    @property
    def xu(self) -> np.ndarray:
        xu: List[float] = []
        if self._searches_model:
            xu.extend([float(len(opts) - 1) for _, opts in self.arch_options])
        if self._searches_hw:
            for _ in range(int(self.num_core_types)):
                xu.extend([
                    float(self.core_axons_bounds[1]),
                    float(self.core_neurons_bounds[1]),
                    float(self.core_count_bounds[1]),
                ])
        xu.extend(axis.upper for axis in self.option_axes)
        return np.array(xu, dtype=float)

    def _decode_arch(self, x: np.ndarray, offset: int) -> Dict[str, Any]:
        raw_arch: Dict[str, Any] = {}
        for i, (key, options) in enumerate(self.arch_options):
            idx = clip_int(x[offset + i], 0, len(options) - 1)
            raw_arch[key] = options[idx]
        return self.model_config_assembler(raw_arch)

    @staticmethod
    def _snap_core_dim(value: int) -> int:
        """Core dimensions live on the declared grid, never between its lines."""
        snapped = int(round(value / CORE_DIM_GRANULARITY)) * CORE_DIM_GRANULARITY
        return max(CORE_DIM_GRANULARITY, snapped)

    def _decode_options(self, x: np.ndarray, offset: int) -> Dict[str, Any]:
        """The searched deployment options: what this candidate DEPLOYS AS."""
        return {
            axis.key: decode_option_value(axis, x[offset + i])
            for i, axis in enumerate(self.option_axes)
        }

    def _decode_hw(
        self, x: np.ndarray, offset: int, options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """The searched dimensions, resolved into a chip by the deployment resolver.

        Every searched option rides the overlay too: the resolver is a pure function
        of the flat config and reads only what it consumes, so a candidate's chip is
        resolved from its OWN options (a searched ``schedule_policy`` really lands on
        the chip) while options the resolver ignores pass through harmlessly.
        """
        base = self.fixed_platform_constraints
        if not base:
            raise ValueError(
                "hardware search requires a platform_resolver: the candidate "
                "chip is the declared platform re-resolved with the searched "
                "core dimensions"
            )
        # A searched core type declares dimensions only; every other core
        # property is the declared platform's — including whether the chip can
        # deliver a bias on-core, which the resolver stamps onto each core.
        base_cores = base.get("cores") or []
        hardware_bias = resolve_platform_mapping_params(base_cores).hardware_bias

        core_types: List[Dict[str, Any]] = []
        idx = offset
        for _ in range(int(self.num_core_types)):
            ax = clip_int(x[idx], int(self.core_axons_bounds[0]), int(self.core_axons_bounds[1]))
            neu = clip_int(
                x[idx + 1], int(self.core_neurons_bounds[0]), int(self.core_neurons_bounds[1]),
            )
            count = clip_int(x[idx + 2], int(self.core_count_bounds[0]), int(self.core_count_bounds[1]))
            idx += 3
            core_types.append({
                "max_axons": self._snap_core_dim(ax),
                "max_neurons": self._snap_core_dim(neu),
                "count": count,
                "has_bias": hardware_bias,
            })

        return self.resolve_candidate_platform({
            **options,
            "cores": core_types,
            "target_tq": int(self.target_tq),
        })

    def seed_vectors(self) -> "List[np.ndarray]":
        """[R6] The DECLARED platform as a generation-1 seed.

        The declaration is the one point known feasible (the fixed-mode cell
        deploys on it), and random sampling over shape-constrained spaces can
        miss the feasible needle entirely — the ViT cell rejected 72/72
        offspring exactly that way. Model dims (joint mode) seed at their
        midpoints; option dims at their first choice; out-of-bounds declared
        dims clip, so the seed is then merely NEAR the declaration. Empty
        when the declared core-type count differs from the searched one — a
        seed that silently reshaped the chip would not be the declaration.
        """
        base = self.fixed_platform_constraints or {}
        cores = list(base.get("cores") or ())
        if not self._searches_hw or len(cores) != int(self.num_core_types):
            return []
        lo, hi = self.xl, self.xu
        x: "List[float]" = []
        if self._searches_model:
            x.extend(
                (float(lo[i]) + float(hi[i])) / 2.0
                for i in range(len(self.arch_options))
            )
        for core_type in cores:
            x.extend([
                float(core_type["max_axons"]),
                float(core_type["max_neurons"]),
                float(core_type["count"]),
            ])
        x.extend(0.0 for _ in self.option_axes)
        seed = np.clip(np.asarray(x, dtype=float), lo, hi)
        return [seed]

    def decode(self, x: np.ndarray) -> Dict[str, Any]:
        x = np.array(x, dtype=float).flatten()
        if x.shape[0] != self.n_var:
            raise ValueError(f"Expected x of length {self.n_var}, got {x.shape}")

        offset = 0

        if self._searches_model:
            model_config = self._decode_arch(x, offset)
            offset += self._n_arch_vars
        else:
            model_config = dict(self.fixed_model_config or {})

        options = self._decode_options(x, offset + self._n_hw_vars)

        if self._searches_hw:
            platform_constraints = self._decode_hw(x, offset, options)
        else:
            platform_constraints = self.resolve_candidate_platform(options)

        return {
            "model_config": model_config,
            "platform_constraints": platform_constraints,
            "deployment_options": options,
        }


__all__ = ["JointEncodingMixin"]
