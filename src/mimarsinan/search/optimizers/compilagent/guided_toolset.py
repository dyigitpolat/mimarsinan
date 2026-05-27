"""GuidedToolset — augments selected compilagent tool results with guidance blocks."""

from __future__ import annotations

import functools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from compilagent import ToolDecl, Toolset

from .guidance_blocks import augment_inspect_workload, augment_run_result

if TYPE_CHECKING:
    from mimarsinan.search.results import ObjectiveSpec

    from .backend import MimarsinanLayoutBackend
    from .sink import MultiObjectiveSink


_INSPECT_WORKLOAD = "inspect_workload"
_RUN_CANDIDATE = "run_candidate"
_RUN_CANDIDATES = "run_candidates"


@dataclass
class _GuidanceState:
    sink: "MultiObjectiveSink"
    backend: "MimarsinanLayoutBackend"
    objectives: list["ObjectiveSpec"]
    baseline_injected: bool = False
    seen_metric_values: dict[str, set[float]] = field(default_factory=dict)


class GuidedToolset:
    """Drop-in replacement for ``Toolset`` that augments key tool results."""

    def __init__(
        self,
        base: Toolset,
        *,
        sink: "MultiObjectiveSink",
        backend: "MimarsinanLayoutBackend",
        objectives: Sequence["ObjectiveSpec"],
    ) -> None:
        self._base = base
        self._state = _GuidanceState(
            sink=sink, backend=backend, objectives=list(objectives),
        )
        self._wrapped_cache: dict[str, ToolDecl] = {}

    def by_name(self, name: str) -> ToolDecl:
        if name in self._wrapped_cache:
            return self._wrapped_cache[name]
        decl = self._base.by_name(name)
        if name in (_INSPECT_WORKLOAD, _RUN_CANDIDATE, _RUN_CANDIDATES):
            wrapped = _wrap_decl(decl, self._state, name)
            self._wrapped_cache[name] = wrapped
            return wrapped
        return decl

    def names(self) -> list[str]:
        return self._base.names()

    @property
    def tools(self) -> tuple[ToolDecl, ...]:
        out: list[ToolDecl] = []
        for decl in self._base.tools:
            if decl.name in (_INSPECT_WORKLOAD, _RUN_CANDIDATE, _RUN_CANDIDATES):
                out.append(self.by_name(decl.name))
            else:
                out.append(decl)
        return tuple(out)

    def read_only_subset(self) -> "GuidedToolset":
        inner = self._base.read_only_subset()
        clone = GuidedToolset(
            inner,
            sink=self._state.sink,
            backend=self._state.backend,
            objectives=self._state.objectives,
        )
        clone._state = self._state
        return clone

    def with_extra(self, extra: Iterable[ToolDecl]) -> "GuidedToolset":
        extended = self._base.with_extra(extra)
        clone = GuidedToolset(
            extended,
            sink=self._state.sink,
            backend=self._state.backend,
            objectives=self._state.objectives,
        )
        clone._state = self._state
        return clone


def _wrap_decl(decl: ToolDecl, state: _GuidanceState, kind: str) -> ToolDecl:
    original_handler = decl.handler

    @functools.wraps(original_handler)
    def _augmented_handler(*args: Any, **kwargs: Any) -> str:
        raw = original_handler(*args, **kwargs)
        if kind == _INSPECT_WORKLOAD:
            return augment_inspect_workload(raw, state)
        if kind in (_RUN_CANDIDATE, _RUN_CANDIDATES):
            return augment_run_result(raw, state)
        return raw

    return ToolDecl(
        name=decl.name,
        description=decl.description,
        args_schema=decl.args_schema,
        handler=_augmented_handler,
        read_only=decl.read_only,
        args_model=decl.args_model,
        returns_kind=decl.returns_kind,
        metadata=decl.metadata,
    )


__all__ = ["GuidedToolset"]
