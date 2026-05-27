"""Lava Loihi segment timing and profiling helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping

_LAVA_DTYPE = np.float64


@dataclass
class _StageTrace:
    name: str
    kind: str
    seconds: float
    cores: int = 0
    samples: int = 0


@dataclass
class _RunProfile:
    stages: List[_StageTrace] = field(default_factory=list)
    total_seconds: float = 0.0

    def log(self) -> None:
        print("=== LavaLoihiRunner profile ===")
        by_kind: Dict[str, float] = defaultdict(float)
        for st in self.stages:
            per = f", {st.cores} cores" if st.kind == "neural" else ""
            print(f"  [{st.kind:>7}] {st.seconds:7.2f}s  {st.name}{per}")
            by_kind[st.kind] += st.seconds
        print(f"  ---")
        for k, s in by_kind.items():
            print(f"  Σ {k:>7}: {s:7.2f}s")
        print(f"  Σ total : {self.total_seconds:7.2f}s")


@dataclass(frozen=True)
class _SegmentTiming:
    """Logical sample layout for one Lava neural-segment run."""

    T: int
    segment_latency: int
    sample_stride: int
    pad_head: int = 2
    pipeline_delay: int = 1
    tail: int = 3

    @classmethod
    def from_mapping(cls, mapping: HardCoreMapping, T: int) -> "_SegmentTiming":
        latency = int(ChipLatency(mapping).calculate())
        return cls(T=int(T), segment_latency=latency, sample_stride=int(T) + latency)

    @property
    def warmup_cycles(self) -> int:
        return self.sample_stride

    @property
    def logical_start(self) -> int:
        return self.pad_head + self.warmup_cycles + self.pipeline_delay

    def total_steps(self, n_samples: int) -> int:
        return self.pad_head + self.warmup_cycles + n_samples * self.sample_stride + self.tail

    def core_latency(self, core: HardCore) -> int:
        return max(int(core.latency) if core.latency is not None else 0, 0)

    def active_start(self, core: HardCore) -> int:
        return (self.logical_start + self.core_latency(core) + 1) % self.sample_stride

    def sample_start(self) -> int:
        return (self.logical_start + 1) % self.sample_stride

    def extract_logical(self, raw: np.ndarray, n_samples: int) -> np.ndarray:
        start = self.logical_start
        end = start + n_samples * self.sample_stride
        return np.asarray(raw[:, start:end], dtype=_LAVA_DTYPE).reshape(
            raw.shape[0], n_samples, self.sample_stride,
        )
