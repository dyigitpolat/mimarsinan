"""Nevresim compile/export/execute profiling records and helpers."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mimarsinan.chip_simulation.nevresim.profiling.mapping_metrics import (
    MappingConnectivityMetrics,
    metrics_from_chip_model,
    metrics_from_hardcore_mapping,
)
from mimarsinan.mapping.packing.softcore.hard_core_mapping import HardCoreMapping


@dataclass
class ArtifactSizes:
    generate_chip_hpp: int = 0
    main_cpp: int = 0
    chip_json: int = 0
    chip_weights_txt: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class NevresimCompileProfile:
    experiment: str = ""
    preset: str = ""
    core_count: int = 0
    axons_per_core: int = 0
    neurons_per_core: int = 0
    total_axon_slots: int = 0
    max_spans_per_core: int = 0
    total_spans: int = 0
    span_compression_ratio: float = 0.0
    cross_core_source_count: int = 0
    unique_cross_core_pairs: int = 0
    input_source_count: int = 0
    off_source_count: int = 0
    artifact_sizes: ArtifactSizes = field(default_factory=ArtifactSizes)
    export_s: float = 0.0
    codegen_s: float = 0.0
    compile_s: float = 0.0
    execute_s: float = 0.0
    compile_success: bool = False
    compiler: str = ""
    compiler_family: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping_metrics(
        cls,
        metrics: MappingConnectivityMetrics,
        *,
        experiment: str = "",
        preset: str = "",
    ) -> NevresimCompileProfile:
        row = cls(experiment=experiment, preset=preset)
        for key, val in metrics.as_dict().items():
            setattr(row, key, val)
        return row

    def apply_mapping_metrics(self, metrics: MappingConnectivityMetrics) -> None:
        for key, val in metrics.as_dict().items():
            setattr(self, key, val)

    def record_artifact_sizes(self, generated_dir: str | Path) -> None:
        base = Path(generated_dir)
        paths = {
            "generate_chip_hpp": base / "chip" / "generate_chip.hpp",
            "main_cpp": base / "main" / "main.cpp",
            "chip_json": base / "chip.json",
            "chip_weights_txt": base / "weights" / "chip_weights.txt",
        }
        config_header = base / "chip" / "generate_chip_config.hpp"
        spans_file = base / "chip" / "chip_spans.txt"
        for name, path in paths.items():
            setattr(self.artifact_sizes, name, path.stat().st_size if path.is_file() else 0)
        if self.artifact_sizes.generate_chip_hpp == 0 and config_header.is_file():
            self.artifact_sizes.generate_chip_hpp = config_header.stat().st_size
        if spans_file.is_file():
            self.extra["chip_spans_txt"] = spans_file.stat().st_size

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["artifact_sizes"] = self.artifact_sizes.as_dict()
        return d

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2)


class ProfileTimer:
    def __init__(self) -> None:
        self._start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> ProfileTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed = time.perf_counter() - self._start


def metrics_for_mapping(hcm: HardCoreMapping) -> MappingConnectivityMetrics:
    return metrics_from_hardcore_mapping(hcm)


def metrics_for_chip(chip) -> MappingConnectivityMetrics:
    return metrics_from_chip_model(chip)


def write_profile_rows(rows: list[NevresimCompileProfile], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [row.as_dict() for row in rows]
    if path.suffix == ".json":
        path.write_text(json.dumps(payload, indent=2))
        return
    if path.suffix == ".csv":
        import csv

        if not payload:
            path.write_text("")
            return
        keys = list(payload[0].keys())
        flat_keys: list[str] = []
        for k in keys:
            if k == "artifact_sizes":
                for sub in payload[0]["artifact_sizes"]:
                    flat_keys.append(f"artifact_{sub}")
            elif k == "extra":
                continue
            else:
                flat_keys.append(k)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=flat_keys)
            writer.writeheader()
            for row in payload:
                flat: dict[str, Any] = {}
                for k, v in row.items():
                    if k == "artifact_sizes":
                        for sub, sz in v.items():
                            flat[f"artifact_{sub}"] = sz
                    elif k != "extra":
                        flat[k] = v
                writer.writerow(flat)
        return
    path.write_text(json.dumps(payload, indent=2))


def correlate_compile_vs_metric(
    rows: list[NevresimCompileProfile],
    metric: str = "total_axon_slots",
) -> float | None:
    """Pearson r between compile_s and *metric*; None when insufficient data."""
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        if not row.compile_success or row.compile_s <= 0:
            continue
        if metric == "header_bytes":
            val = float(row.artifact_sizes.generate_chip_hpp)
        else:
            val = float(getattr(row, metric))
        xs.append(val)
        ys.append(row.compile_s)
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    den_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)
