"""Compile cache for nevresim segment binaries keyed by mapping + policy hash."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.packing.softcore.hard_core_mapping import HardCoreMapping


def _serialize_spike_source(src: SpikeSource) -> tuple:
    return (
        int(src.core_),
        int(src.neuron_),
        bool(src.is_input_),
        bool(src.is_off_),
        bool(src.is_always_on_),
    )


def mapping_connectivity_hash(hcm: HardCoreMapping) -> str:
    """Hash connectivity topology (axon sources + output taps), not weights."""
    parts: list[Any] = [
        int(hcm.axons_per_core),
        int(hcm.neurons_per_core),
        len(hcm.cores),
    ]
    for core in hcm.cores:
        parts.append(tuple(_serialize_spike_source(s) for s in core.axon_sources))
    parts.append(tuple(_serialize_spike_source(s) for s in hcm.output_sources))
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def policy_hash(
    *,
    spiking_mode: str,
    spike_generation_mode: str,
    firing_mode: str,
    thresholding_mode: str,
    weight_type_name: str,
    threshold_type_name: str,
    simulation_length: int,
    latency: int,
    connectivity_mode: str = "compile_time",
) -> str:
    payload = json.dumps(
        {
            "spiking_mode": spiking_mode,
            "spike_generation_mode": spike_generation_mode,
            "firing_mode": firing_mode,
            "thresholding_mode": thresholding_mode,
            "weight_type": weight_type_name,
            "threshold_type": threshold_type_name,
            "simulation_length": int(simulation_length),
            "latency": int(latency),
            "connectivity_mode": connectivity_mode,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def cache_key(mapping_hash: str, policy_hash_value: str) -> str:
    return f"{mapping_hash[:16]}_{policy_hash_value[:16]}"


class NevresimCompileCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _entry_dir(self, key: str) -> Path:
        return self.cache_dir / key

    def get_binary(self, key: str) -> Path | None:
        candidate = self._entry_dir(key) / "simulator"
        return candidate if candidate.is_file() else None

    def store_binary(self, key: str, binary_path: str | Path, metadata: dict | None = None) -> Path:
        dest_dir = self._entry_dir(key)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "simulator"
        shutil.copy2(binary_path, dest)
        if metadata is not None:
            (dest_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        return dest

    def invalidate(self, key: str | None = None) -> None:
        if key is None:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            return
        entry = self._entry_dir(key)
        if entry.exists():
            shutil.rmtree(entry)
