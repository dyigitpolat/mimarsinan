from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class LayoutSoftCoreSpec:
    """
    Shape-only representation of a deployable soft core.
    """

    input_count: int
    output_count: int

    # Soft approximation: different threshold groups cannot share a hardcore.
    threshold_group_id: int = 0

    # Optional latency tag (kept for parity with real mapping constraints).
    latency_tag: Optional[int] = None

    name: Optional[str] = None

    def get_input_count(self) -> int:
        return int(self.input_count)

    def get_output_count(self) -> int:
        return int(self.output_count)

    @property
    def area(self) -> int:
        return int(self.input_count) * int(self.output_count)


@dataclass
class LayoutHardCoreType:
    max_axons: int
    max_neurons: int
    count: int


@dataclass
class LayoutHardCoreInstance:
    """
    Shape-only approximation of a hardware core instance.
    """

    axons_per_core: int
    neurons_per_core: int

    available_axons: int = field(init=False)
    available_neurons: int = field(init=False)

    threshold_group_id: Optional[int] = None
    latency_tag: Optional[int] = None

    unusable_space: int = 0
    used_area: int = 0

    def __post_init__(self):
        self.available_axons = int(self.axons_per_core)
        self.available_neurons = int(self.neurons_per_core)

    def get_input_count(self) -> int:
        return int(self.axons_per_core)

    def get_output_count(self) -> int:
        return int(self.neurons_per_core)

    @property
    def capacity(self) -> int:
        return int(self.axons_per_core) * int(self.neurons_per_core)

    def add_softcore(self, softcore: LayoutSoftCoreSpec) -> None:
        in_c = int(softcore.get_input_count())
        out_c = int(softcore.get_output_count())

        if in_c > self.available_axons or out_c > self.available_neurons:
            raise ValueError("Softcore does not fit into this hardcore instance")

        axon_offset = self.axons_per_core - self.available_axons
        neuron_offset = self.neurons_per_core - self.available_neurons

        # Track packing inefficiency in the same way as HardCore.add_softcore().
        self.unusable_space += (neuron_offset * in_c) + (axon_offset * out_c)

        self.available_axons -= in_c
        self.available_neurons -= out_c

        self.used_area += int(in_c) * int(out_c)

        if self.threshold_group_id is None:
            self.threshold_group_id = int(softcore.threshold_group_id)

        if self.latency_tag is None:
            self.latency_tag = softcore.latency_tag


@dataclass(frozen=True)
class LayoutPackingResult:
    feasible: bool
    cores_used: int

    total_capacity: int
    used_area: int

    unused_area_total: int
    avg_unused_area_per_core: float

    error: Optional[str] = None


