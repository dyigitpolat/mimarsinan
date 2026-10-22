"""X3a instances: a seeded integer network with its semantics, window and inputs, as canonical JSON."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Tuple

from mimarsinan.common.refusal import ToolchainRefusal

SCHEMA = "x3a-instance/1"
INPUT = "input"
NEURAL = "neural"
HOST = "host"
#: Host operations an instance may name: the two decoded-value maps and the residual add of two equal halves.
HOST_OPS = ("identity", "relu", "sum")

STREAMED_LIF = "streamed_lif"
WINDOWED_LIF = "windowed_lif"
SYNCHRONIZED_LIF = "synchronized_lif"
CASCADED_TTFS = "cascaded_ttfs"
SYNCHRONIZED_TTFS = "synchronized_ttfs"
ANALYTICAL_TTFS = "analytical_ttfs"
VALUE = "value"
SEMANTICS = (STREAMED_LIF, WINDOWED_LIF, SYNCHRONIZED_LIF, CASCADED_TTFS, SYNCHRONIZED_TTFS,
             ANALYTICAL_TTFS, VALUE)
SPIKING_SEMANTICS = SEMANTICS[:-1]
#: The semantics the exactness theorem speaks about: integer accumulation on a cycle grid or in the value
#: domain. Analytical TTFS evaluates real-valued activations (``ttfs_segment``), so it is outside by name.
IN_SCOPE_SEMANTICS = tuple(s for s in SEMANTICS if s != ANALYTICAL_TTFS)


class InstanceError(ToolchainRefusal, ValueError):
    """An instance document that does not describe a well-formed integer network."""


@dataclass(frozen=True)
class Layer:
    """One node: a neural layer reads the concatenation of its sources in order; a host op computes on values."""

    name: str
    kind: str
    sources: Tuple[str, ...]
    width: int
    op: str = ""
    weights: Tuple[Tuple[int, ...], ...] = ()
    bias: Tuple[int, ...] = ()
    theta: int = 1


@dataclass(frozen=True)
class Network:
    input_width: int
    layers: Tuple[Layer, ...]
    outputs: Tuple[str, ...]

    def width_of(self, name: str) -> int:
        if name == INPUT:
            return self.input_width
        for layer in self.layers:
            if layer.name == name:
                return layer.width
        raise InstanceError(f"{name!r} names no layer")

    def fan_in(self, layer: Layer) -> int:
        return sum(self.width_of(s) for s in layer.sources)

    def neural_layers(self) -> Tuple[Layer, ...]:
        return tuple(layer for layer in self.layers if layer.kind == NEURAL)


@dataclass(frozen=True)
class Instance:
    """The function-bearing half of a deployment: network, semantics, window, bias mode and inputs.

    ``inputs`` are integers: under a spiking semantics a value ``k`` in ``[0, T]`` is a spike count, which the
    executors deliver exactly (the identity encoder is fed the midpoint rate ``(k + 1/2) / T`` and emits ``k``
    spikes, ``executors.input_tensor``); under the value semantics it is the input value itself.
    """

    seed: int
    semantics: str
    timesteps: int
    hardware_bias: bool
    network: Network
    inputs: Tuple[Tuple[int, ...], ...]
    motifs: Tuple[str, ...] = ()
    schema: str = field(default=SCHEMA)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        """Canonical text: sorted keys, no spaces, so equal instances are equal bytes."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "Instance":
        if doc.get("schema") != SCHEMA:
            raise InstanceError(f"schema {doc.get('schema')!r} is not {SCHEMA!r}")
        net = doc["network"]
        layers = tuple(
            Layer(name=str(raw["name"]), kind=str(raw["kind"]), sources=tuple(raw["sources"]),
                  width=int(raw["width"]), op=str(raw.get("op", "")),
                  weights=tuple(tuple(int(w) for w in row) for row in raw.get("weights", ())),
                  bias=tuple(int(b) for b in raw.get("bias", ())), theta=int(raw.get("theta", 1)))
            for raw in net["layers"])
        instance = cls(
            seed=int(doc["seed"]), semantics=str(doc["semantics"]), timesteps=int(doc["timesteps"]),
            hardware_bias=bool(doc["hardware_bias"]),
            network=Network(int(net["input_width"]), layers, tuple(net["outputs"])),
            inputs=tuple(tuple(int(v) for v in row) for row in doc["inputs"]),
            motifs=tuple(doc.get("motifs", ())))
        validate(instance)
        return instance

    @classmethod
    def from_json(cls, text: str) -> "Instance":
        return cls.from_dict(json.loads(text))


def validate(instance: Instance) -> None:
    """Refuse a document whose layers, shapes, sources or semantics are not those of an integer network."""
    if instance.semantics not in SEMANTICS:
        raise InstanceError(f"semantics {instance.semantics!r} is not one of {SEMANTICS}")
    if instance.timesteps < 1:
        raise InstanceError(f"timesteps {instance.timesteps} is not a positive window")
    net = instance.network
    seen = {INPUT}
    for layer in net.layers:
        if layer.name in seen:
            raise InstanceError(f"layer {layer.name!r} is defined twice")
        missing = [s for s in layer.sources if s not in seen]
        if missing or not layer.sources:
            raise InstanceError(f"layer {layer.name!r} reads {missing or 'nothing'} before it is defined")
        if layer.kind == NEURAL:
            fan_in = net.fan_in(layer)
            if len(layer.weights) != layer.width or any(len(r) != fan_in for r in layer.weights):
                raise InstanceError(f"layer {layer.name!r}: weights are not {layer.width} x {fan_in}")
            if len(layer.bias) != layer.width or layer.theta < 1:
                raise InstanceError(f"layer {layer.name!r}: bias or theta out of shape")
        elif layer.kind == HOST:
            if layer.op not in HOST_OPS:
                raise InstanceError(f"layer {layer.name!r}: host op {layer.op!r} is not one of {HOST_OPS}")
            fan_in = net.fan_in(layer)
            expected = fan_in // 2 if layer.op == "sum" else fan_in
            if layer.width != expected or (layer.op == "sum" and fan_in % 2):
                raise InstanceError(f"layer {layer.name!r}: width {layer.width} does not follow {layer.op}")
        else:
            raise InstanceError(f"layer {layer.name!r}: kind {layer.kind!r} is neither neural nor host")
        seen.add(layer.name)
    if not net.outputs or any(o not in seen or o == INPUT for o in net.outputs):
        raise InstanceError(f"outputs {net.outputs} name no defined layer")
    bound = instance.timesteps if instance.semantics != VALUE else None
    for row in instance.inputs:
        if len(row) != net.input_width:
            raise InstanceError(f"an input row has {len(row)} values, not {net.input_width}")
        if bound is not None and any(v < 0 or v > bound for v in row):
            raise InstanceError(f"a spiking input count lies outside [0, {bound}]")
