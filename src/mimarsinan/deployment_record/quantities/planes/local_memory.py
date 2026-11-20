"""[U-ENERGY class 3] The core-local memory census on both planes, reads and writes apart.

Each evaluated core reads every used row's input into its array and writes every used column's output once per
evaluation step: at the program's value width (``activation_bits``) under the value domain, one bit per element per
step on a spiking core. The outputs a partial-sum core writes at its accumulator width are counted at the value width
(the census's named residual, stated in the quantity's text)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from mimarsinan.deployment_record.quantities.spec import QuantityValue

READ, WRITE = "local_mem_read_bytes", "local_mem_write_bytes"
NO_WIDTH = "activation_bits undeclared: the local memory traffic has no width"


def _bytes(elements: float, value_bits: Optional[int], spiking_steps: Optional[float]) -> Optional[float]:
    """Elements per evaluation as bytes per inference: at the value width, or one bit per element per step."""
    if value_bits is not None:
        return elements * int(value_bits) / 8.0
    return None if spiking_steps is None else elements * spiking_steps / 8.0


def local_memory(values: Dict[str, QuantityValue], *, reads: Optional[float], writes: Optional[float],
                 value_domain: bool, value_bits: Optional[int], steps: Optional[float]) -> Dict[str, str]:
    """Put both censuses into ``values`` from the elements read and written per evaluation; returns the censuses
    withheld by name (a value-domain program without its value width)."""
    held: Dict[str, str] = {}
    for key, elements in ((READ, reads), (WRITE, writes)):
        if elements is None:
            continue
        if value_domain and value_bits is None:
            held[key] = NO_WIDTH
            continue
        sized = _bytes(elements, value_bits if value_domain else None, None if value_domain else steps)
        if sized is not None:
            values[key] = QuantityValue(float(sized), "static")
    return held


def candidate_local_memory(values: Dict[str, QuantityValue], context: Any) -> Dict[str, str]:
    """The candidate plane: outputs per evaluation (``neurons_used``) and the local memories' reads and writes."""
    if context.output_neurons is not None:
        values["neurons_used"] = QuantityValue(float(context.output_neurons), "static")
    return local_memory(values, reads=context.local_mem_read_elements, writes=context.output_neurons,
                        value_domain=context.value_rule is not None, value_bits=context.value_bits,
                        steps=context.timesteps)


def record_local_memory(values: Dict[str, QuantityValue], platform: Mapping[str, Any],
                        value_domain: bool) -> Dict[str, str]:
    """The record plane: the evaluated cores' used rows and columns (dark cores out) over every evaluation step."""
    if "timesteps" not in values:
        return {}
    steps = values["timesteps"].value
    per = {key: None if census not in values else values[census].value * steps
           for key, census in ((READ, "axons_used"), (WRITE, "neurons_used"))}
    bits = platform.get("activation_bits")
    return local_memory(values, reads=per[READ], writes=per[WRITE], value_domain=value_domain,
                        value_bits=None if bits is None else int(bits), steps=1.0)
