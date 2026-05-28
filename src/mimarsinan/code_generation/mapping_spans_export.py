"""Export runtime mapping span files and chip-config headers for RuntimeChip."""

from __future__ import annotations

from pathlib import Path

from mimarsinan.code_generation.cpp_chip_model import ChipModel
from mimarsinan.code_generation.cpp_chip_model_types import CodegenSpan


def _span_kind_char(span: CodegenSpan) -> str:
    if span.core_str == "off":
        return "o"
    if span.core_str == "in":
        return "i"
    if span.core_str == "on":
        return "a"
    return "c"


def _span_stored_core(span: CodegenSpan) -> int:
    if span.core_str in ("off", "in", "on"):
        return 0
    return int(span.core_str)


def write_mapping_spans_file(chip: ChipModel, path: str | Path) -> None:
    """Write chip_spans.txt consumed by nevresim mapping_loader.hpp."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    max_spans = chip._max_spans_per_core()  # noqa: SLF001
    lines: list[str] = [
        f"{chip.core_count} {max_spans} {chip.output_size}",
    ]
    for con in chip.connections:
        spans = con.get_spans()
        lines.append(str(len(spans)))
        for span in spans:
            lines.append(
                f"{_span_kind_char(span)} {_span_stored_core(span)} {span.start} {span.count}"
            )
    for out in chip.output_buffer:
        if out.is_off_:
            lines.append("-1 0")
        elif out.is_input_:
            lines.append(f"-2 {out.neuron_}")
        elif out.is_always_on_:
            lines.append("-3 0")
        else:
            lines.append(f"{out.core_} {out.neuron_}")
    path.write_text("\n".join(lines) + "\n")


def chip_config_header(chip: ChipModel, *, weight_cpp: str, threshold_cpp: str) -> str:
    """Emit generate_chip_config.hpp with dimensions only (no connectivity consteval)."""
    max_spans = chip._max_spans_per_core()  # noqa: SLF001
    return f"""#pragma once

#include "simulator/chip_utilities.hpp"
#include "simulator/chip/runtime_chip.hpp"

namespace nevresim
{{

template <typename ComputePolicy, typename WeightType, typename ThresholdType>
struct RuntimeChipConfig
{{
    using weight_t = WeightType;
    using threshold_t = ThresholdType;
    static constexpr std::size_t axon_count{{{chip.axon_count}}};
    static constexpr std::size_t neuron_count{{{chip.neuron_count}}};
    static constexpr std::size_t core_count{{{chip.core_count}}};
    static constexpr std::size_t input_size{{{chip.input_size}}};
    static constexpr std::size_t output_size{{{chip.output_size}}};
    static constexpr std::size_t max_spans_per_core{{{max_spans}}};
    static constexpr MembraneLeak<weight_t> leak{{{chip.leak}}};

    using cfg = ChipConfiguration<
        weight_t,
        threshold_t,
        axon_count,
        neuron_count,
        core_count,
        input_size,
        output_size,
        max_spans_per_core,
        leak
    >;

    using chip_t = RuntimeChip<cfg, ComputePolicy>;
}};

}} // namespace nevresim
"""
