#!/usr/bin/env python3
"""Phase 3 wave 4: split oversized mapping/viz modules (no shims)."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "mimarsinan"


def git_show(rel: str) -> list[str]:
    return subprocess.check_output(
        ["git", "show", f"HEAD:src/mimarsinan/{rel}"],
        cwd=ROOT,
        text=True,
    ).splitlines(keepends=True)


def read(rel: str) -> list[str]:
    p = SRC / rel
    if p.exists():
        return p.read_text(encoding="utf-8").splitlines(keepends=True)
    return git_show(rel)


def write(rel: str, header: str, lines: list[str]) -> None:
    p = SRC / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(header + "".join(lines), encoding="utf-8")


def slice_lines(all_lines: list[str], start: int, end: int) -> list[str]:
    return all_lines[start - 1 : end]


def replace_file(rel: str, header: str, parts: list[tuple[int, int]]) -> None:
    lines = read(rel)
    body: list[str] = []
    for start, end in parts:
        body.extend(slice_lines(lines, start, end))
    write(rel, header, body)


def split_to_new(src_rel: str, dests: list[tuple[str, str, int, int]]) -> None:
    lines = read(src_rel)
    for dest_rel, header, start, end in dests:
        write(dest_rel, header, slice_lines(lines, start, end))
    (SRC / src_rel).unlink(missing_ok=True)


IMPORTS = textwrap.dedent(
    """\

    """
)


def main() -> None:
    # layout_verification_stats
    split_to_new(
        "mapping/verification/layout_verification_stats.py",
        [
            (
                "mapping/verification/layout_verification_types.py",
                "from __future__ import annotations\nfrom dataclasses import asdict, dataclass\nfrom typing import Any, Dict\n\n",
                22,
                96,
            ),
            (
                "mapping/verification/layout_verification_helpers.py",
                "from __future__ import annotations\nfrom typing import Dict, List, Optional, Sequence\n"
                "from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec\n\n",
                98,
                143,
            ),
            (
                "mapping/verification/layout_verification_packing.py",
                textwrap.dedent(
                    """\
                    from __future__ import annotations
                    from typing import Optional, Sequence
                    from mimarsinan.mapping.layout.layout_packer import pack_layout
                    from mimarsinan.mapping.layout.layout_types import (
                        LayoutHardCoreType, LayoutPackingResult, LayoutSoftCoreSpec,
                    )
                    from mimarsinan.mapping.verification.layout_verification_helpers import (
                        _latency_stats, _pct, _safe_median,
                    )
                    from mimarsinan.mapping.verification.layout_verification_types import LayoutVerificationStats
                    """
                ),
                146,
                364,
            ),
            (
                "mapping/verification/layout_verification_scheduling.py",
                textwrap.dedent(
                    """\
                    from __future__ import annotations
                    from dataclasses import replace
                    from typing import Dict, List, Optional, Sequence, Tuple
                    from mimarsinan.mapping.layout.layout_packer import pack_layout
                    from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
                    from mimarsinan.mapping.verification.layout_verification_packing import (
                        _empty_stats, _stats_from_packing,
                    )
                    from mimarsinan.mapping.verification.layout_verification_types import LayoutVerificationStats
                    """
                ),
                367,
                501,
            ),
            (
                "mapping/verification/layout_verification_hybrid.py",
                "from __future__ import annotations\nfrom typing import Any, Dict, Optional\n\n",
                504,
                624,
            ),
        ],
    )

    # hw_config_suggester
    hw_lines = read("mapping/verification/hw_config_suggester.py")
    write(
        "mapping/verification/hw_suggestion_types.py",
        "from __future__ import annotations\nfrom dataclasses import dataclass\nfrom typing import Any, Dict, List\n\n",
        slice_lines(hw_lines, 16, 25),
    )
    write(
        "mapping/verification/hw_suggestion_helpers.py",
        textwrap.dedent(
            """\
            from __future__ import annotations
            import math
            from typing import Sequence
            from mimarsinan.mapping.layout.layout_packer import pack_layout
            from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
            """
        ),
        slice_lines(hw_lines, 27, 246),
    )
    write(
        "mapping/verification/hw_config_suggester.py",
        textwrap.dedent(
            """\
            from __future__ import annotations
            from typing import Any, Dict, List, Sequence
            from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
            from mimarsinan.mapping.verification.suggester.hw_suggestion_helpers import (
                _count_cores_needed_two_types, _count_per_type_usage, _dimension_bounds,
                _make_two_core_types, _median, _min_hw_coverage, _next_multiple,
                _occupancy_ok, _pack_with_two_types,
            )
            from mimarsinan.mapping.verification.suggester.hw_suggestion_types import HardwareSuggestion
            """
        ),
        slice_lines(hw_lines, 248, 567),
    )

    # mapping_verifier
    split_to_new(
        "mapping/verification/mapping_verifier.py",
        [
            (
                "mapping/verification/mapping_verifier_types.py",
                "from __future__ import annotations\nfrom dataclasses import dataclass, field\nfrom typing import Any, Dict, List, Optional\n\n",
                36,
                58,
            ),
            (
                "mapping/verification/mapping_verifier_soft.py",
                textwrap.dedent(
                    """\
                    from __future__ import annotations
                    from typing import Any, Dict, List, Optional, Tuple
                    import numpy as np
                    from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
                    from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
                    from mimarsinan.mapping.verification.verifier.mapping_verifier_types import MappingVerificationResult
                    """
                ),
                60,
                169,
            ),
            (
                "mapping/verification/mapping_verifier_hw.py",
                textwrap.dedent(
                    """\
                    from __future__ import annotations
                    from typing import Any, Dict, List, Optional, Tuple
                    from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
                    from mimarsinan.mapping.layout.layout_packer import pack_layout
                    from mimarsinan.mapping.verification.layout_verification_packing import build_stats_from_packing_result
                    from mimarsinan.mapping.verification.layout_verification_scheduling import compute_schedule_sync_count
                    from mimarsinan.mapping.verification.verifier.mapping_verifier_types import MappingVerificationResult
                    """
                ),
                171,
                422,
            ),
        ],
    )

    # ir_pruning
    irp = read("mapping/pruning/ir_pruning.py")
    write(
        "mapping/pruning/ir_pruning_masks.py",
        textwrap.dedent(
            """\
            from __future__ import annotations
            from typing import Dict, List, Sequence, Set, Tuple
            import numpy as np
            from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
            """
        ),
        slice_lines(irp, 46, 201),
    )
    write(
        "mapping/pruning/ir_pruning_helpers.py",
        textwrap.dedent(
            """\
            from __future__ import annotations
            from typing import Dict, List, Sequence, Set, Tuple
            import numpy as np
            from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
            from mimarsinan.mapping.pruning.ir_pruning_analysis import compute_graph_io_exemption
            from mimarsinan.mapping.pruning.graph.pruning_graph_types import GlobalPruningResult
            """
        ),
        slice_lines(irp, 292, 488),
    )
    write(
        "mapping/pruning/ir_pruning_compact.py",
        textwrap.dedent(
            """\
            from __future__ import annotations
            from typing import Dict, Set, Tuple
            import numpy as np
            from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
            from mimarsinan.mapping.pruning.pruning_apply import compact_hardware_bias_columns
            from mimarsinan.mapping.pruning.graph.pruning_graph_types import GlobalPruningResult
            """
        ),
        slice_lines(irp, 490, 621),
    )
    write(
        "mapping/pruning/ir_pruning_core.py",
        textwrap.dedent(
            """\
            from __future__ import annotations
            from typing import Dict, List, Sequence, Set, Tuple
            import numpy as np
            from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
            from mimarsinan.mapping.pruning.ir_liveness import NodeLiveness, compute_liveness
            from mimarsinan.mapping.pruning.graph.pruning_graph_core import compute_global_pruned_sets
            from mimarsinan.mapping.pruning.graph.pruning_graph_types import GlobalPruningResult
            from mimarsinan.mapping.pruning.ir_pruning_helpers import (
                _attach_pre_compaction_metadata,
                _collect_exemptions,
                _collect_initial_seeds,
                _force_dead_nodes_fully_pruned,
                _log_value_based_summary,
                _rewire_sources,
            )
            from mimarsinan.mapping.pruning.ir_pruning_compact import (
                _attach_bank_metadata,
                _compact_node,
                _reset_post_compaction_masks,
                _validate_outputs_remain,
            )
            """
        ),
        slice_lines(irp, 203, 289),
    )
    (SRC / "mapping/pruning/ir_pruning.py").unlink(missing_ok=True)

    # pruning_graph_propagation
    pgp = read("mapping/pruning/pruning_graph_propagation.py")
    write(
        "mapping/pruning/pruning_graph_types.py",
        "from __future__ import annotations\nfrom dataclasses import dataclass, field\nfrom typing import AbstractSet, Dict, Set, Tuple\n\n",
        slice_lines(pgp, 50, 58),
    )
    write(
        "mapping/pruning/pruning_graph_refresh.py",
        textwrap.dedent(
            """\
            from __future__ import annotations
            from typing import Mapping, Set, Tuple
            import numpy as np
            from mimarsinan.mapping.ir import NeuralCore, WeightBank
            from mimarsinan.mapping.pruning.graph.pruning_propagation import compute_propagated_pruned_rows_cols
            """
        ),
        slice_lines(pgp, 311, 543),
    )
    write(
        "mapping/pruning/pruning_graph_core.py",
        textwrap.dedent(
            """\
            from __future__ import annotations
            from collections import defaultdict
            from typing import AbstractSet, Dict, Mapping, Set, Tuple
            import numpy as np
            from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore, WeightBank
            from mimarsinan.mapping.pruning.graph.pruning_graph_types import GlobalPruningResult
            from mimarsinan.mapping.pruning.graph.pruning_propagation import compute_propagated_pruned_rows_cols
            from mimarsinan.mapping.pruning.graph.pruning_graph_refresh import (
                _refresh_bank_pruning,
                _refresh_node_pruning,
                _resolve_node_matrix,
                _seed_off_source_axons,
                _seed_value_based,
            )
            """
        ),
        slice_lines(pgp, 60, 310),
    )
    (SRC / "mapping/pruning/pruning_graph_propagation.py").unlink(missing_ok=True)

    # ir_mapping
    irm = read("mapping/ir_mapping.py")
    header = "".join(slice_lines(irm, 1, 29))
    write("mapping/ir_mapping_class.py", header, slice_lines(irm, 31, 368))
    write("mapping/map_model_to_ir.py", header, slice_lines(irm, 369, 386))
    (SRC / "mapping/ir_mapping.py").unlink(missing_ok=True)

    # conv mappers
    conv = read("mapping/mappers/conv.py")
    conv_header = "".join(slice_lines(conv, 1, 15))
    write("mapping/mappers/conv_helpers.py", conv_header, slice_lines(conv, 16, 24))
    write(
        "mapping/mappers/conv2d_mapper.py",
        conv_header + "from mimarsinan.mapping.mappers.base import Mapper\n"
        "from mimarsinan.mapping.mappers.conv2d_mapper_helpers import _chunk_sizes\n\n",
        slice_lines(conv, 26, 245),
    )
    write(
        "mapping/mappers/conv1d_mapper.py",
        conv_header + "from mimarsinan.mapping.mappers.base import Mapper\n"
        "from mimarsinan.mapping.mappers.conv2d_mapper_helpers import _chunk_sizes\n\n",
        slice_lines(conv, 247, 420),
    )
    (SRC / "mapping/mappers/conv.py").unlink(missing_ok=True)

    # perceptron mappers
    perc = read("mapping/mappers/perceptron.py")
    perc_header = "".join(slice_lines(perc, 1, 16))
    write(
        "mapping/mappers/perceptron_mapper.py",
        perc_header + "from mimarsinan.mapping.mappers.base import Mapper\n\n",
        slice_lines(perc, 17, 97),
    )
    write(
        "mapping/mappers/compute_op_mapper.py",
        perc_header + "from mimarsinan.mapping.mappers.base import Mapper\n\n",
        slice_lines(perc, 99, 345),
    )
    write(
        "mapping/mappers/module_mapper.py",
        perc_header + "from mimarsinan.mapping.mappers.base import Mapper\n\n",
        slice_lines(perc, 347, 358),
    )
    (SRC / "mapping/mappers/perceptron.py").unlink(missing_ok=True)

    # layout_source_view
    lsv = read("mapping/layout/layout_source_view.py")
    lsv_header = "".join(slice_lines(lsv, 1, 32))
    write("mapping/layout/layout_source_view.py", lsv_header, slice_lines(lsv, 33, 245))
    write(
        "mapping/layout/layout_source_view_ops.py",
        "from __future__ import annotations\n\n"
        "from mimarsinan.mapping.layout.layout_source_view import LayoutSourceView\n\n",
        slice_lines(lsv, 247, 354),
    )

    # layout_ir_mapping_fc: extract psum helper module (keeps class intact)
    fc = read("mapping/layout/layout_ir_mapping_fc.py")
    fc_header = "".join(slice_lines(fc, 1, 13))
    write(
        "mapping/layout/layout_ir_mapping_fc_psum.py",
        fc_header + "import numpy as np\n\n",
        slice_lines(fc, 195, 309),
    )
    # Replace method body with delegate in main file (handled post-script)
    write("mapping/layout/layout_ir_mapping_fc.py", fc_header, slice_lines(fc, 14, 194))

    # schedule_partitioner
    sp = read("mapping/support/schedule_partitioner.py")
    sp_header = "".join(slice_lines(sp, 1, 35))
    write("mapping/support/schedule_budget.py", sp_header, slice_lines(sp, 36, 71))
    write(
        "mapping/support/schedule_split.py",
        sp_header + "from mimarsinan.mapping.support.schedule.schedule_budget import _coalescing_bundles\n\n",
        slice_lines(sp, 73, 186),
    )
    write(
        "mapping/support/schedule_partitioner.py",
        sp_header
        + "from mimarsinan.mapping.support.schedule.schedule_budget import effective_core_budget\n"
        + "from mimarsinan.mapping.support.schedule.schedule_split import split_softcores_by_capacity\n\n",
        slice_lines(sp, 188, 305),
    )

    # hybrid_segment
    hs = read("mapping/packing/hybrid_segment.py")
    hs_header = "".join(slice_lines(hs, 1, 12))
    write("mapping/packing/hybrid_segment_helpers.py", hs_header, slice_lines(hs, 13, 155))
    write(
        "mapping/packing/hybrid_segment.py",
        hs_header
        + "from mimarsinan.mapping.packing.hybrid_segment_helpers import (\n"
        + "    _apply_reindex_to_ir_sources,\n"
        + "    _check_no_split_coalescing_groups,\n"
        + "    _make_available_hardware_cores,\n"
        + "    _reindex_nodes,\n"
        + "    _remap_external_sources_to_segment_inputs,\n"
        + ")\n\n",
        slice_lines(hs, 157, 335),
    )

    # hybrid_build
    hb = read("mapping/packing/hybrid_build.py")
    hb_header = "".join(slice_lines(hb, 1, 24))
    write("mapping/packing/hybrid_build_pool.py", hb_header, slice_lines(hb, 25, 222))
    write("mapping/packing/hybrid_build_scheduled.py", hb_header, slice_lines(hb, 224, 331))
    (SRC / "mapping/packing/hybrid_build.py").unlink(missing_ok=True)

    # graphviz ir
    irv = read("visualization/graphviz/ir.py")
    irv_header = "".join(slice_lines(irv, 1, 32))
    write("visualization/graphviz/ir_summary.py", irv_header, slice_lines(irv, 33, 256))
    write("visualization/graphviz/ir_dot.py", irv_header, slice_lines(irv, 258, 387))
    (SRC / "visualization/graphviz/ir.py").unlink(missing_ok=True)

    # graphviz hybrid
    hyv = read("visualization/graphviz/hybrid.py")
    hyv_header = "".join(slice_lines(hyv, 1, 25))
    write("visualization/graphviz/hybrid_types.py", hyv_header, slice_lines(hyv, 26, 30))
    write(
        "visualization/graphviz/hybrid_dots.py",
        hyv_header + "from mimarsinan.visualization.graphviz.hybrid_segment_dots_types import HybridVizArtifacts\n\n",
        slice_lines(hyv, 32, 326),
    )
    (SRC / "visualization/graphviz/hybrid.py").unlink(missing_ok=True)

    # softcore_flowchart
    scf = read("visualization/softcore_flowchart.py")
    scf_header = "".join(slice_lines(scf, 1, 28))
    write("visualization/softcore_flowchart_estimate.py", scf_header, slice_lines(scf, 29, 158))
    write(
        "visualization/softcore_flowchart_dot.py",
        scf_header
        + "from mimarsinan.visualization.softcore_flowchart_dot_estimate import HWEstimate, estimate_fc_cores\n\n",
        slice_lines(scf, 160, 350),
    )
    (SRC / "visualization/softcore_flowchart.py").unlink(missing_ok=True)

    print("All wave 4 file splits written (ir_graph_snapshot handled separately).")


if __name__ == "__main__":
    main()
