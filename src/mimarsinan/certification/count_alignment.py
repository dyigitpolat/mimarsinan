"""[calculus §17/PR44] perceptron-aligned count capture via IR provenance."""

from __future__ import annotations

import torch

from mimarsinan.certification.spike_certificate import certify_spike_counts
from mimarsinan.spiking.segment_forward import (
    LifSegmentPolicy,
    SegmentForwardDriver,
)


def nf_perceptron_counts(repr_, T: int, batch: torch.Tensor) -> dict[int, torch.Tensor]:
    """Reference counts ``{perceptron_index: (B, n)}`` from the NF synchronized
    walk: decoded value / theta * T, flattened to canonical (column, channel)
    order — column = token/position — matching provenance placement."""
    driver = SegmentForwardDriver(repr_, T, LifSegmentPolicy(synchronized=True))
    rec: dict = {}
    with torch.no_grad():
        driver(batch, node_value_recorder=rec)
    out: dict[int, torch.Tensor] = {}
    for k, p in enumerate(repr_.get_perceptrons()):
        v = rec.get(id(p))
        if v is None:
            continue
        if v.dim() == 4:
            v = v.permute(0, 2, 3, 1)
        theta = float(torch.as_tensor(p.activation_scale).float().mean())
        counts = v.reshape(v.shape[0], -1) / max(theta, 1e-12) * T
        out[k] = counts.round().cpu()
    return out


class PerceptronCountAssembler:
    """Maps backend stage-output counts onto perceptron channel vectors.

    Identity comes from IR provenance (``perceptron_index``,
    ``perceptron_output_column``, ``perceptron_output_slice``); flat placement
    is ``column * channels + slice``, so packing splits (col/cap tiles,
    reindexing) need no name parsing. Nodes without provenance are internal
    and skipped; a perceptron with uncaptured nodes, coverage gaps, or
    overlapping writes is dropped and reported, never silently compared."""

    def __init__(self, ir_graph):
        self._meta: dict[int, tuple[int, int, tuple[int, int] | None]] = {}
        for n in getattr(ir_graph, "nodes", ()):
            pi = getattr(n, "perceptron_index", None)
            if pi is None:
                continue
            col = getattr(n, "perceptron_output_column", None)
            sl = getattr(n, "perceptron_output_slice", None)
            self._meta[int(n.id)] = (
                int(pi), 0 if col is None else int(col),
                None if sl is None else (int(sl[0]), int(sl[1])),
            )
        if not self._meta:
            raise ValueError(
                "IR graph carries no perceptron provenance; call "
                "assign_perceptron_indices() on the mapper repr before mapping"
            )
        self._node_counts: dict[int, torch.Tensor] = {}
        self.last_report: str = ""

    def capture_stage(self, output_map, counts: torch.Tensor) -> None:
        """Record one neural stage's raw output counts (output-map order)."""
        for sl in output_map:
            piece = counts[:, sl.offset : sl.offset + sl.size]
            self._node_counts[int(sl.node_id)] = piece.detach().cpu()

    def assemble(self) -> dict[int, torch.Tensor]:
        """``{perceptron_index: (B, n) counts}`` for fully covered perceptrons."""
        per_pi: dict[int, list[int]] = {}
        for nid, (pi, _c, _s) in self._meta.items():
            per_pi.setdefault(pi, []).append(nid)
        out: dict[int, torch.Tensor] = {}
        dropped: list[str] = []
        for pi, nids in sorted(per_pi.items()):
            entries = []
            missing = 0
            for nid in nids:
                t = self._node_counts.get(nid)
                if t is None:
                    missing += 1
                    continue
                _pi, col, sl = self._meta[nid]
                a, b = sl if sl is not None else (0, t.shape[1])
                if b - a != t.shape[1]:
                    raise ValueError(
                        f"node {nid}: captured width {t.shape[1]} != "
                        f"provenance slice width {b - a}"
                    )
                entries.append((col, a, b, t))
            if missing:
                dropped.append(f"p{pi}:{missing}/{len(nids)} nodes uncaptured")
                continue
            channels = max(b for _c, _a, b, _t in entries)
            ncols = max(c for c, _a, _b, _t in entries) + 1
            batch = entries[0][3].shape[0]
            buf = torch.zeros(batch, ncols * channels, dtype=entries[0][3].dtype)
            writes = torch.zeros(ncols * channels, dtype=torch.int32)
            for col, a, b, t in entries:
                buf[:, col * channels + a : col * channels + b] = t
                writes[col * channels + a : col * channels + b] += 1
            if int(writes.min()) != 1 or int(writes.max()) != 1:
                dropped.append(
                    f"p{pi}:coverage gap/overlap "
                    f"(min={int(writes.min())} max={int(writes.max())})"
                )
                continue
            out[pi] = buf.round()
        self.last_report = "; ".join(dropped) if dropped else "all covered"
        return out


def certify_flow_counts(
    repr_, ir_graph, flow, samples: torch.Tensor, *, backend: str
):
    """One-call certificate: NF oracle vs a hybrid flow's captured stage counts.

    Runs the flow under the synchronized LIF discipline (the exact count cell;
    the staircase theorem extends equality to streaming) with the
    ``stage_count_recorder`` seam attached; restores the flow's discipline."""
    ref = nf_perceptron_counts(repr_, int(flow.simulation_length), samples)
    assembler = PerceptronCountAssembler(ir_graph)
    flow.stage_count_recorder = (
        lambda stage, counts: assembler.capture_stage(stage.output_map, counts)
    )
    prev_sync = getattr(flow, "lif_execution_synchronized", False)
    flow.lif_execution_synchronized = True
    try:
        with torch.no_grad():
            flow(samples)
    finally:
        flow.stage_count_recorder = None
        flow.lif_execution_synchronized = prev_sync
    aligned_ref, aligned_got, report = intersect_aligned(ref, assembler.assemble())
    cert = certify_spike_counts(
        lambda _b: aligned_ref, lambda _b: aligned_got, [samples],
        backend=backend,
    )
    return cert, f"{assembler.last_report} | {report}"


def intersect_aligned(
    ref: dict[int, torch.Tensor], got: dict[int, torch.Tensor]
) -> tuple[dict, dict, str]:
    """Common-key views plus a dropped-key report; width mismatch fails loud."""
    common = sorted(set(ref) & set(got))
    for k in common:
        if ref[k].shape[1] != got[k].shape[1]:
            raise ValueError(
                f"perceptron {k}: reference width {ref[k].shape[1]} != "
                f"backend width {got[k].shape[1]}"
            )
    report = (
        f"aligned={common} reference-only={sorted(set(ref) - set(got))} "
        f"backend-only={sorted(set(got) - set(ref))}"
    )
    return {k: ref[k] for k in common}, {k: got[k] for k in common}, report
