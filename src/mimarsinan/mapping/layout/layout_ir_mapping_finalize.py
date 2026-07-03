from __future__ import annotations

from typing import Any, Dict, List

from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
from mimarsinan.mapping.layout.segmentation import (
    compute_host_side_segment_count,
    compute_node_latencies,
    compute_segment_ids,
)


class _LayoutIRMappingFinalize:

    def _compute_latencies(self) -> Dict[int, int]:
        return compute_node_latencies(
            self._node_input_node_ids, self._node_is_neural
        )

    def _compute_segment_ids(self) -> Dict[int, int]:
        return compute_segment_ids(
            self._node_input_node_ids, self._node_is_neural
        )

    def _compute_host_side_segment_count(
        self, segment_ids: Dict[int, int]
    ) -> int:
        return compute_host_side_segment_count(segment_ids, self._node_is_neural)

    def _build_layout_preview(
        self,
        segment_ids: Dict[int, int],
        latencies: Dict[int, int],
    ) -> Dict[str, Any]:
        neural_latency_tags = sorted({
            int(sc.latency_tag) for sc in self.layout_softcores
            if sc.latency_tag is not None
        })
        latency_to_index = {lat: idx for idx, lat in enumerate(neural_latency_tags)}
        min_neural_latency = neural_latency_tags[0] if neural_latency_tags else 0

        host_counts: Dict[int, int] = {}
        for node_id, is_neural in self._node_is_neural.items():
            if is_neural or node_id not in latencies:
                continue
            slot = max(0, int(latencies[node_id] - min_neural_latency + 1))
            host_counts[slot] = host_counts.get(slot, 0) + 1

        neural_summary: Dict[int, Dict[str, Any]] = {}
        for sc in self.layout_softcores:
            if sc.latency_tag is None:
                continue
            lat_tag = int(sc.latency_tag)
            info = neural_summary.setdefault(lat_tag, {
                "latency_tag": lat_tag,
                "latency_group_index": latency_to_index.get(lat_tag, 0),
                "softcore_count": 0,
                "segment_ids": set(),
            })
            info["softcore_count"] += 1
            if sc.segment_id is not None:
                info["segment_ids"].add(int(sc.segment_id))

        for info in neural_summary.values():
            seg_ids = sorted(info.pop("segment_ids"))
            info["segment_count"] = len(seg_ids)
            info["segment_ids"] = seg_ids

        neural_groups = [neural_summary[k] for k in sorted(neural_summary)]
        host_segments = [
            {"slot": slot, "compute_op_count": host_counts[slot]}
            for slot in sorted(host_counts)
        ]

        flow: List[Dict[str, Any]] = [{"kind": "input"}]
        max_slot = max(
            [len(neural_groups)] + list(host_counts.keys())
        ) if (neural_groups or host_counts) else 0
        for slot in range(max_slot + 1):
            if slot in host_counts:
                flow.append({
                    "kind": "host",
                    "slot": slot,
                    "compute_op_count": host_counts[slot],
                })
            if slot < len(neural_groups):
                group = next(
                    (g for g in neural_groups if g["latency_group_index"] == slot),
                    None,
                )
                if group is not None:
                    flow.append({
                        "kind": "neural",
                        "latency_group_index": group["latency_group_index"],
                        "latency_tag": group["latency_tag"],
                        "softcore_count": group["softcore_count"],
                        "segment_count": group["segment_count"],
                    })
        flow.append({"kind": "output"})

        return {
            "neural_segments": [
                {
                    "segment_id": int(seg_id),
                    "softcore_count": sum(
                        1 for sc in self.layout_softcores
                        if sc.segment_id == seg_id
                    ),
                    "latency_group_count": len({
                        int(sc.latency_tag) for sc in self.layout_softcores
                        if sc.segment_id == seg_id and sc.latency_tag is not None
                    }),
                }
                for seg_id in sorted({
                    int(sc.segment_id) for sc in self.layout_softcores
                    if sc.segment_id is not None
                })
            ],
            "latency_groups": neural_groups,
            "host_segments": host_segments,
            "flow": flow,
        }

    def _finalize_softcores(self) -> None:
        """Rewrite each softcore with its finalised latency_tag, segment_id, and
        threshold_group_id (= perceptron_index, or a unique fallback when None)."""
        latencies = self._compute_latencies()
        segment_ids = self._compute_segment_ids()
        self.host_side_segment_count = self._compute_host_side_segment_count(segment_ids)

        # Non-perceptron cores get unique negative ids so they never collide with perceptron indices (>= 0).
        for node_id, sc_idx in self._node_id_to_softcore_idx.items():
            latency = latencies.get(node_id, 0)
            segment_id = segment_ids.get(node_id, 0)

            pi = self._sc_idx_to_perceptron_index.get(sc_idx)
            tg = int(pi) if pi is not None else -(sc_idx + 1)

            old = self.layout_softcores[sc_idx]
            self.layout_softcores[sc_idx] = LayoutSoftCoreSpec(
                input_count=old.input_count,
                output_count=old.output_count,
                threshold_group_id=tg,
                latency_tag=int(latency),
                segment_id=int(segment_id),
                name=old.name,
            )

        self.layout_preview = self._build_layout_preview(segment_ids, latencies)
