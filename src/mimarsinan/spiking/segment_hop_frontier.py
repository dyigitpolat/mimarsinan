"""[5v B2] the intra-segment hop frontier: genuine prefix hops + proxy suffix."""

from __future__ import annotations

from mimarsinan.spiking.segment_partition import perceptron_of


def run_segment_hop_hybrid(policy, driver, seg_nodes, values, x, k: int):
    """[5v B2] the intra-segment k-hybrid: hops below the frontier run the
    deployed cycle loop (bit-invariant to the suffix mode — the depth-k
    prefix is closed under in-segment deps), deeper hops run the trained
    proxy on the frontier's DECODED values."""
    depth = policy.segment_depths(driver, seg_nodes)
    n_levels = max(depth.values(), default=0) + 1
    if k >= n_levels:
        return policy._run_segment_genuine(driver, seg_nodes, values, x)
    if k <= 0:
        return policy._run_segment_value_mode(driver, seg_nodes, values, x)
    prefix = [n for n in seg_nodes if depth[n] < k]
    suffix = [n for n in seg_nodes if depth[n] >= k]

    hybrid_values = dict(values)
    policy._run_segment_genuine(driver, prefix, hybrid_values, x)

    vmode = policy._value_mode_forward(driver, suffix, hybrid_values, x)
    suffix_set = set(suffix)
    for n in driver.external_consumed(seg_nodes):
        values[n] = vmode[n] if n in suffix_set else hybrid_values[n]
    if policy.node_value_recorder is not None:
        for n in suffix:
            if perceptron_of(n) is not None:
                policy.node_value_recorder[n] = vmode[n].detach()
    if driver._output in suffix_set:
        return vmode[driver._output]
    if driver._output in set(prefix):
        return hybrid_values.get(driver._output)
    return None
