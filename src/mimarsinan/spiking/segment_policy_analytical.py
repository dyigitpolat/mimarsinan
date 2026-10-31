"""The value-domain analytical segment policy of the unified segment-forward driver."""

from __future__ import annotations


class AnalyticalSegmentPolicy:
    """Value-domain analytical execution: every node runs once on ideal values
    (the pointwise-analytical NF of ``ttfs`` / ``ttfs_quantized``)."""

    def prepare(self, driver):
        pass

    def finalize(self, driver):
        pass

    def run_segment(self, driver, seg_nodes, values, x):
        seg_set = set(seg_nodes)
        local: dict = {}

        def val_of(dep):
            return local[dep] if dep in seg_set else values[dep]

        for node in seg_nodes:
            d = driver._deps.get(node, [])
            if len(d) == 0:
                local[node] = node.forward(x)
            elif len(d) == 1:
                local[node] = node.forward(val_of(d[0]))
            else:
                local[node] = node.forward(tuple(val_of(dep) for dep in d))

        for n in driver.external_consumed(seg_nodes):
            values[n] = local[n]
        if driver._output in seg_set:
            return local[driver._output]
        return None
