"""In-flight observation of value-flow weight uploads: lifetime and identity."""

import contextlib
import gc
import weakref


class UploadProbe:
    """Records every ``_upload`` return weakly, plus per-neural-stage marks of
    upload counts and live uploaded bytes (deduped by storage pointer).

    ``keep=True`` additionally retains the returned tensors and the
    ``_PreparedValueSegment`` instances strongly (for identity assertions);
    lifetime tests must use ``keep=False`` so the probe itself retains nothing.
    """

    def __init__(self, keep: bool = False):
        self.keep = keep
        self.records = []      # (core, tensor_weakref, nbytes, data_ptr)
        self.kept = []         # (core, tensor) when keep=True
        self.prepared = []     # (hcm, _PreparedValueSegment) when keep=True
        self.stage_marks = []  # (stage, records_after, live_bytes_after)

    def on_upload(self, core, tensor) -> None:
        storage = tensor.untyped_storage()
        self.records.append((
            core, weakref.ref(tensor), int(storage.nbytes()),
            int(storage.data_ptr()),
        ))
        if self.keep:
            self.kept.append((core, tensor))

    def recorder(self, stage, _seg_output) -> None:
        gc.collect()  # reference cycles must not inflate the live sample
        self.stage_marks.append(
            (stage, len(self.records), self.live_uploaded_bytes())
        )

    def live_uploaded_bytes(self) -> int:
        by_ptr = {}
        for _core, ref, nbytes, ptr in self.records:
            if ref() is not None:
                by_ptr[ptr] = nbytes
        return sum(by_ptr.values())

    def per_stage_upload_counts(self):
        counts, prev = [], 0
        for _stage, n_after, _live in self.stage_marks:
            counts.append(n_after - prev)
            prev = n_after
        return counts

    def per_chain_uploaded_bytes(self):
        """Unique uploaded bytes per residency chain (a chain = one
        non-resident head stage + its consecutive resident passes)."""
        chains, prev = [], 0
        for stage, n_after, _live in self.stage_marks:
            is_head = not bool(
                getattr(stage, "schedule_weights_resident", False)
            )
            if is_head or not chains:
                chains.append({})
            for _core, _ref, nbytes, ptr in self.records[prev:n_after]:
                chains[-1][ptr] = nbytes
            prev = n_after
        return [sum(c.values()) for c in chains]


@contextlib.contextmanager
def probe_uploads(flow=None, keep: bool = False):
    """Patch ``value_execution._upload`` (and, when ``keep``,
    ``_PreparedValueSegment``) to observe uploads in flight; when ``flow`` is
    given, attach the per-neural-stage recorder to its recorder seam."""
    import mimarsinan.chip_simulation.value_run.value_execution as ve

    probe = UploadProbe(keep=keep)
    real_upload = ve._upload

    def spy(core, dtype, device, memo):
        tensor = real_upload(core, dtype, device, memo)
        probe.on_upload(core, tensor)
        return tensor

    real_prepared_cls = ve._PreparedValueSegment

    class _RecordingPrepared(real_prepared_cls):
        def __init__(self, hcm, *args, **kwargs):
            super().__init__(hcm, *args, **kwargs)
            probe.prepared.append((hcm, self))

    ve._upload = spy
    if keep:
        ve._PreparedValueSegment = _RecordingPrepared
    prev_recorder = flow.stage_count_recorder if flow is not None else None
    if flow is not None:
        flow.stage_count_recorder = probe.recorder
    try:
        yield probe
    finally:
        ve._upload = real_upload
        if keep:
            ve._PreparedValueSegment = real_prepared_cls
        if flow is not None:
            flow.stage_count_recorder = prev_recorder
