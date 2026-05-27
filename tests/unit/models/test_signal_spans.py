import torch

from mimarsinan.models.signal_spans import fill_signal_from_spans


class _Span:
    def __init__(self, kind, dst_start, dst_end, src_start=0, src_end=1, src_node_id=0, src_core=0):
        self.kind = kind
        self.dst_start = dst_start
        self.dst_end = dst_end
        self.src_start = src_start
        self.src_end = src_end
        self.src_node_id = src_node_id
        self.src_core = src_core


def test_fill_signal_input_and_on():
    out = torch.zeros(2, 4)
    inp = torch.ones(2, 2)
    fill_signal_from_spans(
        out,
        [_Span("on", 0, 1), _Span("input", 2, 4, src_end=2)],
        read_input=lambda sp: out[:, int(sp.dst_start) : int(sp.dst_end)].copy_(inp),
        read_upstream=lambda sp: None,
    )
    assert out[0, 0] == 1.0
    assert out[0, 2] == 1.0
