import numpy as np

from mimarsinan.mapping.ir import ComputeOp, IRGraph, NeuralCore
from mimarsinan.mapping.pruning.ir_segmentation import get_neural_segments

_EMPTY_SRC = np.array([], dtype=object)


def test_get_neural_segments_splits_on_compute_ops():
    nc1 = NeuralCore(id=0, name="n1", input_sources=_EMPTY_SRC)
    nc2 = NeuralCore(id=1, name="n2", input_sources=_EMPTY_SRC)
    co = ComputeOp(id=2, name="add", op_type="add", input_sources=_EMPTY_SRC)
    nc3 = NeuralCore(id=3, name="n3", input_sources=_EMPTY_SRC)
    graph = IRGraph(nodes=[nc1, nc2, co, nc3], output_sources=_EMPTY_SRC)
    segs = get_neural_segments(graph)
    assert len(segs) == 2
    assert [n.id for n in segs[0]] == [0, 1]
    assert [n.id for n in segs[1]] == [3]
