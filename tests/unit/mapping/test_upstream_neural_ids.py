from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.latency.upstream import iter_upstream_neural_ids


def test_iter_upstream_skips_off():
    sources = [IRSource(-1, 0), IRSource(3, 1), IRSource(5, 0)]
    assert list(iter_upstream_neural_ids(sources)) == [3, 5]
