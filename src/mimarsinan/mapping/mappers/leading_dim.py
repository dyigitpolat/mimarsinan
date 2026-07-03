"""Leading-dimension mappers: merge/split/ensure 2D for batch-feature layout."""

from __future__ import annotations

from mimarsinan.mapping.mappers.base import Mapper


class MergeLeadingDimsMapper(Mapper):
    """Flatten all leading dims except the last feature dim ((B,N,F)->(B*N,F)); IR mapping is identity."""

    def __init__(self, source_mapper):
        super(MergeLeadingDimsMapper, self).__init__(source_mapper)

    def _map_to_ir(self, ir_mapping):
        return self.source_mapper.map_to_ir(ir_mapping)

    def _forward_impl(self, x):
        if x.dim() <= 2:
            return x
        return x.reshape(-1, x.shape[-1])


class SplitLeadingDimMapper(Mapper):
    """Inverse of MergeLeadingDimsMapper for (B*N,F)->(B,N,F) where N=second_dim_size; IR mapping is identity."""

    def __init__(self, source_mapper, second_dim_size: int):
        super(SplitLeadingDimMapper, self).__init__(source_mapper)
        self.second_dim_size = int(second_dim_size)

    def _map_to_ir(self, ir_mapping):
        return self.source_mapper.map_to_ir(ir_mapping)

    def _forward_impl(self, x):
        if x.dim() != 2:
            return x
        n = self.second_dim_size
        assert x.shape[0] % n == 0, f"Cannot split leading dim {x.shape[0]} by {n}"
        b = x.shape[0] // n
        return x.view(b, n, x.shape[1])


class Ensure2DMapper(Mapper):
    """Ensure a 2D (Instances, Features) mapping layout and a (Batch, Features) forward layout."""

    def __init__(self, source_mapper):
        super(Ensure2DMapper, self).__init__(source_mapper)

    def _map_to_ir(self, ir_mapping):
        sources = self.source_mapper.map_to_ir(ir_mapping)
        if len(sources.shape) == 1:
            return sources.reshape(1, -1)
        return sources

    def _forward_impl(self, x):
        if x.dim() == 1:
            return x.unsqueeze(0)
        return x
