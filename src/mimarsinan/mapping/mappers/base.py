"""Base Mapper class for the mapper DAG (forward pass and hardware mapping)."""

from __future__ import annotations

import torch.nn as nn


class Mapper(nn.Module):
    def __init__(self, source_mapper=None):
        super(Mapper, self).__init__()
        self._source_mapper_container = [source_mapper]
        self.sources = None
        self._cached_mapping_id = None
        self._cached_output = None
        self._cached_input_id = None
        self._ir_sources = None
        self._cached_ir_mapping_id = None

    def get_source_mappers(self):
        """Return the list of source mappers this node depends on."""
        if self.source_mapper is not None:
            return [self.source_mapper]
        return []

    def owned_perceptron_groups(self):
        """Introspection hook for Perceptron-first pipelines. Default: no perceptrons."""
        return []

    @property
    def source_mapper(self):
        return self._source_mapper_container[0]

    def clear_cache(self):
        self._cached_output = None
        self._cached_input_id = None
        self._ir_sources = None
        self._cached_ir_mapping_id = None

    def map(self, mapping):
        if self.sources is not None and self._cached_mapping_id == id(mapping):
            return self.sources
        self.sources = self._map(mapping)
        self._cached_mapping_id = id(mapping)
        return self.sources

    def _map(self, mapping):
        raise NotImplementedError

    def map_to_ir(self, ir_mapping):
        if self._ir_sources is not None and self._cached_ir_mapping_id == id(ir_mapping):
            return self._ir_sources
        self._ir_sources = self._map_to_ir(ir_mapping)
        self._cached_ir_mapping_id = id(ir_mapping)
        return self._ir_sources

    def _map_to_ir(self, ir_mapping):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _map_to_ir. "
            "Override this method to support unified IR mapping."
        )

    def forward(self, x):
        if self._cached_input_id == id(x) and self._cached_output is not None:
            return self._cached_output
        out = self._forward_impl(x)
        self._cached_input_id = id(x)
        self._cached_output = out
        return out

    def _forward_impl(self, x):
        raise NotImplementedError
