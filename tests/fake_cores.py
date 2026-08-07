"""Duck-typed hard-core stand-ins that speak the core-matrix carrier protocol."""

from types import SimpleNamespace


class FakeCore(SimpleNamespace):
    """A ``SimpleNamespace`` core that also resolves its weight grid.

    Real hard cores store placement descriptors and materialize the padded
    dense grid on demand, so every consumer reads weights through
    ``get_core_matrix()``; a stand-in must answer the same protocol.
    """

    def has_core_matrix(self) -> bool:
        return getattr(self, "core_matrix", None) is not None

    def get_core_matrix(self):
        matrix = getattr(self, "core_matrix", None)
        if matrix is None:
            raise ValueError(
                "FakeCore: no core_matrix to resolve a weight grid from."
            )
        return matrix
