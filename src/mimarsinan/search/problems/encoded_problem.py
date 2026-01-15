from __future__ import annotations

from typing import Protocol, Sequence, TypeVar

import numpy as np

from mimarsinan.search.problem import SearchProblem


ConfigT = TypeVar("ConfigT")


class EncodedProblem(SearchProblem[ConfigT], Protocol):
    """
    A SearchProblem that is evaluated on decoded configurations, but optimized over an encoded vector x.
    """

    @property
    def n_var(self) -> int:
        ...

    @property
    def xl(self) -> np.ndarray:
        ...

    @property
    def xu(self) -> np.ndarray:
        ...

    def decode(self, x: np.ndarray) -> ConfigT:
        ...


