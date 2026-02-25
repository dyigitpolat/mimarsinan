from __future__ import annotations

from typing import Generic, TypeVar

from mimarsinan.search.problem import SearchProblem
from mimarsinan.search.results import SearchResult


ConfigT = TypeVar("ConfigT")


class SearchOptimizer(Generic[ConfigT]):
    """
    Search backend interface.
    """

    def optimize(self, problem: SearchProblem[ConfigT], reporter=None) -> SearchResult[ConfigT]:
        raise NotImplementedError


