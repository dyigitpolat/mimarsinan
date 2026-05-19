"""Shared SearchSpaceDescription factories for search unit tests."""

from __future__ import annotations

from mimarsinan.search.search_space_description import SearchSpaceDescription


def make_search_space_description(**overrides) -> SearchSpaceDescription:
    arch_search = {
        "objectives": ["estimated_accuracy", "total_params"],
        "pop_size": 4,
        "generations": 2,
        **overrides.get("arch_search", {}),
    }
    return SearchSpaceDescription.from_arch_search(arch_search)
