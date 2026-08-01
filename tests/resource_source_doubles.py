"""Test doubles for the GUI resource-source interface."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from mimarsinan.gui.resources import ResourceSource


class CallableSource(ResourceSource):
    """An arbitrary zero-arg producer behind the source interface.

    For exercising the LIVE path (store laziness, HTTP serving) with a payload a
    test controls. It declares no ``SOURCE_TYPE``, so it can never be mistaken
    for a persistable format: a real resource must carry data, not a closure.
    """

    def __init__(self, produce: Callable[[], Any]) -> None:
        self._produce = produce

    def render(self) -> Any:
        return self._produce()

    def to_state(self) -> dict[str, Any]:
        raise NotImplementedError("a callable test double is not persistable")

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "CallableSource":
        raise NotImplementedError("a callable test double is not persistable")
