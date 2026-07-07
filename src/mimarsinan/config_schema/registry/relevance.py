"""Declarative relevance predicates over a resolved config, with a JSON codec."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple


@dataclass(frozen=True)
class Relevance:
    """A combinator tree the frontend can evaluate without shipping Python.

    Ops: ``always`` · ``in`` (key value in values) · ``true`` (key truthy) ·
    ``set`` (key present, non-null) · ``all`` / ``any`` (over ``items``).
    """

    op: str = "always"
    key: str = ""
    values: Tuple[Any, ...] = ()
    items: Tuple["Relevance", ...] = field(default=())

    @staticmethod
    def always() -> "Relevance":
        return Relevance(op="always")

    @staticmethod
    def when(key: str, *, in_: Tuple[Any, ...]) -> "Relevance":
        return Relevance(op="in", key=key, values=tuple(in_))

    @staticmethod
    def when_true(key: str) -> "Relevance":
        return Relevance(op="true", key=key)

    @staticmethod
    def when_set(key: str) -> "Relevance":
        return Relevance(op="set", key=key)

    @staticmethod
    def all_of(*items: "Relevance") -> "Relevance":
        return Relevance(op="all", items=tuple(items))

    @staticmethod
    def any_of(*items: "Relevance") -> "Relevance":
        return Relevance(op="any", items=tuple(items))

    def evaluate(self, config: Mapping[str, Any]) -> bool:
        if self.op == "always":
            return True
        if self.op == "in":
            return config.get(self.key) in self.values
        if self.op == "true":
            return bool(config.get(self.key))
        if self.op == "set":
            return config.get(self.key) is not None
        if self.op == "all":
            return all(item.evaluate(config) for item in self.items)
        if self.op == "any":
            return any(item.evaluate(config) for item in self.items)
        raise ValueError(f"unknown relevance op {self.op!r}")

    def to_json(self) -> Dict[str, Any]:
        if self.op == "always":
            return {"op": "always"}
        if self.op in ("in",):
            return {"op": "in", "key": self.key, "values": list(self.values)}
        if self.op in ("true", "set"):
            return {"op": self.op, "key": self.key}
        return {"op": self.op, "items": [item.to_json() for item in self.items]}

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> "Relevance":
        op = str(data.get("op", "always"))
        if op == "always":
            return Relevance.always()
        if op == "in":
            return Relevance.when(str(data["key"]), in_=tuple(data.get("values", ())))
        if op == "true":
            return Relevance.when_true(str(data["key"]))
        if op == "set":
            return Relevance.when_set(str(data["key"]))
        if op in ("all", "any"):
            items = tuple(Relevance.from_json(item) for item in data.get("items", ()))
            return Relevance(op=op, items=items)
        raise ValueError(f"unknown relevance op {op!r}")
