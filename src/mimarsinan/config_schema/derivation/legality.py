"""The legal-value-set law's derivation surface (registry-backed)."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Tuple

from mimarsinan.common.refusal.legal_value import illegal_value_message
from mimarsinan.config_schema.registry import REGISTRY


def legal_value_error(flat_key: str, value: Any, legal: Iterable[Any]) -> ValueError:
    """THE canonical illegal-value message (the wizard renders the same text)."""
    return ValueError(illegal_value_message(flat_key, value, legal))


def legal_values_for(flat_key: str, cfg: Mapping[str, Any]) -> Optional[Tuple[Any, ...]]:
    """The registry's legal value set for ``flat_key`` under this config state.
    ``None`` = legality does not apply here (the rule was not consulted): neither
    locked nor judged. An EMPTY tuple = consulted and admits nothing."""
    result = REGISTRY[flat_key].legal_values(cfg)  # type: ignore[misc]
    return None if result is None else tuple(result)


def legality_bearing_keys() -> Tuple[str, ...]:
    """Keys whose legality depends on other config (registry-declared)."""
    return tuple(k for k, e in REGISTRY.items() if e.legal_values is not None)
