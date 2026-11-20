"""The canonical illegal-value text: one sentence for every key whose legal set depends on the config's state."""

from __future__ import annotations

from typing import Any, Iterable


def illegal_value_message(flat_key: str, value: Any, legal: Iterable[Any]) -> str:
    """THE canonical illegal-value message (the wizard renders the same text): the key, its value and the legal set the
    current config admits, with the remedy that accepts the derived value."""
    options = ", ".join(repr(option) for option in legal)
    return (f"{flat_key}={value!r} is not legal here: the current config admits {{{options}}}. Remove {flat_key} to "
            "accept the derived value.")
