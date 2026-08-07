"""Streamed-lif structural contract gates."""

from mimarsinan.mapping.verification.streamed.streamability import (
    NotStreamableError,
    assert_streamable_ir,
    assert_streamable_model_or_raise,
)

__all__ = [
    "NotStreamableError",
    "assert_streamable_ir",
    "assert_streamable_model_or_raise",
]
