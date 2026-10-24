"""The refusal of lane fusion (WI-5e): a planned lane the IR cannot realize, raised by name."""

from __future__ import annotations

from mimarsinan.common.refusal import ToolchainRefusal


class LaneFusionError(ToolchainRefusal, ValueError):
    """A planned lane the IR cannot realize (its producer is not a core, or is read beside the lane)."""
