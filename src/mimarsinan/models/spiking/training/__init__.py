"""Differentiable spike-train training forwards (KD fine-tuning through deployment dynamics)."""

from mimarsinan.models.spiking.training.ttfs_segment_forward import (
    TTFSSegmentForward,
    partition_perceptron_segments,
)

__all__ = ["TTFSSegmentForward", "partition_perceptron_segments"]
