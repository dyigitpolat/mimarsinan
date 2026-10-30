"""Deployment-native vehicle models (streamable-by-construction architectures)."""

from mimarsinan.models.vehicles.narrow_conv import NarrowConvNet
from mimarsinan.models.vehicles.stream_cnn import StreamCNN

__all__ = ["NarrowConvNet", "StreamCNN"]
