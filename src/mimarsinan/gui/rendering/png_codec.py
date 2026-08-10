"""Minimal stdlib PNG encoder for RGB images (no matplotlib, no PIL)."""

from __future__ import annotations

import struct
import zlib

import numpy as np

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# Fixed compression level keeps the bytes deterministic across renders.
_ZLIB_LEVEL = 6


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def encode_rgb_png(rgb: np.ndarray) -> bytes:
    """Encode an ``(H, W, 3)`` uint8 array as PNG bytes.

    Pure function of the array (deterministic bytes): 8-bit truecolor,
    filter type 0 on every scanline, one IDAT chunk.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) array, got shape {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise ValueError(f"expected a uint8 array, got dtype {rgb.dtype}")
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    if height < 1 or width < 1:
        raise ValueError(f"cannot encode an empty image ({height}x{width})")

    scanlines = np.empty((height, 1 + width * 3), dtype=np.uint8)
    scanlines[:, 0] = 0
    scanlines[:, 1:] = np.ascontiguousarray(rgb).reshape(height, width * 3)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    body = zlib.compress(scanlines.tobytes(), _ZLIB_LEVEL)
    return (
        _PNG_SIGNATURE
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", body)
        + _chunk(b"IEND", b"")
    )


__all__ = ["encode_rgb_png"]
