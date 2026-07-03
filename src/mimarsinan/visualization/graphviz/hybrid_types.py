"""Graphviz writer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HybridVizArtifacts:
    program_dot: str
    segment_dots: list[str]

