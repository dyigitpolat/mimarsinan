"""Graphviz writer."""

from __future__ import annotations

import html
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping
from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRNode, IRSource, NeuralCore
from mimarsinan.mapping.packing.softcore import HardCoreMapping
from mimarsinan.common.layer_key import layer_key_from_node_name
from mimarsinan.common.safe_numeric import safe_float

import re


from mimarsinan.visualization.graphviz.common import try_render_dot, _embed_svg_images, _percent, _compress_ranges, _truncate, _dot_html_label, _dot_html_label_mixed, _stack_sample_lines

@dataclass
class HybridVizArtifacts:
    program_dot: str
    segment_dots: list[str]

