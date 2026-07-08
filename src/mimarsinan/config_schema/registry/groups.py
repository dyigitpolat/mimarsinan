"""The nine concern groups every config key belongs to (one taxonomy, one home)."""

from __future__ import annotations

from typing import Dict, Tuple

CONCERN_GROUPS: Tuple[Dict[str, str], ...] = (
    {"id": "workload", "title": "Workload",
     "subtitle": "Dataset, preprocessing, input encoding", "accent": "91,141,245"},
    {"id": "model", "title": "Model",
     "subtitle": "Architecture builder, weight init, pruning, NAS",
     "accent": "45,212,191"},
    {"id": "spiking", "title": "Spiking semantics",
     "subtitle": "Firing/sync/encoding/thresholding/spike-gen/bias modes",
     "accent": "168,85,247"},
    {"id": "hardware", "title": "Hardware platform / capabilities",
     "subtitle": "Core grid, weight quantization, capability gates",
     "accent": "249,115,22"},
    {"id": "conversion", "title": "Conversion process",
     "subtitle": "Activation quant, calibration health, optimization driver",
     "accent": "139,92,246"},
    {"id": "tuning", "title": "Adaptation & tuning controller",
     "subtitle": "Rate-search budget, rollback, recovery, stabilization knobs",
     "accent": "251,191,36"},
    {"id": "training", "title": "Training",
     "subtitle": "Learning rate, epochs, recipes", "accent": "74,222,128"},
    {"id": "deployment_target", "title": "Deployment target",
     "subtitle": "Simulation backends and acceptance/parity gates",
     "accent": "103,232,249"},
    {"id": "run", "title": "Run / runtime",
     "subtitle": "Run identity and pipeline-resolved values", "accent": "34,211,238"},
)

VALID_GROUP_IDS = frozenset(g["id"] for g in CONCERN_GROUPS)
