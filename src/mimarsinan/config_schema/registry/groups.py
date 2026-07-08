"""The concern groups every config key belongs to (one taxonomy, one home).

A group may declare ``empty_state``: the quiet one-line status its card shows
when none of its keys exist under the current config (instead of vanishing).
"""

from __future__ import annotations

from typing import Dict, Tuple

CONCERN_GROUPS: Tuple[Dict[str, str], ...] = (
    {"id": "workload", "title": "Workload",
     "subtitle": "Dataset, preprocessing, input encoding", "accent": "91,141,245"},
    {"id": "model", "title": "Model",
     "subtitle": "Architecture builder, weight init", "accent": "45,212,191"},
    {"id": "spiking", "title": "Spiking semantics",
     "subtitle": "Firing/sync/encoding/thresholding/spike-gen/bias modes",
     "accent": "168,85,247"},
    {"id": "hardware", "title": "Hardware platform / capabilities",
     "subtitle": "Core grid, weight precision, capability gates (what the "
                 "hardware CAN do)",
     "accent": "249,115,22"},
    {"id": "mapping_strategy", "title": "Mapping strategy",
     "subtitle": "What we CHOOSE when mapping: scheduling, encoding "
                 "placement, pruning",
     "accent": "56,189,148"},
    {"id": "co_search", "title": "Co-search",
     "subtitle": "Automated co-design: the search discovers model and/or "
                 "hardware configs",
     "accent": "244,114,182",
     "empty_state": "off — model and hardware are hand-specified"},
    {"id": "conversion", "title": "Conversion process",
     "subtitle": "Activation quant, calibration health, optimization driver",
     "accent": "139,92,246"},
    {"id": "tuning", "title": "Adaptation & tuning controller",
     "subtitle": "Rate-search budget, rollback, recovery, tuning recipe",
     "accent": "251,191,36"},
    {"id": "training", "title": "Training",
     "subtitle": "Learning rate, epochs, pretrained weights, training recipe",
     "accent": "74,222,128"},
    {"id": "deployment_target", "title": "Deployment target",
     "subtitle": "Deployment options, simulation vehicles, acceptance/parity gates",
     "accent": "103,232,249"},
    {"id": "run", "title": "Run / runtime",
     "subtitle": "Run identity and pipeline-resolved values", "accent": "34,211,238"},
)

VALID_GROUP_IDS = frozenset(g["id"] for g in CONCERN_GROUPS)
