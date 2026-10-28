"""Registry entries: the opt-in event census of the deployed program (U-S)."""

from __future__ import annotations

from mimarsinan.config_schema.registry.relevance import Relevance as R
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
    frozen_default as _frozen,
)

ENTRIES = (
    _E("event_census_samples", domain="event", group="deployment_target",
       owner="pipelining/pipeline_steps/verification/event_census_step", type=T.INT,
       category=Category.ADVANCED, exposure="user", bounds=(0, None),
       label="Event Census Samples",
       effect="Measures the deployed program's synaptic events on this many validation samples",
       doc="The opt-in event census: the first N validation samples (never the test split) run through the "
           "deployed program on the Hard Core Mapping executor, one sample at a time, and each sample's "
           "synaptic events (per core, the recorded input spikes times the used columns) are counted; the "
           "record seals their mean per inference as the energy fragment's synaptic_events, which prices the "
           "record's spike-dependent energy. 0 or absent runs no census.",
       relevant=R.when("spiking_mode", in_=("lif", "ttfs_cycle_based")),
       provenance="consumer frozen default", derived_default=_frozen(0),
       empty_means="no event census: the record's energy fragment is SANA-FE's or absent, byte for byte"),
)
