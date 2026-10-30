"""Registry entries: the HACC NUS - ODIN Deployment export and its campaign."""

from __future__ import annotations

from mimarsinan.config_schema.registry.entries_platform_backends import (
    _meta_backend_enable,
    _why_backend_enable,
)
from mimarsinan.config_schema.registry.relevance import Relevance as R
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
)

_ENABLE = "enable_odin_hacc_export"


ENTRIES = (
    _E(_ENABLE, domain="event", group="deployment_target",
       owner="ConversionPolicy/backend_registry", type=T.BOOL,
       category=Category.DERIVED, derivation="derived", exposure="derived",
       label="HACC NUS - ODIN Deployment",
       doc="Whether the deployed network is FROZEN into a sealed HACC "
           "deployment bundle: per-core programs, per-sample stimuli, the "
           "routing plan and the expectations a board certifies itself "
           "against. OPT-IN: the capability derivation admits it wherever the "
           "ODIN crossbar has an executor for the declared soma law, but it "
           "stays off until the document asks, because freezing costs a full "
           "pass over the shipped sample set.",
       derived_from=("spiking_mode", "firing_granularity", "membrane_arithmetic"),
       why=_why_backend_enable(
           "odin_hacc",
           "the ODIN crossbar has no executor for this soma law or mode"),
       meta=_meta_backend_enable("odin_hacc"), provenance="ConversionPolicy recipe"),
    _E("odin_hacc_bundle_samples", domain="event", group="deployment_target",
       owner="odin_hacc_export", type=T.INT, category=Category.ADVANCED,
       exposure="user", label="ODIN Bundle Samples", bounds=(1, None),
       effect="HOW MANY test-set samples the frozen bundle can execute",
       doc="A bundle carries the entry raster its HOST stages produced for "
           "each sample, so it is executable for exactly the samples it ships. "
           "The board campaign may run FEWER of them (ODIN_DEPLOY_SAMPLES on "
           "the node); shipping the whole test set is a matter of raising this "
           "number and paying the export wall and the bundle size.",
       relevant=R.when_true(_ENABLE)),
    _E("odin_hacc_certification_samples", domain="event",
       group="deployment_target", owner="odin_hacc_export", type=T.INT,
       category=Category.ADVANCED, exposure="user",
       label="ODIN Certification Subset", bounds=(1, None),
       effect="HOW MANY samples carry frozen PER-PASS counts, not just a readout",
       doc="Per-pass certification is what proves the inter-core transcode on "
           "the board, and it does not need every sample: the subset's "
           "per-core window counts are frozen, every shipped sample's readout "
           "is. Must not exceed odin_hacc_bundle_samples.",
       relevant=R.when_true(_ENABLE)),
    _E("odin_hacc_bundle_name", domain="event", group="deployment_target",
       owner="odin_hacc_export", type=T.STR, category=Category.ADVANCED,
       exposure="user", label="ODIN Bundle Name",
       doc="The frozen bundle's name, which the board report and the package "
           "index carry. Empty = the experiment name.",
       relevant=R.when_true(_ENABLE)),
)
