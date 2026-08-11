"""The deployment record: one typed, versioned, provenance-carrying artifact per run."""

from mimarsinan.deployment_record.build import DeploymentRecordBuilder, SealPlanView
from mimarsinan.deployment_record.schema import (
    DEPLOYMENT_RECORD_FILENAME,
    DEPLOYMENT_RECORD_FORMAT_VERSION,
    DeploymentRecord,
    load_deployment_record,
    save_deployment_record,
)

__all__ = [
    "DEPLOYMENT_RECORD_FILENAME",
    "DEPLOYMENT_RECORD_FORMAT_VERSION",
    "DeploymentRecord",
    "DeploymentRecordBuilder",
    "SealPlanView",
    "load_deployment_record",
    "save_deployment_record",
]
