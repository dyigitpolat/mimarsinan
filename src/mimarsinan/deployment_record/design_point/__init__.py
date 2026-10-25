"""Design-point instances: one network deployed on several chip rows, each certified on two executors, censused per
operator, timed per pass with its write events, judged for lifetime and minimum batch, priced on the exploration
plane, or refused by name. ``schema`` is the document's closed field rules (and their JSON Schema); ``rows`` builds
one judged row from a built program; ``chip_rows`` refuses a declaration whose capability is not built.
"""

from mimarsinan.deployment_record.design_point.chip_rows import declaration_refusals
from mimarsinan.deployment_record.design_point.rows import judged_row
from mimarsinan.deployment_record.design_point.schema import (
    SCHEMA_ID,
    SCHEMA_VERSION,
    instance_json_schema,
    validate_instance,
)

__all__ = [
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "declaration_refusals",
    "instance_json_schema",
    "judged_row",
    "validate_instance",
]
