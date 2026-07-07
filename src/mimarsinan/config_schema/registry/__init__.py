"""The configurability SSOT: one declarative registry for every config key."""

from mimarsinan.config_schema.registry.build import (
    NON_PIPELINE_DOC_KEYS,
    REGISTRY,
    keys_in_category,
    schema_for,
    section_keys,
    serialize_registry,
)
from mimarsinan.config_schema.registry.groups import CONCERN_GROUPS
from mimarsinan.config_schema.registry.parse import ParsedDocument, parse_deployment_document
from mimarsinan.config_schema.registry.relevance import Relevance
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema,
    FieldType,
    NO_DEFAULT,
)

__all__ = [
    "CONCERN_GROUPS",
    "Category",
    "ConfigKeySchema",
    "FieldType",
    "NON_PIPELINE_DOC_KEYS",
    "NO_DEFAULT",
    "ParsedDocument",
    "REGISTRY",
    "Relevance",
    "keys_in_category",
    "parse_deployment_document",
    "schema_for",
    "section_keys",
    "serialize_registry",
]
