"""[W15] The registry of program forms (``registry``) and this tree's base forms, attribute by attribute (``base``)."""

from mimarsinan.certification.program_check.forms.base import BACKEND, IDENTITY, LABEL, UNREAD
from mimarsinan.certification.program_check.forms.registry import (
    ELEMENTS, ProgramForm, extra_forms, register_program_form, registered_attributes, run_form_predicates,
    scan_forms, unknown_forms, unregister_program_form,
)

__all__ = [
    "BACKEND", "ELEMENTS", "IDENTITY", "LABEL", "ProgramForm", "UNREAD", "extra_forms", "register_program_form",
    "registered_attributes", "run_form_predicates", "scan_forms", "unknown_forms", "unregister_program_form",
]
