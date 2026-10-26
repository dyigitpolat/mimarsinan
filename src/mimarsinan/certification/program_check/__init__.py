"""[W15] The static per-deployment check of Theorem 1's program conditions and the statically checkable executor
conditions, and of Proposition 2's network conditions (``check_deployment``): its condition map and named rules, the
executor code clauses its verdicts assume, its declaration and its registry of program forms."""

from mimarsinan.certification.program_check.check import (
    CONDITION_MAP, REQUIRED, RULES, UNMUTATED, check_deployment,
)
from mimarsinan.certification.program_check.declaration import CheckDeclaration, declaration_from_config
from mimarsinan.certification.program_check.forms import ProgramForm, register_program_form, unknown_forms
from mimarsinan.certification.program_check.report import ASSUMED, Assumption, StaticCheckReport

__all__ = [
    "ASSUMED", "Assumption", "CONDITION_MAP", "REQUIRED", "RULES", "UNMUTATED", "CheckDeclaration", "ProgramForm",
    "StaticCheckReport", "check_deployment", "declaration_from_config", "register_program_form", "unknown_forms",
]
