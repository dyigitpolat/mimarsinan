"""Error types for the cascade equivalence certificate."""

from __future__ import annotations


class CascadeCertificateError(RuntimeError):
    """The pruned+compacted program is not value-identical to the seeded reference."""


class CascadeCertificatePreconditionError(CascadeCertificateError):
    """The certificate refuses to run: an exactness precondition is unmet."""
