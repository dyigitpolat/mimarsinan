"""Classify MNIST mixer Slurmech artifacts against closure gates."""

from __future__ import annotations

from artifact_classifier import (  # noqa: F401
    ArtifactRecord as MixerResultRecord,
    classify_artifact,
    classify_many,
    iter_artifact_dirs,
    main,
)

__all__ = [
    "MixerResultRecord",
    "classify_artifact",
    "classify_many",
    "iter_artifact_dirs",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
