"""Chip descriptions read from their sources: the computation keys each states or names unstated, the reading rule its
statements are judged by, and the produced realizations a description declares (``computation_keys``, ``readings``,
``realizations``); [U-CHIPKEYS3] the run door's reading of a stated pricing absence (``departures``)."""

# [U-CHIPKEYS3] the loader first: target_bundles imports this package's modules in the one order that closes the
# cycle through the registry, so each of them imports as a process's first import.
from mimarsinan.mapping.platform import target_bundles as target_bundles  # noqa: E402
