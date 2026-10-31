"""[U-HV] The sample split the ODIN steps read: declared by ``odin_sample_split``, the development split by default.

The HACC export ships a card campaign and certifies a subset of it; the device run runs samples through a device.
Both draw their samples from ONE declared split. Absent the key, that is the development split (the provider's
validation carve, the registry's frozen default read through ``effective_value``), so no ODIN step builds or reads
the test split unless its run declares ``"test"``, which only the test lock's runs do; any other value is refused by
name before a sample is read. The split and the per-split sample indices travel with the samples: the bundle's
provenance, each step's cache entry (``sample_read``), and the sealed record's ``device_samples``, which the
Deployment Record step joins from the entries ``device_sample_entries`` names.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.config_schema.registry import effective_value
from mimarsinan.data_handling.sample_loader import load_pairs_by_index, require_sample_split, sample_read
from mimarsinan.deployment_record.schema import Provenance

SAMPLE_SPLIT_KEY = "odin_sample_split"


class UnrecordedSampleRead(ToolchainRefusal, ValueError):
    """An ODIN step's cached entry with no ``sample_read`` (written before the declared split): its split is unknown."""

#: Each ODIN step that reads a declared split: its plan switch and the cache entry whose ``sample_read`` the record
#: seals, in the order the backend tail runs them.
DEVICE_SAMPLE_READS: Tuple[Tuple[str, str], ...] = (
    ("enable_odin_fpga_simulation", "odin_fpga_deployment_results"),
    ("enable_odin_hacc_export", "odin_hacc_deployment_bundle"),
)


def declared_split(config: Mapping[str, Any]) -> str:
    """The split this run's ODIN steps read: the declaration, else the development split; any other name refused."""
    return require_sample_split(effective_value(config, SAMPLE_SPLIT_KEY), declared_by=SAMPLE_SPLIT_KEY)


def read_campaign(pipeline: Any, indices: Iterable[int], *, backend: str
                  ) -> Tuple[List[Tuple[Any, Any]], Dict[str, Any]]:
    """The ``(sample, label)`` pairs of ``indices`` in the declared split, and the read's record for ``backend``."""
    config = pipeline.config
    split, wanted = declared_split(config), [int(index) for index in indices]
    pairs = load_pairs_by_index(pipeline.data_provider_factory, wanted, split=split,
                                num_workers=int(config.get("num_workers", 4)))
    return pairs, {"backend": str(backend), **sample_read(split, wanted, declared_by=SAMPLE_SPLIT_KEY)}


def device_sample_entries(plan: Any) -> List[str]:
    """The ODIN entries whose sample reads ``plan``'s record seals; none on a run without an ODIN step."""
    return [entry for switch, entry in DEVICE_SAMPLE_READS if bool(getattr(plan, switch, False))]


def attach_device_samples(builder: Any, step: Any, plan: Any, step_name: str) -> None:
    """Attach each ODIN step's sample read (from ``step``'s required entries) as the record's ``device_samples``."""
    reads = tuple(_sample_read_of(step.get_entry(entry), entry) for entry in device_sample_entries(plan))
    if reads:
        builder.attach("device_samples", reads, Provenance(
            kind="declared", producer=f"{SAMPLE_SPLIT_KEY} + each ODIN step's verified sample read", step=step_name))


def _sample_read_of(value: Mapping[str, Any], entry: str) -> Dict[str, Any]:
    """The sample read an ODIN step's entry carries; an entry cached before the declared split is refused by name."""
    if "sample_read" not in value:
        raise UnrecordedSampleRead(
            f"{entry}: the cached entry carries no sample_read (it was written before {SAMPLE_SPLIT_KEY} existed), "
            f"so the split its samples came from is unknown; rerun its step on this tree")
    return dict(value["sample_read"])
