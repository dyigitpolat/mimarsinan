import torch
import json
import logging
import os
import pickle

from mimarsinan.transformations.pruning.committed_masks import (
    commit_model_pruning,
    verify_model_pruning,
)

logger = logging.getLogger(__name__)

# Failure modes an unreadable/torn payload can raise at load time; the cache
# wraps these into CorruptCacheEntryError (anything else is a code bug).
ENTRY_LOAD_FAILURES = (
    OSError, EOFError, RuntimeError, ValueError, KeyError,
    pickle.UnpicklingError, json.JSONDecodeError,
)


class CorruptCacheEntryError(RuntimeError):
    """A cache entry's on-disk payload is unreadable (torn write / corruption)."""

    def __init__(self, name: str, directory: str, strategy: str, cause: BaseException):
        super().__init__(
            f"cache entry {name!r} in {directory!r} is unreadable "
            f"(strategy={strategy}): {cause!r}. The artifact is corrupt or torn "
            f"(e.g. a mid-write kill). Quarantine it so the producing step "
            f"re-runs: PipelineCache.quarantine_entry({directory!r}, {name!r})."
        )
        self.entry_name = name
        self.directory = directory


def write_atomically(final_path, write_payload) -> None:
    """A torn write must never replace a good artifact (the 0-byte-cache class):
    write to a sibling tmp file, then os.replace into place."""
    tmp_path = f"{final_path}.tmp"
    try:
        write_payload(tmp_path)
        os.replace(tmp_path, final_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _clear_transient_ir_caches(model) -> None:
    """Drop the walk-scoped IR memo before serialization.

    ``map_to_ir`` memoizes a per-node IRGraph (``_cached_ir_mapping``) that
    dedupes only within one traversal; left on the model it pins a multi-GB
    IRGraph into the cached ``.pt`` (a 197-token ViT bloated 328 MB -> 58 GB).
    The memo is never a persistent artifact — it rebuilds lazily on demand.
    """
    get_repr = getattr(model, "get_mapper_repr", None)
    if not callable(get_repr):
        return
    clear = getattr(get_repr(), "clear_ir_caches", None)
    if callable(clear):
        clear()


class LoadStoreStrategy:
    extension = ""

    def __init__(self, filename):
        self.filename = filename

    def path(self, cache_directory) -> str:
        return f"{cache_directory}/{self.filename}.{self.extension}"

    def load(self, cache_directory):
        raise NotImplementedError

    def store(self, cache_directory, object):
        raise NotImplementedError


class BasicLoadStoreStrategy(LoadStoreStrategy):
    extension = "json"

    def load(self, cache_directory):
        with open(self.path(cache_directory), "r") as f:
            return json.load(f)

    def store(self, cache_directory, object):
        def _write(path):
            with open(path, "w") as f:
                json.dump(object, f)

        write_atomically(self.path(cache_directory), _write)


class TorchModelLoadStoreStrategy(LoadStoreStrategy):
    extension = "pt"

    def load(self, cache_directory):
        (object, device) = torch.load(
            self.path(cache_directory),
            map_location=torch.device('cpu'),
            weights_only=False,
        )
        object._cached_original_device = device
        if isinstance(object, torch.nn.Module):
            # Round-trip half of the prune-parity contract: an artifact whose raw
            # params violate its committed masks must fail loud, never silently
            # reproduce different metrics than the live run (theory §5g-v (i)).
            verify_model_pruning(object, where=f"cache-load:{self.filename}")
        return object  # stay on CPU; consumers move to device as needed

    def store(self, cache_directory, object):
        if hasattr(object, "device"):
            device = object.device
        else:
            p = next(object.parameters(), None)
            device = p.device if p is not None else torch.device("cpu")

        if isinstance(object, torch.nn.Module):
            # The store boundary is an enforcement point like any hooked forward:
            # masks must hold in committed raw params before the artifact is written.
            commit_model_pruning(object)
            verify_model_pruning(object, where=f"cache-store:{self.filename}")

        _clear_transient_ir_caches(object)
        object.cpu()
        write_atomically(
            self.path(cache_directory),
            lambda path: torch.save((object, device), path),
        )
        # If the recorded device is no longer visible (narrower CUDA_VISIBLE_DEVICES) fall back to CPU rather than crash mid-save.
        try:
            object.to(device)
        except RuntimeError:
            logger.warning(
                "could not restore %s to %s after save; leaving on CPU",
                self.filename, device, exc_info=True,
            )


class PickleLoadStoreStrategy(LoadStoreStrategy):
    extension = "pickle"

    def load(self, cache_directory):
        with open(self.path(cache_directory), "rb") as f:
            return pickle.load(f)

    def store(self, cache_directory, object):
        def _write(path):
            with open(path, "wb") as f:
                pickle.dump(object, f)

        write_atomically(self.path(cache_directory), _write)
