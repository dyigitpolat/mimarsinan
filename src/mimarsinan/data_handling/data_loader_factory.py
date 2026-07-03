import atexit
import multiprocessing as _mp
import os
import sys

from mimarsinan.common.best_effort import best_effort
from mimarsinan.common.env import resource_debug_enabled
from mimarsinan.data_handling.data_provider import DataProvider

import torch
import torch.multiprocessing as torch_mp


# DataLoader workers use forkserver (not the global spawn start method) to fork from a clean CUDA-free process, avoiding truncated dataset pickles under resource pressure.
_DATALOADER_MP_CONTEXT = torch_mp.get_context("forkserver")


def _resource_snapshot(tag):
    if not resource_debug_enabled():
        return
    with best_effort("resource snapshot"):
        pid = os.getpid()
        try:
            fd_count = len(os.listdir(f"/proc/{pid}/fd"))
        except OSError:
            fd_count = -1
        try:
            shm = os.listdir("/dev/shm")
            shm_count = len(shm)
            sem_count = sum(1 for n in shm if n.startswith("sem."))
        except OSError:
            shm_count = -1
            sem_count = -1
        rss_kb = -1
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_kb = int(line.split()[1])
                        break
        except OSError:
            pass
        children = len(_mp.active_children())
        print(
            f"[resource] {tag} pid={pid} rss_kb={rss_kb} "
            f"fd={fd_count} shm={shm_count} sem={sem_count} children={children}",
            file=sys.stderr,
            flush=True,
        )


def _unregister_dataloader_atexit_handlers(it):
    """Unregister PyTorch's per-worker atexit cleanup callbacks for fast, quiet exit.

    Best-effort: never raises.
    """
    with best_effort("unregister dataloader atexit handlers"):
        workers = getattr(it, "_workers", None)
        if not workers:
            return
        from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter
        cleanup_func = _MultiProcessingDataLoaderIter._clean_up_worker
        atexit.unregister(cleanup_func)


def shutdown_data_loader(loader):
    """Shut down a multi-worker DataLoader's workers/queues before process exit.

    Idempotent and best-effort: never raises.
    """
    if loader is None:
        return
    if getattr(loader, "num_workers", 0) == 0:
        return
    _resource_snapshot("shutdown:enter")
    with best_effort("dataloader worker shutdown"):
        it = getattr(loader, "_iterator", None)
        if it is not None and hasattr(it, "_shutdown_workers"):
            it._shutdown_workers()
            _unregister_dataloader_atexit_handlers(it)
        if hasattr(loader, "_iterator"):
            loader._iterator = None
    _resource_snapshot("shutdown:exit")


class DataLoaderFactory:
    def __init__(self, data_provider_factory, num_workers=4):
        self._data_provider_factory = data_provider_factory
        self._num_workers = num_workers

        self._persistent_workers = num_workers > 0
        self._pin_memory = True
        self._ffcv_factory = None

    @classmethod
    def for_pipeline(cls, pipeline) -> "DataLoaderFactory":
        """The single pipeline-facing seam: honors config ``num_workers``."""
        return cls(
            pipeline.data_provider_factory,
            num_workers=pipeline.config.get("num_workers", 4),
        )

    def _ffcv_loader(self, kind: str, batch_size, data_provider):
        """Return an FFCV loader iff the provider opts in; else ``None``.

        When the provider opts in FFCV is a hard requirement: import and
        FFCV-side errors propagate unchanged, with no silent fallback.
        """
        if not data_provider.enable_ffcv():
            return None
        if self._ffcv_factory is None:
            from mimarsinan.data_handling.ffcv.loader_factory import FFCVLoaderFactory
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._ffcv_factory = FFCVLoaderFactory(
                self._data_provider_factory,
                num_workers=self._num_workers,
                device=device,
            )
        if kind == "train":
            return self._ffcv_factory.create_training_loader(batch_size, data_provider)
        if kind == "val":
            return self._ffcv_factory.create_validation_loader(batch_size, data_provider)
        if kind == "test":
            return self._ffcv_factory.create_test_loader(batch_size, data_provider)
        raise ValueError(f"unknown split kind: {kind!r}")

    def _get_torch_dataloader(
            self, dataset, batch_size, shuffle, mp_safe):

        if not mp_safe:
            workers = 0
            pw = False
        else:
            workers = self._num_workers
            pw = self._persistent_workers

        mp_ctx = _DATALOADER_MP_CONTEXT if workers > 0 else None
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=workers, pin_memory=self._pin_memory,
            persistent_workers=pw, multiprocessing_context=mp_ctx)

    def create_data_provider(self) -> DataProvider:
        return self._data_provider_factory.create()

    def create_training_loader(self, batch_size, data_provider):
        ffcv = self._ffcv_loader("train", batch_size, data_provider)
        if ffcv is not None:
            return ffcv
        return self._get_torch_dataloader(
            data_provider._get_training_dataset(),
            batch_size=batch_size, shuffle=True, mp_safe=data_provider.is_mp_safe())

    def create_validation_loader(self, batch_size, data_provider):
        ffcv = self._ffcv_loader("val", batch_size, data_provider)
        if ffcv is not None:
            return ffcv
        return self._get_torch_dataloader(
            data_provider._get_validation_dataset(),
            batch_size=batch_size, shuffle=False, mp_safe=data_provider.is_mp_safe())

    def create_test_loader(self, batch_size, data_provider):
        ffcv = self._ffcv_loader("test", batch_size, data_provider)
        if ffcv is not None:
            return ffcv
        return self._get_torch_dataloader(
            data_provider._get_test_dataset(),
            batch_size=batch_size, shuffle=False, mp_safe=data_provider.is_mp_safe())