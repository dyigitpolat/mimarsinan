import atexit
import os

from mimarsinan.data_handling.data_provider import DataProvider

import torch
import torch.multiprocessing as torch_mp


_RESOURCE_DEBUG = os.environ.get("MIMARSINAN_RESOURCE_DEBUG") == "1"

# DataLoader workers never touch CUDA in the parent's device context, so we
# avoid the global ``spawn`` start method (set in ``src/init.py`` for
# NevresimDriver / CUDA-aware ProcessPoolExecutors). ``forkserver`` forks from
# a clean, CUDA-free helper process, which bypasses the ~170 MiB-per-worker
# dataset pickle-over-pipe path that was failing with truncated pickles under
# accumulated resource pressure.
_DATALOADER_MP_CONTEXT = torch_mp.get_context("forkserver")


def _resource_snapshot(tag):
    if not _RESOURCE_DEBUG:
        return
    import sys
    try:
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
        import multiprocessing as _mp
        children = len(_mp.active_children())
        print(
            f"[resource] {tag} pid={pid} rss_kb={rss_kb} "
            f"fd={fd_count} shm={shm_count} sem={sem_count} children={children}",
            file=sys.stderr,
            flush=True,
        )
    except Exception:
        pass


def _unregister_dataloader_atexit_handlers(it):
    """Remove PyTorch's atexit worker cleanup callbacks so they don't run at process exit.

    When persistent_workers and pin_memory are True, PyTorch registers one atexit
    callback per worker. We already shut down workers in _shutdown_workers(); if we
    leave those callbacks registered, process exit runs them and causes slow exit
    (many join timeouts) and/or "Exception ignored in atexit callback" on Ctrl+C.

    We use the public atexit.unregister(func) API. On CPython 3.10+ the atexit
    module is a C built-in and does not expose _exithandlers, so the previous
    approach of mutating _exithandlers never removed any handlers. unregister(func)
    removes all registrations of that function, which is correct because we shut
    down all our loaders before process exit and we are the only user of
    multi-worker DataLoaders in this process. Best-effort: never raises.
    """
    try:
        workers = getattr(it, "_workers", None)
        if not workers:
            return
        from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter
        cleanup_func = _MultiProcessingDataLoaderIter._clean_up_worker
        # Prefer public API: works on all Python versions (incl. 3.10+ C atexit).
        # Removes all registrations of _clean_up_worker; we shut down all loaders
        # we create, so this is safe.
        atexit.unregister(cleanup_func)
    except Exception:
        pass


def shutdown_data_loader(loader):
    """Shut down a multi-worker DataLoader's worker processes and queues.

    Call this when done with a DataLoader that uses num_workers > 0 so that
    workers and IPC are cleaned up before process exit. Also unregisters
    PyTorch's atexit worker cleanup callbacks so process exit is fast and
    Ctrl+C does not produce atexit callback exceptions. Idempotent and
    best-effort: never raises.
    """
    if loader is None:
        return
    if getattr(loader, "num_workers", 0) == 0:
        return
    _resource_snapshot("shutdown:enter")
    try:
        it = getattr(loader, "_iterator", None)
        if it is not None and hasattr(it, "_shutdown_workers"):
            it._shutdown_workers()
            _unregister_dataloader_atexit_handlers(it)
        if hasattr(loader, "_iterator"):
            loader._iterator = None
    except Exception:
        pass
    _resource_snapshot("shutdown:exit")


class DataLoaderFactory:
    def __init__(self, data_provider_factory, num_workers=4):
        self._data_provider_factory = data_provider_factory
        self._num_workers = num_workers

        self._persistent_workers = num_workers > 0
        self._pin_memory = True

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
        return self._get_torch_dataloader(
            data_provider._get_training_dataset(),
            batch_size=batch_size, shuffle=True, mp_safe=data_provider.is_mp_safe())
    
    def create_validation_loader(self, batch_size, data_provider):
        return self._get_torch_dataloader(
            data_provider._get_validation_dataset(),
            batch_size=batch_size, shuffle=False, mp_safe=data_provider.is_mp_safe())
    
    def create_test_loader(self, batch_size, data_provider):
        return self._get_torch_dataloader(
            data_provider._get_test_dataset(),
            batch_size=batch_size, shuffle=False, mp_safe=data_provider.is_mp_safe())