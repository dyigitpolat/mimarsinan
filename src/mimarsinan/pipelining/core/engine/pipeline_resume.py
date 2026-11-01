"""Resume-from-step requirement resolution for :class:`Pipeline`.

A run asked to start at step ``X`` starts at the latest step index ``i`` at or before ``X`` that is a valid start:

* the entries the steps before ``i`` required and did not consume are in the cache (``get_all_requirements``, the
  precondition a direct start has always had), and
* every requirement of every step from ``i`` to ``X`` is in the cache or is written (promised or updated) by an
  earlier step at or after ``i``.

A start this returns is valid for itself, so resolving it again returns it (a fixed point): the step a session reports
is the step it executes, and the backward walk happens once. When no start is valid, and when a step's requirements are
absent as it runs, the pipeline refuses by name (:class:`MissingRequirementsError`), never by a bare assertion.
"""

from __future__ import annotations

from mimarsinan.common.refusal import ToolchainRefusal


class MissingRequirementsError(ToolchainRefusal, AssertionError):
    """Cache entries a step requires are absent, and no step that would run before it writes them.

    Names the step, the missing entries (as cache keys), the steps that write them and the remedy; for a resume that no
    start can feed it also names the requested step. Keeps the builtin base of the anonymous assertion it replaces.
    """

    def __init__(self, step_name: str, writers: dict, *, requested_step: str | None = None):
        self.step_name = step_name
        self.missing_keys = tuple(sorted(writers))
        self.writers = {key: tuple(writers[key]) for key in self.missing_keys}
        self.requested_step = requested_step
        written = "; ".join(
            f"{key} (written by {', '.join(repr(w) for w in self.writers[key]) or 'no step'})"
            for key in self.missing_keys
        )
        head = (f"cannot resume at {requested_step!r}: from every start at or before it, the run requires entries "
                f"that are neither in the cache nor written by an earlier step of the run, first at step {step_name!r}"
                if requested_step is not None else f"step {step_name!r} cannot run: the cache lacks its requirements")
        super().__init__(
            f"{head}: {written}. Remedy: rerun from the step that writes them (a start_step at or before it), or "
            "quarantine a stale entry (PipelineCache.quarantine_entry(<run dir>, <key>)) and rerun from its writer."
        )


def find_starting_step_idx(pipeline, step_name: str) -> int:
    requested = get_step_idx(pipeline, step_name)
    assert requested is not None, f"Step '{step_name}' does not exist in pipeline"
    writers = _writer_indices(pipeline)
    cached = set(pipeline.cache.keys())
    unmet: list = []
    for start in range(requested, -1, -1):
        unmet = unmet_requirements(pipeline, start, requested, writers, cached)
        if not unmet:
            if start != requested:
                print(f"Starting from '{pipeline.steps[start][0]}'")
            return start
        if start == requested:
            print(f"Cannot start from '{step_name}' because of missing requirements: {set(k for _, k in unmet)}")
            print("Finding the latest step that can be started from...")
    raise MissingRequirementsError(
        pipeline.steps[min(idx for idx, _ in unmet)][0], _writer_names(pipeline, writers, [k for _, k in unmet]),
        requested_step=step_name,
    )


def unmet_requirements(pipeline, start: int, end: int, writers: dict, cached: set) -> list:
    """``(step index, cache key)`` for every requirement a run from ``start`` through ``end`` would miss.

    A carried entry (required before ``start`` and not consumed there) is charged to ``start``; a requirement of a
    step ``s`` in range is met by the cache or by a writer at an index in ``[start, s)``.
    """
    unmet = {(start, key) for key in get_all_requirements(pipeline, pipeline.steps[start][0]) if key not in cached}
    for idx in range(start, end + 1):
        step = pipeline.steps[idx][1]
        for requirement in step.requires:
            key = pipeline._translate_key(step.name, requirement)
            if key not in cached and not any(start <= w < idx for w in writers.get(key, ())):
                unmet.add((idx, key))
    return sorted(unmet)


def assert_requirements_present(pipeline, step) -> None:
    """Refuse by name a step whose requirements are not in the cache as it is about to run."""
    missing = [key for key in (pipeline._translate_key(step.name, r) for r in step.requires)
               if key not in pipeline.cache]
    if missing:
        raise MissingRequirementsError(step.name, _writer_names(pipeline, _writer_indices(pipeline), missing))


def _writer_indices(pipeline) -> dict:
    """Cache key -> indices of the steps that write it (promise or update), as the engine names their outputs."""
    writers: dict = {}
    for idx, (_, step) in enumerate(pipeline.steps):
        for entry in (*step.promises, *step.updates):
            writers.setdefault(pipeline._create_real_key(step.name, entry), []).append(idx)
    return writers


def _writer_names(pipeline, writers: dict, keys) -> dict:
    return {key: [pipeline.steps[idx][0] for idx in writers.get(key, ())] for key in keys}


def get_all_requirements(pipeline, step_name: str) -> set:
    requirements = set()
    for name, step in pipeline.steps:
        for requirement in step.requires:
            requirements.add(pipeline._translate_key(step.name, requirement))

        if name == step_name:
            break

        for entry in step.updates:
            real_entry = pipeline._translate_key(step.name, entry)
            if real_entry in requirements:
                requirements.remove(real_entry)

        for entry in step.clears:
            real_entry = pipeline._create_real_key(step.name, entry)
            if real_entry in requirements:
                requirements.remove(real_entry)

    return requirements


def get_step_idx(pipeline, step_name: str) -> int | None:
    for idx, (name, _) in enumerate(pipeline.steps):
        if name == step_name:
            return idx
    return None
