"""Resume-from-step requirement resolution for :class:`Pipeline`."""

from __future__ import annotations


def find_starting_step_idx(pipeline, step_name: str) -> int:
    requirements = get_all_requirements(pipeline, step_name)
    missing_requirements = requirements - set(pipeline.cache.keys())

    if len(missing_requirements) > 0:
        print(f"Cannot start from '{step_name}' because of missing requirements: {missing_requirements}")
        starting_step_idx = find_latest_possible_step_idx(
            pipeline, missing_requirements, step_name
        )
    else:
        starting_step_idx = get_step_idx(pipeline, step_name)

    return starting_step_idx


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


def find_latest_possible_step_idx(pipeline, missing_requirements: set, step_name: str) -> int:
    print("Finding the earliest step that can be started from...")

    starting_step_idx = None

    begin_idx = get_step_idx(pipeline, step_name)
    for idx in range(begin_idx - 1, -1, -1):
        name = pipeline.steps[idx][0]
        step = pipeline.steps[idx][1]

        for promise in step.promises:
            real_promise = pipeline._create_real_key(step.name, promise)
            if real_promise in missing_requirements:
                missing_requirements.remove(real_promise)

        for entry in step.updates:
            real_entry = pipeline._create_real_key(step.name, entry)
            if real_entry in missing_requirements:
                missing_requirements.remove(real_entry)

        if len(missing_requirements) == 0:
            starting_step_idx = idx
            print(f"Starting from '{name}'")
            break

    assert starting_step_idx is not None
    return starting_step_idx


def get_step_idx(pipeline, step_name: str) -> int | None:
    for idx, (name, _) in enumerate(pipeline.steps):
        if name == step_name:
            return idx
    return None
