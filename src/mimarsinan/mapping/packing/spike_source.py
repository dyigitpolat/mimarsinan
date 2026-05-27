"""Shared spike-source index predicates for packing modules."""


def is_off(idx):
    return idx == -1


def is_input(idx):
    return idx == -2


def is_always_on(idx):
    return idx == -3


def source_is_off(source) -> bool:
    return bool(getattr(source, "is_off_", False))


def source_is_input(source) -> bool:
    return bool(getattr(source, "is_input_", False))


def source_is_always_on(source) -> bool:
    return bool(getattr(source, "is_always_on_", False))


def source_is_special(source) -> bool:
    return source_is_off(source) or source_is_input(source) or source_is_always_on(source)
