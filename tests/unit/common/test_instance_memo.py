"""instance_memo: per-instance derived values for unhashable hosts, without leaks."""

from __future__ import annotations

import gc

from mimarsinan.common.instance_memo import InstanceMemo, instance_memo, memo_size


class _Host:
    """Unhashable stand-in for IR dataclasses (eq without hash)."""

    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other):
        return isinstance(other, _Host)


def test_builds_once_and_caches():
    host = _Host()
    calls = []

    def build(h):
        calls.append(h)
        return {"value": 42}

    a = instance_memo(host, build)
    b = instance_memo(host, build)
    assert a is b
    assert calls == [host]


def test_distinct_instances_get_distinct_entries():
    h1, h2 = _Host(), _Host()
    v1 = instance_memo(h1, lambda h: object())
    v2 = instance_memo(h2, lambda h: object())
    assert v1 is not v2


def test_entry_evicted_when_host_dies():
    before = memo_size()
    host = _Host()
    instance_memo(host, lambda h: [1])
    assert memo_size() == before + 1
    del host
    gc.collect()
    assert memo_size() == before


def test_separate_memos_do_not_collide_on_one_host():
    host = _Host()
    kind_a: InstanceMemo = InstanceMemo()
    kind_b: InstanceMemo = InstanceMemo()
    a = kind_a.get(host, lambda h: "plan-a")
    b = kind_b.get(host, lambda h: "plan-b")
    assert (a, b) == ("plan-a", "plan-b")
    assert kind_a.get(host, lambda h: "never") == "plan-a"
