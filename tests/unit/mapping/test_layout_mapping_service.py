"""Tests for ``LayoutMappingService`` and ``LayoutMappingRequest``."""

from __future__ import annotations

import threading

import pytest


def _wizard_body() -> dict:
    return {
        "model_type": "simple_mlp",
        "input_shape": [1, 28, 28],
        "num_classes": 10,
        "model_config": {"mlp_width_1": 64, "mlp_width_2": 32},
        "max_axons": 4096,
        "max_neurons": 4096,
        "allow_coalescing": False,
        "hardware_bias": True,
        "target_tq": 32,
    }


# LayoutMappingRequest


def test_request_is_frozen() -> None:
    from dataclasses import FrozenInstanceError
    from mimarsinan.mapping.verification.layout_request import LayoutMappingRequest

    r = LayoutMappingRequest.from_wizard_body(_wizard_body())
    with pytest.raises(FrozenInstanceError):
        r.max_axons = 8192  # type: ignore[misc]


def test_request_hashable_and_equal_for_same_body() -> None:
    from mimarsinan.mapping.verification.layout_request import LayoutMappingRequest

    a = LayoutMappingRequest.from_wizard_body(_wizard_body())
    b = LayoutMappingRequest.from_wizard_body(_wizard_body())
    assert a == b
    assert hash(a) == hash(b)


def test_request_distinguishes_each_field() -> None:
    from mimarsinan.mapping.verification.layout_request import LayoutMappingRequest

    base = _wizard_body()
    a = LayoutMappingRequest.from_wizard_body(base)
    for k, alt in {
        "model_type": "torch_vit",
        "num_classes": 100,
        "target_tq": 64,
        "max_axons": 8192,
        "max_neurons": 8192,
        "allow_coalescing": True,
        "hardware_bias": False,
    }.items():
        body = dict(base)
        body[k] = alt
        b = LayoutMappingRequest.from_wizard_body(body)
        assert a != b, f"flipping {k!r} did not change the request"
        assert hash(a) != hash(b), f"flipping {k!r} did not change the hash"


def test_request_model_config_change_changes_hash() -> None:
    from mimarsinan.mapping.verification.layout_request import LayoutMappingRequest

    base = _wizard_body()
    a = LayoutMappingRequest.from_wizard_body(base)
    b_body = dict(base)
    b_body["model_config"] = {"hidden_dims": [32, 32]}
    b = LayoutMappingRequest.from_wizard_body(b_body)
    assert a != b


def test_request_input_shape_change_changes_hash() -> None:
    from mimarsinan.mapping.verification.layout_request import LayoutMappingRequest

    base = _wizard_body()
    a = LayoutMappingRequest.from_wizard_body(base)
    b_body = dict(base)
    b_body["input_shape"] = [3, 32, 32]
    b = LayoutMappingRequest.from_wizard_body(b_body)
    assert a != b


def test_model_identity_key_invariant_under_tiling_changes() -> None:
    from mimarsinan.mapping.verification.layout_request import LayoutMappingRequest

    base = _wizard_body()
    a = LayoutMappingRequest.from_wizard_body(base)
    b_body = dict(base)
    b_body["max_axons"] = 8192
    b_body["max_neurons"] = 8192
    b_body["allow_coalescing"] = True
    b_body["hardware_bias"] = not base["hardware_bias"]
    b = LayoutMappingRequest.from_wizard_body(b_body)
    assert a.model_identity_key() == b.model_identity_key()
    assert a.verification_key() != b.verification_key()


# LayoutMappingService


def _build_service(maxsize: int = 16):
    from mimarsinan.mapping.verification.layout_mapping_service import LayoutMappingService
    return LayoutMappingService(
        model_repr_maxsize=maxsize, verification_maxsize=maxsize,
    )


def test_service_caches_model_repr_across_calls() -> None:
    from mimarsinan.mapping.verification.layout_request import LayoutMappingRequest

    svc = _build_service()
    req = LayoutMappingRequest.from_wizard_body(_wizard_body())
    a = svc.get_model_repr(req)
    b = svc.get_model_repr(req)
    assert a is b


def test_service_verification_cache_returns_same_softcores() -> None:
    from mimarsinan.mapping.verification.layout_request import LayoutMappingRequest

    svc = _build_service()
    req = LayoutMappingRequest.from_wizard_body(_wizard_body())
    a = svc.get_verification(req)
    b = svc.get_verification(req)
    assert a is b
    assert a.softcores == b.softcores


def test_service_separate_model_repr_and_verification_caches() -> None:
    from mimarsinan.mapping.verification.layout_request import LayoutMappingRequest

    svc = _build_service()
    body_a = _wizard_body()
    body_b = dict(body_a)
    body_b["max_axons"] = 8192
    req_a = LayoutMappingRequest.from_wizard_body(body_a)
    req_b = LayoutMappingRequest.from_wizard_body(body_b)

    repr_a = svc.get_model_repr(req_a)
    repr_b = svc.get_model_repr(req_b)
    assert repr_a is repr_b, (
        "Same model-identity should share the model_repr slot regardless "
        "of tiling parameters"
    )
    v_a = svc.get_verification(req_a)
    v_b = svc.get_verification(req_b)
    assert v_a is not v_b


def test_service_lru_eviction_at_maxsize() -> None:
    from mimarsinan.mapping.verification.layout_request import LayoutMappingRequest

    svc = _build_service(maxsize=2)

    bodies = []
    reqs = []
    for w in (16, 32, 64):
        body = _wizard_body()
        body["model_config"] = {"mlp_width_1": w, "mlp_width_2": w}
        bodies.append(body)
        reqs.append(LayoutMappingRequest.from_wizard_body(body))

    svc.get_verification(reqs[0])
    svc.get_verification(reqs[1])
    svc.get_verification(reqs[2])

    # req[0] should have been evicted -- re-fetching must build a fresh repr
    repr_before = svc.get_model_repr(reqs[2])
    repr_after_evict = svc.get_model_repr(reqs[0])
    assert repr_after_evict is not None
    # The third slot is still in cache (most recent two)
    assert svc.get_model_repr(reqs[2]) is repr_before


def test_service_invalidate_clears_both_caches() -> None:
    from mimarsinan.mapping.verification.layout_request import LayoutMappingRequest

    svc = _build_service()
    req = LayoutMappingRequest.from_wizard_body(_wizard_body())
    a = svc.get_model_repr(req)
    svc.invalidate()
    b = svc.get_model_repr(req)
    assert a is not b


def test_service_thread_safe_concurrent_first_calls() -> None:
    from mimarsinan.mapping.verification.layout_request import LayoutMappingRequest

    svc = _build_service()
    req = LayoutMappingRequest.from_wizard_body(_wizard_body())
    results: list = []
    barrier = threading.Barrier(8)

    def _worker() -> None:
        barrier.wait()
        results.append(svc.get_verification(req))

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 8
    first = results[0]
    for r in results[1:]:
        assert r is first
