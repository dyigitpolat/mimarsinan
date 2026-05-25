"""``server._get_layout_result_from_request`` must route through the cached
``DEFAULT_LAYOUT_MAPPING_SERVICE`` so repeated wizard requests with identical
bodies don't rebuild the mapper graph from scratch."""

from __future__ import annotations


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


def test_two_identical_requests_share_underlying_softcores() -> None:
    from mimarsinan.gui.server import _get_layout_result_from_request
    from mimarsinan.mapping.layout_mapping_service import (
        DEFAULT_LAYOUT_MAPPING_SERVICE,
    )

    DEFAULT_LAYOUT_MAPPING_SERVICE.invalidate()
    a = _get_layout_result_from_request(_wizard_body())
    b = _get_layout_result_from_request(_wizard_body())
    assert a is b


def test_verify_and_softcores_helpers_share_cache() -> None:
    from mimarsinan.gui.server import (
        _get_layout_result_from_request,
        _get_softcores_from_request,
    )
    from mimarsinan.mapping.layout_mapping_service import (
        DEFAULT_LAYOUT_MAPPING_SERVICE,
    )

    DEFAULT_LAYOUT_MAPPING_SERVICE.invalidate()
    result = _get_layout_result_from_request(_wizard_body())
    softcores = _get_softcores_from_request(_wizard_body())
    assert softcores is result.softcores


def test_different_tiling_reuses_model_repr() -> None:
    """Two requests differing only on tiling parameters must share the
    model_repr cache slot -- only the verification slot differs."""
    from mimarsinan.gui.server import _get_layout_result_from_request
    from mimarsinan.mapping.layout_mapping_service import (
        DEFAULT_LAYOUT_MAPPING_SERVICE,
    )
    from mimarsinan.mapping.layout_request import LayoutMappingRequest

    DEFAULT_LAYOUT_MAPPING_SERVICE.invalidate()
    body_a = _wizard_body()
    body_b = dict(body_a)
    body_b["max_axons"] = 8192
    body_b["max_neurons"] = 8192

    _get_layout_result_from_request(body_a)
    _get_layout_result_from_request(body_b)

    req_a = LayoutMappingRequest.from_wizard_body(body_a)
    req_b = LayoutMappingRequest.from_wizard_body(body_b)
    assert req_a.model_identity_key() == req_b.model_identity_key()
    assert DEFAULT_LAYOUT_MAPPING_SERVICE.get_model_repr(req_a) is (
        DEFAULT_LAYOUT_MAPPING_SERVICE.get_model_repr(req_b)
    )
