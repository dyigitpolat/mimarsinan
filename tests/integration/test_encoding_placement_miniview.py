"""Phase 5 regression: the wizard layout/miniview path honours the Encoding
Layer Placement (subsume vs offload) selection.

Previously the placement was dropped before reaching the layout verifier, so
switching subsume -> offload left the mapping miniview unchanged.  These tests
pin that (1) the request cache key distinguishes the two placements and (2) the
model repr the layout path builds maps more perceptrons on-chip under offload.
"""

from __future__ import annotations

from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.verification.layout_request import LayoutMappingRequest
from mimarsinan.mapping.verification.wizard_layout_verify import (
    model_repr_from_wizard_body,
)


def _body(placement: str) -> dict:
    return {
        "model_type": "mlp_mixer_core",
        "input_shape": [1, 28, 28],
        "num_classes": 10,
        "model_config": {
            "patch_n_1": 4, "patch_m_1": 4, "patch_c_1": 6,
            "fc_w_1": 8, "fc_w_2": 6, "base_activation": "ReLU",
        },
        "max_axons": 512,
        "max_neurons": 512,
        "encoding_layer_placement": placement,
    }


def test_request_key_distinguishes_placement():
    sub = LayoutMappingRequest.from_wizard_body(_body("subsume"))
    off = LayoutMappingRequest.from_wizard_body(_body("offload"))

    assert sub.encoding_layer_placement == "subsume"
    assert off.encoding_layer_placement == "offload"
    # Placement changes the mapper graph, so it must split both cache levels.
    assert sub.model_identity_key() != off.model_identity_key()
    assert sub.verification_key() != off.verification_key()
    # to_body round-trips the placement back to the repr builder.
    assert sub.to_body()["encoding_layer_placement"] == "subsume"
    assert off.to_body()["encoding_layer_placement"] == "offload"


def _neural_core_count(body: dict) -> int:
    repr_ = model_repr_from_wizard_body(body)
    ir = IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=512, max_neurons=512,
    ).map(repr_)
    return sum(isinstance(n, NeuralCore) for n in ir.nodes)


def test_offload_maps_more_cores_on_chip_in_layout_path():
    n_sub = _neural_core_count(_body("subsume"))
    n_off = _neural_core_count(_body("offload"))
    assert n_off > n_sub, (
        f"offload should map the encoding layer on-chip "
        f"(subsume={n_sub} cores, offload={n_off} cores)"
    )
