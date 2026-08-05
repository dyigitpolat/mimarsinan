"""[wsm V3] transient weight residency: chain-local tensor lifetime, bit-stable outputs."""

import gc

import torch

from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
    assemble_segment_input_torch,
)
from mimarsinan.chip_simulation.value_run import (
    ValueHybridCoreFlow,
    run_neural_segment_values,
)
from unit.chip_simulation.test_value_flow_memory import (
    _identity_flow_and_fused,
    _token_bank_hybrid,
)
from unit.chip_simulation.value_upload_probe import probe_uploads

_RAW_INPUT_NODE_ID = -2

# Float-hex goldens captured from the PRE-transience flow (HEAD cfb5b16f):
# identical tensors, ops and order — only weight-tensor lifetime changed — so
# outputs must stay bit-identical. Identity vehicle: 5 stages / 3 neural
# chains; token vehicle: 3 neural stages in ONE residency chain.
_IDENTITY_FP64_GOLDEN = (
    '-0x1.26d300444ac20p-5', '-0x1.34df31962e60ap-3', '0x1.2fbf652ad1ebfp-5',
    '0x1.120f8fc8888a4p-2', '-0x1.9a9e27b94a77cp-4', '-0x1.f7a58928f827cp-5',
    '0x1.79229f075fe16p-5', '0x1.5322eca103745p-2', '0x1.8f0cbc197c804p-6',
    '-0x1.badc928a303bap-3', '-0x1.f90aab704e1bfp-6', '0x1.0a73ac8e344b6p-2',
)
_TOKEN_BANK_FP64_GOLDEN = (
    '-0x1.320c3ceeea383p-1', '0x1.6dc9c9dac6e92p-2', '-0x1.2f88bf1d97d17p+1',
    '-0x1.aa0966016d590p+0', '-0x1.abcd2b1f2f3e3p+0', '0x1.cf1beba08462cp+0',
    '-0x1.d42c1c8350d78p+0', '-0x1.15e067da64298p+2', '-0x1.6ebb02db3edaep+0',
    '-0x1.43a29c5bfe220p+0', '-0x1.292c53539a489p+1', '-0x1.616754b4678c0p-6',
    '-0x1.cb99100e31183p+0', '-0x1.a3932d9184ac6p+1', '-0x1.e0ff43176d9aap+0',
    '0x1.b43560775e4fcp+0', '-0x1.bb1102eb99accp+0', '0x1.987782f1489d6p-2',
    '-0x1.faa6f116bca64p+0', '-0x1.7577156ef88b4p+1', '-0x1.345b21d8f232ap+0',
    '-0x1.71c8e30ee28f8p-3', '-0x1.0543384db70bfp+1', '-0x1.6baf8d7ef750bp+0',
    '-0x1.2eae4aea352fbp+0', '-0x1.10c769df1fcf1p+1', '-0x1.b86e5f97263bfp+0',
    '0x1.18a51e0b8882cp-2', '-0x1.4d8de1a411562p-1', '0x1.1038916cb2b99p-1',
    '-0x1.39ce301902bfap+1', '-0x1.8b729a49552b5p+1', '-0x1.3dc8ac81d22e8p+0',
    '-0x1.21a868d8c264dp+1', '-0x1.a21232c8e3244p+0', '0x1.5d9be1663bca8p-2',
    '-0x1.9a9e360bb8b85p+0', '-0x1.e957b3c7acc7cp+0', '-0x1.6567780719eb6p+0',
    '0x1.18233b136eb80p+0',
)
_TOKEN_BANK_FP32_GOLDEN = (
    '-0x1.320c3e0000000p-1', '0x1.6dc9c60000000p-2', '-0x1.2f88c00000000p+1',
    '-0x1.aa09660000000p+0', '-0x1.abcd2c0000000p+0', '0x1.cf1bea0000000p+0',
    '-0x1.d42c1c0000000p+0', '-0x1.15e0680000000p+2', '-0x1.6ebb040000000p+0',
    '-0x1.43a29c0000000p+0', '-0x1.292c540000000p+1', '-0x1.6168000000000p-6',
    '-0x1.cb99100000000p+0', '-0x1.a3932e0000000p+1', '-0x1.e0ff420000000p+0',
    '0x1.b435620000000p+0', '-0x1.bb11040000000p+0', '0x1.9877860000000p-2',
    '-0x1.faa6f00000000p+0', '-0x1.7577180000000p+1', '-0x1.345b220000000p+0',
    '-0x1.71c8e40000000p-3', '-0x1.0543380000000p+1', '-0x1.6baf8e0000000p+0',
    '-0x1.2eae4c0000000p+0', '-0x1.10c76a0000000p+1', '-0x1.b86e600000000p+0',
    '0x1.18a5200000000p-2', '-0x1.4d8de00000000p-1', '0x1.1038900000000p-1',
    '-0x1.39ce300000000p+1', '-0x1.8b729a0000000p+1', '-0x1.3dc8ac0000000p+0',
    '-0x1.21a8680000000p+1', '-0x1.a212320000000p+0', '0x1.5d9be80000000p-2',
    '-0x1.9a9e360000000p+0', '-0x1.e957b20000000p+0', '-0x1.6567780000000p+0',
    '0x1.18233a0000000p+0',
)


def _hexes(t) -> tuple:
    return tuple(float(v).hex() for v in t.detach().flatten().tolist())


class TestBitIdenticalOutputs:
    def test_identity_multi_segment_fp64_matches_pre_change_hex(self):
        flow, _fused = _identity_flow_and_fused()
        torch.manual_seed(1234)
        x = torch.randn(3, 8)
        with torch.no_grad():
            out = flow(x)
        assert _hexes(out) == _IDENTITY_FP64_GOLDEN

    def test_token_bank_residency_fp64_matches_pre_change_hex(self):
        hybrid = _token_bank_hybrid()
        torch.manual_seed(4321)
        x = torch.randn(2, 20)
        with torch.no_grad():
            out = ValueHybridCoreFlow(hybrid, dtype=torch.float64)(x)
        assert _hexes(out) == _TOKEN_BANK_FP64_GOLDEN

    def test_token_bank_residency_fp32_matches_pre_change_hex(self):
        hybrid = _token_bank_hybrid()
        torch.manual_seed(4321)
        x = torch.randn(2, 20)
        with torch.no_grad():
            out = ValueHybridCoreFlow(hybrid, dtype=torch.float32)(x)
        assert _hexes(out) == _TOKEN_BANK_FP32_GOLDEN

    def test_repeated_forwards_are_bitwise_stable(self):
        # Re-uploading the same ndarrays through the same torch.as_tensor
        # calls yields bitwise-equal tensors: forwards are exactly repeatable.
        flow, _fused = _identity_flow_and_fused()
        torch.manual_seed(99)
        x = torch.randn(3, 8)
        with torch.no_grad():
            first = flow(x)
            second = flow(x)
        assert torch.equal(first, second)


class TestTransientWeightResidency:
    def test_peak_resident_weight_bytes_is_one_chain(self):
        # Peak live weight bytes = the LARGEST single chain, never the sum
        # over segments: equality pins clear-before-prepare, strict < proves
        # each chain is freed when the next one begins.
        flow, _fused = _identity_flow_and_fused()
        neural = [
            s for s in flow.hybrid_mapping.stages if s.kind == "neural"
        ]
        assert len(neural) >= 2
        assert not any(
            getattr(s, "schedule_weights_resident", False) for s in neural
        )
        with probe_uploads(flow) as probe:
            torch.manual_seed(1234)
            with torch.no_grad():
                flow(torch.randn(3, 8))
        chain_bytes = probe.per_chain_uploaded_bytes()
        assert len(chain_bytes) == len(neural)
        assert all(b > 0 for b in chain_bytes)
        peak = max(live for _stage, _n, live in probe.stage_marks)
        assert peak == max(chain_bytes)
        assert peak < sum(chain_bytes)

    def test_uploaded_tensors_die_with_the_forward(self):
        hybrid = _token_bank_hybrid()
        flow = ValueHybridCoreFlow(hybrid, dtype=torch.float64)
        with probe_uploads(flow) as probe:
            torch.manual_seed(4321)
            with torch.no_grad():
                flow(torch.randn(2, 20))
        assert probe.records
        gc.collect()
        assert all(ref() is None for _core, ref, _n, _p in probe.records)

    def test_standalone_run_uses_a_per_call_scope(self):
        # scope=None means single-call lifetime: no cross-call caching, and
        # nothing outlives the call.
        hybrid = _token_bank_hybrid()
        head = [s for s in hybrid.stages if s.kind == "neural"][0]
        torch.manual_seed(4321)
        x = torch.randn(2, 20, dtype=torch.float64)
        seg_input = assemble_segment_input_torch(
            head.input_map, {_RAW_INPUT_NODE_ID: x}, 2,
            torch.device("cpu"), torch.float64,
        )
        with probe_uploads() as probe:
            with torch.no_grad():
                first = run_neural_segment_values(
                    head.hard_core_mapping, seg_input
                )
                second = run_neural_segment_values(
                    head.hard_core_mapping, seg_input
                )
        assert torch.equal(first, second)
        n_cores = len(head.hard_core_mapping.cores)
        assert len(probe.records) == 2 * n_cores
        gc.collect()
        assert all(ref() is None for _core, ref, _n, _p in probe.records)
