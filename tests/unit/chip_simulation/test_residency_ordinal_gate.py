"""[wsm V3] the residency alias must be per-ORDINAL, and that must be provable.

The pre-existing residency goldens all ran on a SINGLE-bank vehicle, where
every resident core holds the same payload — so reversing
``resident_from.weights[:n]`` passed the entire suite (verified by mutation).
This gate closes that hole with a chain whose ordinals carry DIFFERENT banks
(distinct matrices and distinct hardware biases), plus the mutation proof that
the golden actually fails when the ordinals are permuted.
"""

import torch

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow

from unit.chip_simulation.heterogeneous_residency_vehicle import (
    N_BANKS,
    build_heterogeneous_residency_mapping,
)

# Captured on HEAD 7e629a80 from the heterogeneous chain (2 banks x 3
# instances, 3 passes, one residency chain).
_HETEROGENEOUS_FP64_GOLDEN = (
    '0x1.4253cbaece264p+2', '0x1.3a9347e172340p+3', '0x1.fcff3772e64fcp+1',
    '0x1.0c1b3b41f3e92p+3', '0x1.d5d5ec6af1126p+3', '0x1.e13e73a54f140p+3',
    '0x1.0301a342a2160p+4', '0x1.11fdb642b1a0fp+4', '-0x1.389d9b60960c8p+0',
    '-0x1.2ac571e427c04p+1', '-0x1.0f661eb732678p+1', '-0x1.4d1264a060248p+0',
    '0x1.8b1e6ce83bc74p+3', '0x1.5fc3061972fddp+3', '0x1.7cf86d73638b9p+3',
    '0x1.c27d78335b33cp+3', '0x1.7859e484c1f5cp+2', '0x1.dbab22f20c510p+0',
    '0x1.6db6a6684bac0p+0', '0x1.8b41516bf07b0p+2', '-0x1.9e7cbd8f7d29cp+4',
    '-0x1.7aa76f35fffadp+4', '-0x1.9da792a1f3b60p+4', '-0x1.7978594efa54ap+4',
    '-0x1.3952baeb36f74p+5', '-0x1.44763f8c33aaep+5', '-0x1.2fefbc4af6492p+5',
    '-0x1.289d2371418f2p+5', '-0x1.0b85d43a94a9bp+5', '-0x1.c375ba16a0a6cp+4',
    '-0x1.0758f08326e2ep+5', '-0x1.f0ae22c6ba650p+4', '0x1.fb55501a71322p+3',
    '0x1.bf605e73e9708p+3', '0x1.478328e03d32cp+4', '0x1.24b68d32c3a64p+4',
    '0x1.25905ef020c3fp+5', '0x1.3f8e96dafd3c1p+5', '0x1.3ec9d65ce6172p+5',
    '0x1.363b903dc7761p+5', '0x1.7ee026d03238cp+5', '0x1.85062c060e05cp+5',
    '0x1.88b7570adcfe9p+5', '0x1.88e11630c91bcp+5', '0x1.25a1b91e7a1bbp+6',
    '0x1.32a57562d3274p+6', '0x1.2d1b242c03abep+6', '0x1.2a35bd3972218p+6',
)


def _run(dtype=torch.float64):
    flow = ValueHybridCoreFlow(build_heterogeneous_residency_mapping(), dtype=dtype)
    torch.manual_seed(11)
    x = torch.randn(2, 24, dtype=dtype)
    return flow(x)


class TestOrdinalsAreDistinctAndPinned:
    def test_chain_ordinals_carry_different_payloads(self):
        """Without this, the golden below could not detect a permutation."""
        hm = build_heterogeneous_residency_mapping()
        neural = [s for s in hm.stages if s.hard_core_mapping is not None]
        resident = [s for s in neural if getattr(s, "schedule_weights_resident", False)]
        assert len(neural) >= 2 and resident, "vehicle must form a residency chain"
        for stage in neural:
            cores = stage.hard_core_mapping.cores
            assert len(cores) == N_BANKS
            keys = {c.core_matrix_key() for c in cores}
            assert len(keys) == len(cores), (
                "ordinals must hold DISTINCT weight payloads or an ordinal "
                "permutation is undetectable"
            )
            biases = {float(c.hardware_bias[0]) for c in cores}
            assert len(biases) == len(cores), "ordinals must hold distinct biases"

    def test_output_matches_golden(self):
        got = tuple(float(v).hex() for v in _run().flatten())
        assert got == _HETEROGENEOUS_FP64_GOLDEN


class TestTheOrdinalGateCanFail:
    """Permute the resident alias and prove the golden rejects it."""

    def test_reversed_resident_alias_is_caught(self, monkeypatch):
        import mimarsinan.chip_simulation.value_run.value_execution as ve

        real_init = ve._PreparedValueSegment.__init__

        def sabotaged(self, hcm, device, dtype, resident_from=None, **kw):
            real_init(self, hcm, device, dtype, resident_from=resident_from, **kw)
            if resident_from is not None:      # the aliasing branch under test
                self.weights = list(reversed(self.weights))
                self.biases = list(reversed(self.biases))

        monkeypatch.setattr(ve._PreparedValueSegment, "__init__", sabotaged)
        got = tuple(float(v).hex() for v in _run().flatten())
        assert got != _HETEROGENEOUS_FP64_GOLDEN, (
            "reversing the per-ordinal resident alias must change the output; "
            "if it does not, this gate proves nothing"
        )
