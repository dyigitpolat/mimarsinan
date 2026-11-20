"""With no Wave 2 grid declaration, every value program is the base's: IR, packed cores, NF, SCM, HCM and golden
outputs equal digests captured on the base 75998141 before the declarations were added
(``RA/11_wave2/lanes/identity/program_digests_base.json``, by ``RA/11_wave2/lanes/scripts/program_digests.py``).
Integration re-pin (tcad/wave1-merged, lanes onto pack): conv1d's packed cores are the pack head b777e3a1's, where the
pack unit's W04 class 6 (a neuron-split fragment keeps its parent's residency class) lets its six 7-row fragments share
one core, so it packs on 7 hard cores instead of 12; reverting class 6 alone restores the base's packed digest
(``RA/11_wave2/integration/fix_round/logs/conv1d_class6.json``); every output digest stays the base's
(``RA/11_wave2/integration/lanes/logs/identity_digests_by_head.jsonl``).

[U-BRANGE] Re-pin: every boundary scale is the range rule's (the least squared error of its own grid), so the programs
are pinned under that rule (``BASE_DIGESTS``; [U-BRANGE2] the whole interval up to the calibration maximum). With the
boundary scales put back at the former 0.999 quantile (read by the tests' own reference), every program is the earlier
base byte for byte (``QUANTILE_RULE_DIGESTS``, the digests captured on the base): the rule moves the boundary scales
and nothing else. Under revision 1's interval conv1d's NF digest differed from its identity digest by one signed zero
(the identity program writes -0.0 where NF writes 0.0: equal values); under the whole interval conv1d's are equal and
pruned_bank_conv's differ so, by one signed zero (equal values, read by the build's probe q5_nf_identity.py).

[U-BRANGE2 stage 1b] Re-pin (the stage verify's m-5): the programs' readouts are calibrated by Readout Law's own
calibration (``calibrate_readout_quantizers``: the rule at every readout, read-together grids pooled, no floor) where
the fixture had read the former 0.999 quantile with a 1e-3 floor. ``BASE_DIGESTS`` are the programs under the rule at
both the boundaries and the readouts; with the former quantile put back at both through the tests' own reference, every
program is still the earlier base byte for byte (``QUANTILE_RULE_DIGESTS``, unchanged: the floor moves no program).
conv1d's NF and identity outputs now differ by one signed zero (NF 0.0 where the identity program writes -0.0, equal
values), pruned_bank_conv's are byte-equal (this step's probe nf_identity.py)."""

from __future__ import annotations

import pytest

import mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step as boundary_step
import mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step as readout_step
from unit.mapping.value_grid_identity import IDENTITY_INSTANCES, program_digest
from unit.pipelining.range_rule_reference import former_rule

BASE_DIGESTS = {
    "mlp": {"ir": "689af367aa418b1d", "packed": "59c2b483490abb93", "identity_out": "f18d4b7f9b329326",
        "packed_out": "f18d4b7f9b329326", "nf_out": "f18d4b7f9b329326", "golden": "85444757d775028da3fa3668b3198e88"},
    "lenet5": {"ir": "3288df7399818388", "packed": "764f40527d3ee9a0", "identity_out": "2732e32eb669a8eb",
        "packed_out": "2732e32eb669a8eb", "nf_out": "2732e32eb669a8eb", "golden": "e5b08cafa989eaf05b03f2e73b37092c"},
    "residual_block": {"ir": "f1c245613dfde256", "packed": "64551c9ca054e79d", "identity_out": "c6b32ab760765998",
        "packed_out": "c6b32ab760765998", "nf_out": "c6b32ab760765998", "golden": "8e5638587818792d078f56586c323aad"},
    "conv1d": {"ir": "920892dae0b13a5d", "packed": "bf1bd1be9e42e3cd", "identity_out": "1387f43c457da72f",
        "packed_out": "1387f43c457da72f", "nf_out": "39ef826f091f2f69", "golden": "8c89c096330170a896ba54cd474d4740"},
    "pruned_bank_conv": {"ir": "7917392f9fc9db9c", "packed": "230b12d3b706e5ee", "identity_out": "d2cc0c6c4204388c",
        "packed_out": "d2cc0c6c4204388c", "nf_out": "d2cc0c6c4204388c", "golden": "c2dec9cace90a7b9fbf4fe7367bfcbe6"},
}

QUANTILE_RULE_DIGESTS = {
    "mlp": {"ir": "d75be8c57e674d84", "packed": "497da9f927e573ad", "identity_out": "e655aeeeb2584699",
            "packed_out": "e655aeeeb2584699", "nf_out": "e655aeeeb2584699",
            "golden": "962a177a47bcfcea2cdea1ecdfae5f98"},
    "lenet5": {"ir": "f52c534a93279e12", "packed": "64622113e459c05e", "identity_out": "b8d58913ef715115",
               "packed_out": "b8d58913ef715115", "nf_out": "b8d58913ef715115",
               "golden": "e1d4e84196e9a01eedaac08b4f340e2a"},
    "residual_block": {"ir": "ce3677088b9d0c07", "packed": "e2ad866e56906c4e", "identity_out": "bcbbd07c6cc37591",
                       "packed_out": "bcbbd07c6cc37591", "nf_out": "bcbbd07c6cc37591",
                       "golden": "2df7d538e308165c747f8b32bb12dc0f"},
    "conv1d": {"ir": "83c1df3ad4c0d878", "packed": "ddbe3e0bcddc1a5b", "identity_out": "e8c49e4a5cc1e60b",
               "packed_out": "e8c49e4a5cc1e60b", "nf_out": "e8c49e4a5cc1e60b",
               "golden": "e1f1db4d0eb456d8a1f0a7872eb488a0"},
    "pruned_bank_conv": {"ir": "d57975d1d8315f13", "packed": "cacaa779c96f7b34",
                         "identity_out": "d05b74fc3bac9f9b", "packed_out": "d05b74fc3bac9f9b",
                         "nf_out": "d05b74fc3bac9f9b", "golden": "37fbd23a1f53f159b837f060b7709b1f"},
}


@pytest.mark.parametrize("name", IDENTITY_INSTANCES)
def test_an_undeclared_program_is_the_base_program(name):
    assert program_digest(name) == BASE_DIGESTS[name]


@pytest.mark.parametrize("name", IDENTITY_INSTANCES)
def test_the_former_quantile_restores_the_earlier_base_byte_for_byte(name, monkeypatch):
    """[U-BRANGE2] The former rule read through the tests' own reference (``range_rule_reference.former_rule``: each
    site's largest per-batch 0.999 quantile), never through the toolchain: [stage 1b] at the boundaries and at the
    readouts, both of which the programs now calibrate by the steps' own rule."""
    monkeypatch.setattr(boundary_step, "least_squares_ranges", former_rule(boundary_step.least_squares_ranges))
    monkeypatch.setattr(readout_step, "least_squares_pooled", former_rule(readout_step.least_squares_pooled,
                                                                          pooled=True))
    assert program_digest(name) == QUANTILE_RULE_DIGESTS[name]
