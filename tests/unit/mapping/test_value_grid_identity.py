"""With no Wave 2 grid declaration, every value program is the base's: IR, packed cores, NF, SCM, HCM and golden
outputs equal digests captured on the base 75998141 before the declarations were added
(``RA/11_wave2/lanes/identity/program_digests_base.json``, by ``RA/11_wave2/lanes/scripts/program_digests.py``).
Integration re-pin (tcad/wave1-merged, lanes onto pack): conv1d's packed cores are the pack head b777e3a1's, where the
pack unit's W04 class 6 (a neuron-split fragment keeps its parent's residency class) lets its six 7-row fragments share
one core, so it packs on 7 hard cores instead of 12; reverting class 6 alone restores the base's packed digest
(``RA/11_wave2/integration/fix_round/logs/conv1d_class6.json``); every output digest stays the base's
(``RA/11_wave2/integration/lanes/logs/identity_digests_by_head.jsonl``)."""

from __future__ import annotations

import pytest

from unit.mapping.value_grid_identity import IDENTITY_INSTANCES, program_digest

BASE_DIGESTS = {
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
