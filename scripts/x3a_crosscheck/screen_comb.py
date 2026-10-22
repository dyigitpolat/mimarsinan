"""The model-level screen with the toolchain's encoder (proof check, part 2): every declaration the golden
generator draws gets ``encoder="comb"`` (the LIF memo's Uniform comb, equal to the toolchain's
``to_uniform_spikes``, see ``golden_vs_code.comb_matches_toolchain``) in place of the golden's even comb.

The workers fork (Linux default), so the patched generator reaches them. Usage, from ``scripts/``:
``python x3a_crosscheck/screen_comb.py --arms all --n-on 1000 --n-off 100 --workers 2 --out OUT.json``.
"""

from __future__ import annotations

import dataclasses
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x3a_golden import generator, screen  # noqa: E402

_draw = generator._declaration


def _comb_declaration(rng, dom):
    return dataclasses.replace(_draw(rng, dom), encoder="comb")


if __name__ == "__main__":
    assert multiprocessing.get_start_method() == "fork"
    generator._declaration = _comb_declaration
    raise SystemExit(screen.main(sys.argv[1:]))
