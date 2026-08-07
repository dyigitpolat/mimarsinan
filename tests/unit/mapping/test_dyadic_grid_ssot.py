"""The dyadic grid is the program's SSOT for "exactly representable, hence precision-invariant".

Grid membership is the predicate the certificate enforces as a precondition. Anything else that
needs to reason about exactness -- notably the per-fold certifiability split -- must consume THIS
predicate rather than carry a second copy of the idea, so the two can never drift apart.
"""

import numpy as np
import pytest

from mimarsinan.mapping.pruning.certificate import (
    DEFAULT_FRACTION_BITS,
    is_on_grid,
)


class TestGridMembershipIsExact:
    @pytest.mark.parametrize("bits", [1, 2, 4, 8, 16])
    def test_representable_values_are_on_grid_at_every_width(self, bits):
        step = 2.0 ** (-bits)
        values = np.array([0.0, step, -step, 1.0, -1.0, 3 * step, 2.0 ** 5])
        assert is_on_grid(values, fraction_bits=bits)

    @pytest.mark.parametrize("bits", [1, 2, 4, 8])
    def test_half_step_is_off_grid(self, bits):
        """The value exactly between two grid points is the tightest negative case."""
        half_step = 2.0 ** (-(bits + 1))
        assert not is_on_grid(np.array([half_step]), fraction_bits=bits)

    def test_zero_is_always_on_grid(self):
        """Why elimination (CONST(0)) always certifies while a GELU constant may not."""
        for bits in (1, 4, 8, 24):
            assert is_on_grid(np.array([0.0]), fraction_bits=bits)

    def test_non_finite_is_not_on_grid(self):
        for bad in (np.inf, -np.inf, np.nan):
            assert not is_on_grid(np.array([bad]), fraction_bits=DEFAULT_FRACTION_BITS)


class TestTrainedFloatsAreNotOnGrid:
    """This is the condition that actually blocks constant folding on a pristine transformer.

    The analysis refuses an operator when its input constants are not precision-invariant. On a
    pristine ViT the first constants are the trained ``cls_token`` / ``pos_embed`` parameters --
    arbitrary floats, essentially never on the grid -- so every op refuses. Zeroing them (the
    "twin") makes every constant 0.0, which the test above pins as always on-grid. That, and not
    any probe literal, is why the twin folded and the pristine graph did not.
    """

    def test_arbitrary_trained_floats_are_off_grid(self):
        rng = np.random.default_rng(0)
        trained = rng.normal(size=256).astype(np.float32).astype(np.float64)
        assert not is_on_grid(trained, fraction_bits=DEFAULT_FRACTION_BITS)

    def test_the_same_vector_zeroed_is_on_grid(self):
        rng = np.random.default_rng(0)
        trained = rng.normal(size=256).astype(np.float64)
        assert is_on_grid(np.zeros_like(trained), fraction_bits=DEFAULT_FRACTION_BITS)


class TestPredicateIsPubliclyReachable:
    def test_exported_from_the_certificate_package(self):
        """A second copy of this idea elsewhere is the drift we are preventing."""
        from mimarsinan.mapping.pruning import certificate

        assert "is_on_grid" in certificate.__all__
        assert "DEFAULT_FRACTION_BITS" in certificate.__all__
