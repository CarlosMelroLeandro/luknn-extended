import torch
import pytest
from luknn.network.crystallization import (
    smooth_crystallize, crisp_crystallize_weights as crisp_crystallize, representation_error
)


def t(v): return torch.tensor(v, dtype=torch.float32)


class TestSmoothCrystallize:
    def test_integers_are_fixed_points(self):
        # Υ_n(w) should map integers to themselves
        for v in [-1.0, 0.0, 1.0]:
            result = smooth_crystallize(t(v), n=2).item()
            assert result == pytest.approx(v, abs=1e-5)

    def test_output_converges_toward_integers(self):
        w = t(0.4)
        prev = w.item()
        for _ in range(20):
            w = smooth_crystallize(w, n=2)
        # After repeated application, should be close to 0
        assert abs(w.item()) < 0.01

    def test_sign_preserved(self):
        w = t(-0.6)
        result = smooth_crystallize(w, n=2).item()
        assert result < 0


class TestCrispCrystallize:
    def test_rounds_to_nearest(self):
        # crisp_crystallize_weights uses round (not floor) then clamp to {-1,0,1}
        assert crisp_crystallize(t(0.9)).item() == pytest.approx(1.0)
        assert crisp_crystallize(t(0.4)).item() == pytest.approx(0.0)
        assert crisp_crystallize(t(1.0)).item() == pytest.approx(1.0)
        assert crisp_crystallize(t(-0.6)).item() == pytest.approx(-1.0)

    def test_clamp(self):
        assert crisp_crystallize(t(2.7)).item() == pytest.approx(1.0)
        assert crisp_crystallize(t(-3.0)).item() == pytest.approx(-1.0)


class TestRepresentationError:
    def test_integer_weights_zero_error(self):
        w = torch.tensor([-1.0, 0.0, 1.0])
        assert representation_error(w).item() == pytest.approx(0.0)

    def test_fractional_weights(self):
        w = torch.tensor([0.5, 0.5])
        assert representation_error(w).item() == pytest.approx(1.0)
