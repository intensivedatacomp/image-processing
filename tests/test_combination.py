"""Unit tests for combination functions."""

import pytest
import torch

from image_processing.combination import (
    max_abs,
    sum_of_abs,
    sum_of_powers,
    sum_of_squares,
)


def _response(c: int, n: int, h: int, w: int) -> torch.Tensor:
    """Return a deterministic (C, N, H, W) response tensor."""
    torch.manual_seed(0)
    return torch.randn(c, n, h, w)


class TestSumOfSquares:
    """Tests for sum_of_squares combination function."""

    def test_output_shape(self) -> None:
        r = _response(3, 10, 64, 64)
        assert sum_of_squares(r).shape == torch.Size([64, 64])

    def test_single_element_correctness(self) -> None:
        r = torch.tensor([[[[3.0]]]])  # (1, 1, 1, 1)
        assert sum_of_squares(r).item() == pytest.approx(9.0)

    def test_multidim_correctness(self) -> None:
        # One channel, one kernel — result should be element-wise square
        r = torch.tensor([[[[1.0, -2.0], [3.0, -4.0]]]])  # (1, 1, 2, 2)
        expected = torch.tensor([[1.0, 4.0], [9.0, 16.0]])
        assert torch.allclose(sum_of_squares(r), expected)

    def test_sums_over_all_channels_and_kernels(self) -> None:
        # Two channels, two kernels: should add all four contributions
        r = torch.ones(2, 2, 1, 1)  # all 1s
        assert sum_of_squares(r).item() == pytest.approx(4.0)

    def test_nonnegative(self) -> None:
        assert (sum_of_squares(_response(3, 5, 8, 8)) >= 0).all()

    def test_sign_invariant(self) -> None:
        r = _response(3, 5, 8, 8)
        assert torch.allclose(sum_of_squares(r), sum_of_squares(-r))


class TestSumOfAbs:
    """Tests for sum_of_abs combination function."""

    def test_output_shape(self) -> None:
        r = _response(3, 10, 64, 64)
        assert sum_of_abs(r).shape == torch.Size([64, 64])

    def test_single_element_correctness(self) -> None:
        r = torch.tensor([[[[-3.0]]]])  # (1, 1, 1, 1)
        assert sum_of_abs(r).item() == pytest.approx(3.0)

    def test_multidim_correctness(self) -> None:
        r = torch.tensor([[[[1.0, -2.0], [-3.0, 4.0]]]])  # (1, 1, 2, 2)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.allclose(sum_of_abs(r), expected)

    def test_sums_over_all_channels_and_kernels(self) -> None:
        r = -torch.ones(2, 2, 1, 1)
        assert sum_of_abs(r).item() == pytest.approx(4.0)

    def test_nonnegative(self) -> None:
        assert (sum_of_abs(_response(2, 3, 5, 5)) >= 0).all()

    def test_sign_invariant(self) -> None:
        r = _response(3, 5, 8, 8)
        assert torch.allclose(sum_of_abs(r), sum_of_abs(-r))


class TestMaxAbs:
    """Tests for max_abs combination function."""

    def test_output_shape(self) -> None:
        r = _response(3, 10, 64, 64)
        assert max_abs(r).shape == torch.Size([64, 64])

    def test_single_element_correctness(self) -> None:
        r = torch.tensor([[[[-5.0]]]])  # (1, 1, 1, 1)
        assert max_abs(r).item() == pytest.approx(5.0)

    def test_picks_max_across_channels(self) -> None:
        # Two channels at a single spatial point
        r = torch.tensor([[[[1.0]]], [[[-3.0]]]])  # (2, 1, 1, 1)
        assert max_abs(r).item() == pytest.approx(3.0)

    def test_picks_max_across_kernels(self) -> None:
        # One channel, two kernels at a single spatial point
        r = torch.tensor([[[[2.0]], [[-5.0]]]])  # (1, 2, 1, 1)
        assert max_abs(r).item() == pytest.approx(5.0)

    def test_nonnegative(self) -> None:
        assert (max_abs(_response(2, 3, 5, 5)) >= 0).all()

    def test_max_abs_le_sum_of_abs(self) -> None:
        """Max is always <= sum (for multiple channels/kernels)."""
        r = _response(3, 5, 8, 8)
        assert (max_abs(r) <= sum_of_abs(r) + 1e-6).all()


class TestSumOfPowers:
    """Tests for the sum_of_powers factory function."""

    def test_output_shape(self) -> None:
        r = _response(3, 10, 64, 64)
        assert sum_of_powers(3.0)(r).shape == torch.Size([64, 64])

    def test_power_one_equals_sum_of_abs(self) -> None:
        r = _response(3, 5, 8, 8)
        assert torch.allclose(sum_of_powers(1.0)(r), sum_of_abs(r))

    def test_power_two_equals_sum_of_squares(self) -> None:
        r = _response(3, 5, 8, 8)
        assert torch.allclose(sum_of_powers(2.0)(r), sum_of_squares(r))

    def test_higher_power_sharpens_contrast(self) -> None:
        """Power > 2 amplifies large responses more than small ones."""
        r = torch.tensor([[[[0.1, 0.9]]]])  # (1, 1, 1, 2)
        low_contrast = sum_of_powers(2.0)(r)
        high_contrast = sum_of_powers(4.0)(r)
        # large/small ratio should grow with exponent
        ratio_p2 = low_contrast[0, 1].item() / low_contrast[0, 0].item()
        ratio_p4 = high_contrast[0, 1].item() / high_contrast[0, 0].item()
        assert ratio_p4 > ratio_p2

    def test_returns_callable(self) -> None:
        fn = sum_of_powers(3.0)
        assert callable(fn)

    def test_nonnegative(self) -> None:
        r = _response(2, 3, 5, 5)
        assert (sum_of_powers(3.0)(r) >= 0).all()

    def test_different_powers_give_different_results(self) -> None:
        r = _response(3, 5, 8, 8)
        assert not torch.allclose(sum_of_powers(1.0)(r), sum_of_powers(3.0)(r))

    @pytest.mark.parametrize("power", [0.5, 1.0, 2.0, 3.0, 4.0])
    def test_various_powers_run(self, power: float) -> None:
        r = _response(2, 4, 16, 16)
        out = sum_of_powers(power)(r)
        assert out.shape == torch.Size([16, 16])
        assert (out >= 0).all()
