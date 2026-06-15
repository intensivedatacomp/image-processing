"""Unit tests for BaseKernel and ElongatedMaskKernel."""

import pytest
import torch

from image_processing import (
    BaseKernel,
    BaseKernelParams,
    ElongatedMaskKernel,
    ElongatedMaskParams,
)


class TestBaseKernelAbstract:
    """BaseKernel is abstract and cannot be instantiated directly."""

    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            BaseKernel(BaseKernelParams())  # type: ignore[abstract]

    def test_subclass_must_implement_build(self) -> None:
        class Incomplete(BaseKernel):
            pass  # no build() method

        with pytest.raises(TypeError):
            Incomplete(BaseKernelParams())  # type: ignore[abstract]

    def test_valid_subclass(self) -> None:
        class ConstantKernel(BaseKernel):
            def build(self) -> torch.Tensor:
                n = self.params.n_angles
                return torch.ones(n, 3, 3, device=self.device)

        k = ConstantKernel(BaseKernelParams(n_angles=5), device="cpu")
        assert k.kernels.shape == torch.Size([5, 3, 3])
        assert k.kernels.device.type == "cpu"


class TestElongatedMaskKernelShape:
    """Kernel output shapes for various parameter combinations."""

    def test_default_shape(self, cpu_kernel: ElongatedMaskKernel) -> None:
        # 10 angles, size = 2 * 20 + 1 = 41
        assert cpu_kernel.kernels.shape == torch.Size([10, 41, 41])

    def test_none_params_uses_defaults(self) -> None:
        k = ElongatedMaskKernel(None, device="cpu")
        assert k.kernels.shape == torch.Size([10, 41, 41])

    def test_custom_n_angles(self) -> None:
        params = ElongatedMaskParams(n_angles=6, kernel_half_size=20)
        k = ElongatedMaskKernel(params, device="cpu")
        assert k.kernels.shape[0] == 6

    def test_custom_half_size(self) -> None:
        # stripe_half_length must be <= kernel_half_size
        params = ElongatedMaskParams(kernel_half_size=15, stripe_half_length=15)
        k = ElongatedMaskKernel(params, device="cpu")
        assert k.kernels.shape == torch.Size([10, 31, 31])  # 2*15+1

    def test_fine_kernel_shape(self, fine_kernel: ElongatedMaskKernel) -> None:
        # 18 angles, size = 2 * 30 + 1 = 61
        assert fine_kernel.kernels.shape == torch.Size([18, 61, 61])


class TestElongatedMaskKernelDevice:
    """Kernel tensors land on the specified device."""

    def test_cpu_device(self, cpu_kernel: ElongatedMaskKernel) -> None:
        assert cpu_kernel.kernels.device.type == "cpu"

    def test_string_device_cpu(self) -> None:
        k = ElongatedMaskKernel(device="cpu")
        assert k.kernels.device.type == "cpu"

    def test_none_device_defaults_to_available(self) -> None:
        k = ElongatedMaskKernel(device=None)
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert k.kernels.device.type == expected


class TestElongatedMaskKernelValues:
    """Kernel value properties derived from the anti-symmetric construction."""

    def test_each_kernel_sums_to_zero(self, cpu_kernel: ElongatedMaskKernel) -> None:
        """Anti-symmetrised kernels (mask - rotate_180(mask)) sum to zero."""
        for mask in cpu_kernel.kernels:
            assert abs(mask.sum().item()) < 1e-4

    def test_kernels_not_all_zero(self, cpu_kernel: ElongatedMaskKernel) -> None:
        assert cpu_kernel.kernels.abs().max().item() > 0.0

    def test_kernels_are_float(self, cpu_kernel: ElongatedMaskKernel) -> None:
        assert cpu_kernel.kernels.is_floating_point()

    def test_width_falloff_changes_kernels(self) -> None:
        """Kernels differ when width falloff is enabled vs. disabled."""
        k_flat = ElongatedMaskKernel(
            ElongatedMaskParams(width_falloff=0.0), device="cpu"
        )
        k_decayed = ElongatedMaskKernel(
            ElongatedMaskParams(width_falloff=1.0), device="cpu"
        )
        assert not torch.allclose(k_flat.kernels, k_decayed.kernels)

    def test_length_falloff_changes_kernels(self) -> None:
        k_slow = ElongatedMaskKernel(
            ElongatedMaskParams(length_falloff=0.01), device="cpu"
        )
        k_fast = ElongatedMaskKernel(
            ElongatedMaskParams(length_falloff=0.5), device="cpu"
        )
        assert not torch.allclose(k_slow.kernels, k_fast.kernels)


class TestElongatedMaskKernelCache:
    """Lazy caching and cache invalidation behaviour."""

    def test_kernels_cached_on_first_access(self) -> None:
        k = ElongatedMaskKernel(device="cpu")
        first = k.kernels
        second = k.kernels
        assert first is second  # exact same object

    def test_reset_clears_cache(self) -> None:
        k = ElongatedMaskKernel(device="cpu")
        _ = k.kernels
        assert k._cache is not None
        k.reset()
        assert k._cache is None

    def test_rebuild_after_reset_matches(self) -> None:
        k = ElongatedMaskKernel(device="cpu")
        before = k.kernels.clone()
        k.reset()
        after = k.kernels
        assert torch.allclose(before, after)


class TestElongatedMaskKernelValidation:
    """ValueError is raised when stripe dimensions exceed the canvas."""

    def test_stripe_length_exceeds_half_size(self) -> None:
        params = ElongatedMaskParams(kernel_half_size=10, stripe_half_length=15)
        with pytest.raises(ValueError, match="stripe_half_length"):
            _ = ElongatedMaskKernel(params, device="cpu").kernels

    def test_stripe_width_exceeds_half_size(self) -> None:
        # stripe_half_length kept within bounds; stripe_half_width is the violator
        params = ElongatedMaskParams(
            kernel_half_size=10, stripe_half_length=8, stripe_half_width=15
        )
        with pytest.raises(ValueError, match="stripe_half_width"):
            _ = ElongatedMaskKernel(params, device="cpu").kernels

    def test_stripe_fits_exactly(self) -> None:
        """Stripe that exactly fits the canvas (length == half_size) is allowed."""
        params = ElongatedMaskParams(kernel_half_size=20, stripe_half_length=20)
        k = ElongatedMaskKernel(params, device="cpu")
        assert k.kernels.shape[0] == params.n_angles
