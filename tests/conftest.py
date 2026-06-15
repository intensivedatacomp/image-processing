"""Shared pytest fixtures for the image_processing test suite."""

import pytest
import torch

from image_processing import EdgeDetector, ElongatedMaskKernel, ElongatedMaskParams


@pytest.fixture(scope="session")
def cpu_kernel() -> ElongatedMaskKernel:
    """Return the default ElongatedMaskKernel on CPU, built once per test session."""
    return ElongatedMaskKernel(device="cpu")


@pytest.fixture(scope="session")
def fine_kernel() -> ElongatedMaskKernel:
    """Higher-resolution kernel (18 angles, 61x61) for integration tests."""
    params = ElongatedMaskParams(
        n_angles=18,
        kernel_half_size=30,
        stripe_half_width=5,
        stripe_half_length=30,
        length_falloff=0.1,
        width_falloff=1.0,
    )
    return ElongatedMaskKernel(params, device="cpu")


@pytest.fixture(scope="session")
def default_detector(cpu_kernel: ElongatedMaskKernel) -> EdgeDetector:
    """Return the EdgeDetector with default settings on CPU."""
    return EdgeDetector(kernel=cpu_kernel)


@pytest.fixture
def rgb_image() -> torch.Tensor:
    """Random (3, 64, 64) float image in [0, 1]."""
    return torch.rand(3, 64, 64)


@pytest.fixture
def gray_image() -> torch.Tensor:
    """Random (64, 64) float image in [0, 1]."""
    return torch.rand(64, 64)
