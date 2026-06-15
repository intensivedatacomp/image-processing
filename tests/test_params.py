"""Unit tests for kernel parameter dataclasses."""

import pytest

from image_processing import BaseKernelParams, ElongatedMaskParams


class TestBaseKernelParams:
    """Tests for BaseKernelParams defaults, construction, and equality."""

    def test_defaults(self) -> None:
        p = BaseKernelParams()
        assert p.n_angles == 10
        assert p.kernel_half_size == 20

    def test_custom_values(self) -> None:
        p = BaseKernelParams(n_angles=18, kernel_half_size=30)
        assert p.n_angles == 18
        assert p.kernel_half_size == 30

    def test_equality(self) -> None:
        assert BaseKernelParams() == BaseKernelParams()
        assert BaseKernelParams(n_angles=5) != BaseKernelParams(n_angles=10)

    def test_is_dataclass(self) -> None:
        import dataclasses

        assert dataclasses.is_dataclass(BaseKernelParams)


class TestElongatedMaskParams:
    """Tests for ElongatedMaskParams defaults, construction, and inheritance."""

    def test_defaults(self) -> None:
        p = ElongatedMaskParams()
        assert p.n_angles == 10
        assert p.kernel_half_size == 20
        assert p.stripe_half_width == 5
        assert p.stripe_half_length == 20
        assert p.length_falloff == pytest.approx(0.05)
        assert p.width_falloff == pytest.approx(0.0)

    def test_custom_values(self) -> None:
        p = ElongatedMaskParams(
            n_angles=18,
            kernel_half_size=30,
            stripe_half_width=3,
            stripe_half_length=25,
            length_falloff=0.1,
            width_falloff=1.0,
        )
        assert p.n_angles == 18
        assert p.kernel_half_size == 30
        assert p.stripe_half_width == 3
        assert p.stripe_half_length == 25
        assert p.length_falloff == pytest.approx(0.1)
        assert p.width_falloff == pytest.approx(1.0)

    def test_inherits_base_params(self) -> None:
        assert issubclass(ElongatedMaskParams, BaseKernelParams)

    def test_base_fields_accessible(self) -> None:
        p = ElongatedMaskParams(n_angles=6, kernel_half_size=15)
        assert p.n_angles == 6
        assert p.kernel_half_size == 15

    def test_equality(self) -> None:
        assert ElongatedMaskParams() == ElongatedMaskParams()
        assert ElongatedMaskParams(n_angles=5) != ElongatedMaskParams(n_angles=10)

    def test_is_dataclass(self) -> None:
        import dataclasses

        assert dataclasses.is_dataclass(ElongatedMaskParams)
