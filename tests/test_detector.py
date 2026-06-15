"""Unit and integration tests for EdgeDetector."""

import pytest
import torch

from image_processing import (
    EdgeDetector,
    ElongatedMaskKernel,
    ElongatedMaskParams,
    max_abs,
    sum_of_abs,
    sum_of_powers,
    sum_of_squares,
)
from image_processing.combination import CombineFn


class TestEdgeDetectorConstruction:
    """EdgeDetector attribute defaults and custom assignments."""

    def test_default_combine_fn(self, cpu_kernel: ElongatedMaskKernel) -> None:
        det = EdgeDetector(kernel=cpu_kernel)
        assert det.combine_fn is sum_of_squares

    def test_default_normalize_true(self, cpu_kernel: ElongatedMaskKernel) -> None:
        det = EdgeDetector(kernel=cpu_kernel)
        assert det.normalize is True

    def test_custom_combine_fn(self, cpu_kernel: ElongatedMaskKernel) -> None:
        det = EdgeDetector(kernel=cpu_kernel, combine_fn=sum_of_abs)
        assert det.combine_fn is sum_of_abs

    def test_normalize_false(self, cpu_kernel: ElongatedMaskKernel) -> None:
        det = EdgeDetector(kernel=cpu_kernel, normalize=False)
        assert det.normalize is False

    def test_none_kernel_uses_default(self) -> None:
        det = EdgeDetector(kernel=None)
        assert isinstance(det.kernel, ElongatedMaskKernel)


class TestDetectOutputShape:
    """Output spatial dimensions always match the input."""

    def test_rgb_image(
        self,
        default_detector: EdgeDetector,
        rgb_image: torch.Tensor,
    ) -> None:
        assert default_detector.detect(rgb_image).shape == torch.Size([64, 64])

    def test_grayscale_2d(
        self,
        default_detector: EdgeDetector,
        gray_image: torch.Tensor,
    ) -> None:
        assert default_detector.detect(gray_image).shape == torch.Size([64, 64])

    def test_single_channel_3d(self, default_detector: EdgeDetector) -> None:
        img = torch.rand(1, 64, 64)
        assert default_detector.detect(img).shape == torch.Size([64, 64])

    @pytest.mark.parametrize(
        "h, w",
        [(128, 128), (100, 200), (55, 55), (37, 41), (1, 64), (64, 1)],
    )
    def test_various_sizes(
        self,
        default_detector: EdgeDetector,
        h: int,
        w: int,
    ) -> None:
        img = torch.rand(3, h, w)
        assert default_detector.detect(img).shape == torch.Size([h, w])


class TestDetectInputHandling:
    """Input dtype conversion and validation."""

    def test_uint8_cast_to_float(
        self,
        default_detector: EdgeDetector,
    ) -> None:
        img = (torch.rand(3, 32, 32) * 255).byte()
        out = default_detector.detect(img)
        assert out.shape == torch.Size([32, 32])
        assert out.is_floating_point()

    def test_int16_cast_to_float(
        self,
        default_detector: EdgeDetector,
    ) -> None:
        img = (torch.rand(3, 32, 32) * 100).short()
        out = default_detector.detect(img)
        assert out.is_floating_point()

    def test_4d_input_raises(
        self,
        default_detector: EdgeDetector,
    ) -> None:
        with pytest.raises(ValueError, match=r"2-D.*3-D"):
            default_detector.detect(torch.rand(2, 3, 4, 5))

    def test_1d_input_raises(
        self,
        default_detector: EdgeDetector,
    ) -> None:
        with pytest.raises(ValueError):
            default_detector.detect(torch.rand(64))


class TestDetectNormalization:
    """Normalization clamps output to [0, 1] and can be disabled."""

    def test_normalized_range(
        self,
        default_detector: EdgeDetector,
        rgb_image: torch.Tensor,
    ) -> None:
        out = default_detector.detect(rgb_image)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0 + 1e-6

    def test_normalized_max_is_one(
        self,
        default_detector: EdgeDetector,
        rgb_image: torch.Tensor,
    ) -> None:
        out = default_detector.detect(rgb_image)
        assert out.max().item() == pytest.approx(1.0, abs=1e-5)

    def test_no_normalization(
        self,
        cpu_kernel: ElongatedMaskKernel,
        rgb_image: torch.Tensor,
    ) -> None:
        det = EdgeDetector(kernel=cpu_kernel, normalize=False)
        out = det.detect(rgb_image)
        assert out.shape == torch.Size([64, 64])

    def test_normalize_false_differs_from_true(
        self,
        cpu_kernel: ElongatedMaskKernel,
        rgb_image: torch.Tensor,
    ) -> None:
        raw = EdgeDetector(kernel=cpu_kernel, normalize=False).detect(rgb_image)
        normed = EdgeDetector(kernel=cpu_kernel, normalize=True).detect(rgb_image)
        # For typical non-trivial images the raw max is > 1, so they differ
        assert not torch.allclose(raw, normed)


class TestDetectDevice:
    """Output tensor lands on the kernel's device."""

    def test_output_on_cpu(
        self,
        default_detector: EdgeDetector,
        rgb_image: torch.Tensor,
    ) -> None:
        out = default_detector.detect(rgb_image)
        assert out.device.type == "cpu"

    def test_input_moved_to_kernel_device(
        self,
        default_detector: EdgeDetector,
    ) -> None:
        img = torch.rand(3, 32, 32)  # on CPU
        out = default_detector.detect(img)
        assert out.device == default_detector.kernel.kernels.device


class TestDetectBatch:
    """detect_batch processes lists of images of varying sizes."""

    def test_basic_batch(self, default_detector: EdgeDetector) -> None:
        imgs = [torch.rand(3, 32, 32), torch.rand(1, 48, 48)]
        results = default_detector.detect_batch(imgs)
        assert [r.shape for r in results] == [
            torch.Size([32, 32]),
            torch.Size([48, 48]),
        ]

    def test_empty_batch(self, default_detector: EdgeDetector) -> None:
        assert default_detector.detect_batch([]) == []

    def test_single_image_batch(
        self,
        default_detector: EdgeDetector,
        rgb_image: torch.Tensor,
    ) -> None:
        results = default_detector.detect_batch([rgb_image])
        assert len(results) == 1
        assert results[0].shape == torch.Size([64, 64])

    def test_batch_matches_individual(
        self,
        default_detector: EdgeDetector,
    ) -> None:
        """Batch and sequential calls must give identical results."""
        imgs = [torch.rand(3, 32, 32) for _ in range(3)]
        batch_results = default_detector.detect_batch(imgs)
        individual_results = [default_detector.detect(img) for img in imgs]
        for b, s in zip(batch_results, individual_results, strict=True):
            assert torch.allclose(b, s)


@pytest.mark.parametrize(
    "combine_fn, fn_id",
    [
        (sum_of_squares, "sum_of_squares"),
        (sum_of_abs, "sum_of_abs"),
        (max_abs, "max_abs"),
        (sum_of_powers(3.0), "sum_of_powers_3"),
        (sum_of_powers(0.5), "sum_of_powers_0.5"),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_full_pipeline_all_combination_fns(
    combine_fn: CombineFn,
    fn_id: str,
    cpu_kernel: ElongatedMaskKernel,
) -> None:
    """Integration: all built-in combination functions complete without error."""
    det = EdgeDetector(kernel=cpu_kernel, combine_fn=combine_fn)
    torch.manual_seed(42)
    img = torch.rand(3, 64, 64)
    out = det.detect(img)
    assert out.shape == torch.Size([64, 64])
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0 + 1e-6


def test_different_combination_fns_give_different_results(
    cpu_kernel: ElongatedMaskKernel,
) -> None:
    """sum_of_squares and sum_of_abs produce distinct edge maps."""
    torch.manual_seed(0)
    img = torch.rand(3, 64, 64)
    det_sq = EdgeDetector(kernel=cpu_kernel, combine_fn=sum_of_squares)
    det_ab = EdgeDetector(kernel=cpu_kernel, combine_fn=sum_of_abs)
    assert not torch.allclose(det_sq.detect(img), det_ab.detect(img))


def test_fine_kernel_preserves_shape(fine_kernel: ElongatedMaskKernel) -> None:
    """Integration with a high-resolution kernel (18 angles, 61x61)."""
    det = EdgeDetector(kernel=fine_kernel)
    for h, w in [(128, 128), (100, 200), (37, 41)]:
        img = torch.rand(3, h, w)
        assert det.detect(img).shape == torch.Size([h, w])


def test_custom_kernel_subclass_integration() -> None:
    """A hand-crafted BaseKernel subclass can be plugged into EdgeDetector."""

    class SobelKernel(ElongatedMaskKernel):
        """Thin wrapper that uses very few angles for speed."""

        def __init__(self) -> None:
            super().__init__(
                ElongatedMaskParams(
                    n_angles=4, kernel_half_size=10, stripe_half_length=8
                ),
                device="cpu",
            )

    det = EdgeDetector(kernel=SobelKernel())
    img = torch.rand(3, 64, 64)
    out = det.detect(img)
    assert out.shape == torch.Size([64, 64])
