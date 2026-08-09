"""Unit tests for the visualization module."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for CI / test runs

import matplotlib.pyplot as plt
import pytest

from image_processing import ElongatedMaskKernel, plot_kernel_grid, plot_single_kernel


class TestPlotKernelGrid:
    """plot_kernel_grid renders every orientation of a kernel bank."""

    def test_returns_figure(self, cpu_kernel: ElongatedMaskKernel) -> None:
        fig = plot_kernel_grid(cpu_kernel, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_axes_count_covers_all_orientations(
        self, cpu_kernel: ElongatedMaskKernel
    ) -> None:
        cols = 6
        fig = plot_kernel_grid(cpu_kernel, cols=cols, show=False)
        n = cpu_kernel.params.n_angles
        expected_rows = (n + cols - 1) // cols
        assert len(fig.axes) >= expected_rows * cols
        plt.close(fig)

    def test_saves_to_disk(
        self, cpu_kernel: ElongatedMaskKernel, tmp_path: Path
    ) -> None:
        out = tmp_path / "grid.png"
        fig = plot_kernel_grid(cpu_kernel, save_path=out, show=False)
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close(fig)

    def test_works_with_custom_params(
        self, fine_kernel: ElongatedMaskKernel
    ) -> None:
        fig = plot_kernel_grid(fine_kernel, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_via_kernel_method(self, cpu_kernel: ElongatedMaskKernel) -> None:
        """BaseKernel.plot_all() is a thin wrapper around plot_kernel_grid."""
        fig = cpu_kernel.plot_all(show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotSingleKernel:
    """plot_single_kernel renders one orientation of a kernel bank."""

    def test_returns_figure(self, cpu_kernel: ElongatedMaskKernel) -> None:
        fig = plot_single_kernel(cpu_kernel, index=0, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_disk(
        self, cpu_kernel: ElongatedMaskKernel, tmp_path: Path
    ) -> None:
        out = tmp_path / "single.png"
        fig = plot_single_kernel(cpu_kernel, save_path=out, show=False)
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close(fig)

    def test_out_of_range_index_raises(
        self, cpu_kernel: ElongatedMaskKernel
    ) -> None:
        n = cpu_kernel.params.n_angles
        with pytest.raises(IndexError):
            plot_single_kernel(cpu_kernel, index=n, show=False)

    def test_negative_index_raises(self, cpu_kernel: ElongatedMaskKernel) -> None:
        with pytest.raises(IndexError):
            plot_single_kernel(cpu_kernel, index=-1, show=False)

    def test_via_kernel_method(self, cpu_kernel: ElongatedMaskKernel) -> None:
        """BaseKernel.plot() is a thin wrapper around plot_single_kernel."""
        fig = cpu_kernel.plot(index=0, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
