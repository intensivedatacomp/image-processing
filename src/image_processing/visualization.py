"""Plotting helpers for inspecting convolution kernel banks.

Two entry points are provided:

- :func:`plot_kernel_grid` - every orientation of a kernel bank in one figure.
- :func:`plot_single_kernel` - a single orientation, larger and detailed.

Both accept a built :class:`~image_processing.kernels.BaseKernel` instance
directly, so they work with any kernel family (not just
:class:`~image_processing.kernels.ElongatedMaskKernel`) as long as
``kernel.params`` is a dataclass.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from .kernels import BaseKernel


def _format_params(params: object) -> str:
    """Render a params dataclass as a compact multi-line string for titles.

    Parameters
    ----------
    params : object
        Typically a :class:`~image_processing.params.BaseKernelParams`
        instance. Non-dataclass inputs yield an empty string.

    Returns
    -------
    str
        ``name=value`` pairs joined by commas, wrapped every three fields.
    """
    if not is_dataclass(params):
        return ""
    pairs = [f"{f.name}={getattr(params, f.name)}" for f in fields(params)]
    lines = [", ".join(pairs[i : i + 3]) for i in range(0, len(pairs), 3)]
    return "\n".join(lines)


def plot_kernel_grid(
    kernel: BaseKernel,
    cols: int = 6,
    save_path: str | Path | None = None,
    show: bool = True,
) -> Figure:
    """Plot every orientation of a kernel bank in a single grid figure.

    Parameters
    ----------
    kernel : BaseKernel
        A kernel instance whose :attr:`~image_processing.kernels.BaseKernel.kernels`
        tensor will be built (if not cached yet), moved to CPU, and plotted.
    cols : int
        Number of grid columns. The number of rows is derived from
        ``kernel.params.n_angles``.
    save_path : str or pathlib.Path or None
        If given, the figure is saved to this path (e.g. ``"kernel_grid.png"``).
    show : bool
        Whether to display the figure with ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    k = kernel.kernels.detach().cpu().numpy()
    n = k.shape[0]
    vmax = float(np.abs(k).max())
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(16, 2.8 * rows), constrained_layout=True
    )
    axes = np.atleast_2d(axes).reshape(rows, cols)

    im = None
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            angle_deg = 180.0 * i / n
            im = ax.imshow(k[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"{angle_deg:.0f}\u00b0", fontsize=10)
        ax.axis("off")

    kernel_name = type(kernel).__name__
    title = f"{kernel_name} \u2014 ALL {n} ORIENTATIONS"
    param_str = _format_params(kernel.params)
    if param_str:
        title += f"\n({param_str})"
    fig.suptitle(title, fontsize=12)

    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label="weight")

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()

    return fig


def plot_single_kernel(
    kernel: BaseKernel,
    index: int = 0,
    save_path: str | Path | None = None,
    show: bool = True,
) -> Figure:
    """Plot a single orientation of a kernel bank.

    Parameters
    ----------
    kernel : BaseKernel
        A kernel instance whose :attr:`~image_processing.kernels.BaseKernel.kernels`
        tensor will be built (if not cached yet), moved to CPU, and plotted.
    index : int
        Which orientation (0-indexed) to plot.
    save_path : str or pathlib.Path or None
        If given, the figure is saved to this path (e.g. ``"kernel_single.png"``).
    show : bool
        Whether to display the figure with ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.

    Raises
    ------
    IndexError
        If ``index`` is out of range for the kernel bank.
    """
    k = kernel.kernels.detach().cpu().numpy()
    n = k.shape[0]
    if not 0 <= index < n:
        raise IndexError(f"index {index} out of range for {n} orientations")

    angle_deg = 180.0 * index / n
    vmax = float(np.abs(k[index]).max())
    kernel_name = type(kernel).__name__

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(k[index], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    size = f"{k.shape[1]}\u00d7{k.shape[2]}"
    ax.set_title(f"{kernel_name} ({angle_deg:.0f}\u00b0) \u2014 {size}")
    fig.colorbar(im, ax=ax, label="weight")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()

    return fig
