"""Extending the package with a custom kernel family.

Demonstrates how to subclass BaseKernel and BaseKernelParams to implement
a new kernel type — here, Laplacian-of-Gaussian (LoG) kernels — that slots
directly into EdgeDetector without any changes to the core package.

Run with:
    python examples/custom_kernel.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from image_processing import BaseKernel, BaseKernelParams, EdgeDetector
from image_processing.combination import max_abs

# ---------------------------------------------------------------------------
# 1. Custom parameter dataclass
# ---------------------------------------------------------------------------


@dataclass
class LogKernelParams(BaseKernelParams):
    """Parameters for multi-scale Laplacian-of-Gaussian kernels.

    Parameters
    ----------
    sigmas : tuple of float
        Standard deviations for the Gaussian blur before the Laplacian.
        Each sigma produces one kernel, so ``n_angles`` is unused here.
    kernel_half_size : int
        Half-size of the square kernel canvas.
    """

    sigmas: tuple[float, ...] = (1.0, 2.0, 4.0)


# ---------------------------------------------------------------------------
# 2. Custom kernel class
# ---------------------------------------------------------------------------


class LogKernel(BaseKernel):
    """Multi-scale Laplacian-of-Gaussian (LoG) edge kernels.

    Generates one LoG kernel per sigma.  The LoG is approximated analytically:

        LoG(x, y) = -1/(pi * sigma^4) * (1 - (x^2+y^2)/(2*sigma^2))
                    * exp(-(x^2+y^2) / (2*sigma^2))

    Parameters
    ----------
    params : LogKernelParams or None
        Kernel configuration.  Defaults to :class:`LogKernelParams`.
    device : torch.device or str or None
        Computation device.
    """

    def __init__(
        self,
        params: LogKernelParams | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(params if params is not None else LogKernelParams(), device)

    def build(self) -> torch.Tensor:
        """Build one LoG kernel per sigma.

        Returns
        -------
        torch.Tensor
            Shape ``(len(sigmas), 2 * kernel_half_size + 1, 2 * kernel_half_size + 1)``.
        """
        from typing import cast

        p = cast(LogKernelParams, self.params)
        half = p.kernel_half_size
        coords = np.arange(-half, half + 1, dtype=np.float64)
        yy, xx = np.meshgrid(coords, coords, indexing="ij")
        r2 = xx**2 + yy**2

        kernels: list[torch.Tensor] = []
        for sigma in p.sigmas:
            s2 = sigma**2
            log = -1.0 / (np.pi * s2**2) * (1 - r2 / (2 * s2)) * np.exp(-r2 / (2 * s2))
            log -= log.mean()  # zero-sum kernel
            k = torch.from_numpy(log).float()
            kernels.append(k)

        stacked = torch.stack(kernels)  # (n_sigmas, size, size)
        return stacked.to(self.device)


# ---------------------------------------------------------------------------
# 3. Plug into EdgeDetector without changing any package code
# ---------------------------------------------------------------------------

params = LogKernelParams(
    sigmas=(0.5, 1.0, 2.0, 4.0),
    kernel_half_size=20,
)
log_kernel = LogKernel(params, device="cpu")

print(f"LoG kernel stack shape : {log_kernel.kernels.shape}")
print(
    f"Each kernel sums to ~0 : {[f'{k.sum().item():.2e}' for k in log_kernel.kernels]}"
)

# Use max_abs so each pixel reports the strongest response across all scales
detector = EdgeDetector(kernel=log_kernel, combine_fn=max_abs)

torch.manual_seed(7)
image = torch.rand(3, 64, 64)
edges = detector.detect(image)

print(f"\nInput  : {image.shape}")
print(f"Output : {edges.shape}, range [{edges.min():.4f}, {edges.max():.4f}]")

# ---------------------------------------------------------------------------
# 4. Compare LoG with the default elongated-mask kernel
# ---------------------------------------------------------------------------

from image_processing import ElongatedMaskKernel  # noqa: E402

elongated_detector = EdgeDetector(kernel=ElongatedMaskKernel(device="cpu"))
edges_elongated = elongated_detector.detect(image)

print("\nPer-pixel correlation between LoG and elongated-mask edge maps:")
flat_log = edges.flatten()
flat_elo = edges_elongated.flatten()
corr = torch.corrcoef(torch.stack([flat_log, flat_elo]))[0, 1]
print(f"  Pearson r = {corr.item():.4f}")
