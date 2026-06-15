"""Convolution kernel classes for edge detection."""

from __future__ import annotations

import abc
from typing import cast

import numpy as np
import PIL.Image
import PIL.ImageOps
import torch
import torchvision.transforms

from .params import BaseKernelParams, ElongatedMaskParams


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class BaseKernel(abc.ABC):
    """Abstract base class for edge-detection convolution kernels.

    Concrete subclasses implement :meth:`build` to construct a ``(N, kH, kW)``
    kernel stack on ``self.device``.  The result is cached after the first
    access and can be discarded with :meth:`reset`.

    Parameters
    ----------
    params : BaseKernelParams
        Kernel configuration.
    device : torch.device or str or None
        Computation device.  Defaults to CUDA when available, otherwise CPU.

    Examples
    --------
    Minimal subclass skeleton::

        class MyKernel(BaseKernel):
            def build(self) -> torch.Tensor:
                # return tensor of shape (n_kernels, kH, kW) on self.device
                ...
    """

    def __init__(
        self,
        params: BaseKernelParams,
        device: torch.device | str | None = None,
    ) -> None:
        self.params = params
        self.device = _resolve_device(device)
        self._cache: torch.Tensor | None = None

    @abc.abstractmethod
    def build(self) -> torch.Tensor:
        """Build and return the kernel stack.

        Returns
        -------
        torch.Tensor
            Shape ``(n_kernels, kH, kW)`` on ``self.device``.
        """

    @property
    def kernels(self) -> torch.Tensor:
        """Return the lazily built and cached kernel tensor.

        Returns
        -------
        torch.Tensor
            Shape ``(n_kernels, kH, kW)``.
        """
        if self._cache is None:
            self._cache = self.build()
        return self._cache

    def reset(self) -> None:
        """Invalidate the cached kernels.

        The next access to :attr:`kernels` triggers a fresh call to
        :meth:`build`.
        """
        self._cache = None


class ElongatedMaskKernel(BaseKernel):
    """Rotated elongated-stripe convolution kernels for edge detection.

    Implements the method from *Contour-texture separation: part 2*
    (Antal, 2024).  The construction proceeds as follows:

    1. A horizontal stripe of decaying values is placed on a blank square
       canvas.  Weights fall off along the stripe length as
       ``1 / (length_falloff * |j| + 1)`` and optionally across the width as
       ``1 / (width_falloff * i + 1)``.
    2. The canvas is converted to a PIL image and rotated to ``n_angles``
       orientations evenly distributed in ``[0°, 180°)``.
    3. Each rotated mask is anti-symmetrised by subtracting its 180° rotation
       (mirror flip).  The resulting kernels detect signed intensity gradients
       perpendicular to the stripe.

    With ``padding='same'`` the convolution output preserves the spatial size
    of the input image.

    Parameters
    ----------
    params : ElongatedMaskParams or None
        Kernel configuration.  Uses default
        :class:`~image_processing.ElongatedMaskParams` when ``None``.
    device : torch.device or str or None
        Computation device.  Defaults to CUDA when available, otherwise CPU.

    Examples
    --------
    Default configuration (10 angles, 41 x 41 kernel):

    >>> from image_processing import ElongatedMaskKernel
    >>> kernel = ElongatedMaskKernel()
    >>> kernel.kernels.shape
    torch.Size([10, 41, 41])

    GPU-tuned configuration matching the notebook's GPU example:

    >>> from image_processing import ElongatedMaskKernel, ElongatedMaskParams
    >>> params = ElongatedMaskParams(
    ...     n_angles=18,
    ...     kernel_half_size=30,
    ...     stripe_half_width=5,
    ...     stripe_half_length=30,
    ...     length_falloff=0.1,
    ...     width_falloff=1.0,
    ... )
    >>> kernel = ElongatedMaskKernel(params, device="cpu")
    >>> kernel.kernels.shape
    torch.Size([18, 61, 61])
    """

    def __init__(
        self,
        params: ElongatedMaskParams | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(
            params if params is not None else ElongatedMaskParams(),
            device,
        )

    def build(self) -> torch.Tensor:
        """Build the anti-symmetrised rotated stripe kernel stack.

        Returns
        -------
        torch.Tensor
            Shape ``(n_angles, 2 * kernel_half_size + 1, 2 * kernel_half_size + 1)``
            on ``self.device``.

        Raises
        ------
        ValueError
            If the stripe dimensions exceed ``kernel_half_size``.
        """
        p = cast(ElongatedMaskParams, self.params)
        half = p.kernel_half_size
        size = 2 * half + 1  # odd size → clear centre pixel, no asymmetric-pad warning

        errors: list[str] = []
        if p.stripe_half_length > half:
            errors.append(
                f"stripe_half_length ({p.stripe_half_length}) must be "
                f"<= kernel_half_size ({half})"
            )
        if p.stripe_half_width > half:
            errors.append(
                f"stripe_half_width ({p.stripe_half_width}) must be "
                f"<= kernel_half_size ({half})"
            )
        if errors:
            raise ValueError(". ".join(errors) + ".")

        canvas = self._make_canvas(p, size, half)
        canvas_pil = torchvision.transforms.ToPILImage()(canvas)

        angles = np.arange(0.0, 180.0, 180.0 / p.n_angles)
        masks: list[torch.Tensor] = []
        for phi in angles:
            rotated = canvas_pil.rotate(
                float(phi), PIL.Image.Resampling.NEAREST, expand=False
            )
            fwd = torchvision.transforms.ToTensor()(rotated)
            rev = torchvision.transforms.ToTensor()(
                PIL.ImageOps.mirror(PIL.ImageOps.flip(rotated))
            )
            masks.append(fwd - rev)

        stacked = torch.cat(masks).view(p.n_angles, size, size)
        return stacked.to(self.device)

    @staticmethod
    def _make_canvas(p: ElongatedMaskParams, size: int, half: int) -> torch.Tensor:
        """Create the base unrotated stripe canvas as a ``(size, size)`` tensor."""
        canvas = torch.zeros((size, size))
        j_offsets = torch.arange(-p.stripe_half_length, p.stripe_half_length)
        length_weights = 1.0 / (p.length_falloff * j_offsets.abs().float() + 1.0)
        col_start = half - p.stripe_half_length
        col_end = half + p.stripe_half_length
        for i in range(p.stripe_half_width):
            width_weight = (
                1.0 / (p.width_falloff * float(i) + 1.0)
                if p.width_falloff > 0.0
                else 1.0
            )
            canvas[half + i, col_start:col_end] = length_weights * width_weight
        return canvas
