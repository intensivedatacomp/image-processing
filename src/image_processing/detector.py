"""Edge detector that applies a kernel stack to an image and combines responses."""

from __future__ import annotations

import torch
import torch.nn.functional as torch_f

from .combination import CombineFn, sum_of_squares
from .kernels import BaseKernel, ElongatedMaskKernel


class EdgeDetector:
    """Apply a stack of convolution kernels to detect edges in an image.

    The detector runs the following steps:

    1. Moves the input image to the kernel's device and casts it to
       ``float32`` if needed.
    2. Applies all ``N`` kernel orientations to each of the ``C`` colour
       channels via :func:`torch.nn.functional.conv2d` with
       ``padding='same'``, producing a ``(C, N, H, W)`` response tensor.
    3. Passes the response tensor to ``combine_fn`` to obtain a ``(H, W)``
       edge-strength map.
    4. Optionally normalises the map to ``[0, 1]``.

    The output spatial size always matches the input spatial size.

    Parameters
    ----------
    kernel : BaseKernel or None
        Provider of the oriented kernel stack.  Defaults to
        :class:`~image_processing.ElongatedMaskKernel` with default parameters.
    combine_fn : CombineFn or None
        Callable ``(C, N, H, W) → (H, W)`` that aggregates the response
        tensor.  Defaults to :func:`~image_processing.combination.sum_of_squares`.
    normalize : bool
        When ``True`` (default), divides the edge map by its maximum so the
        output lies in ``[0, 1]``.  Has no effect when the map is all zeros.

    Examples
    --------
    Minimal usage with default parameters:

    >>> import torch
    >>> from image_processing import EdgeDetector
    >>> detector = EdgeDetector()
    >>> image = torch.rand(3, 256, 256)        # (C, H, W) float image
    >>> edges = detector.detect(image)
    >>> edges.shape
    torch.Size([256, 256])

    Custom kernel and combination function:

    >>> from image_processing import (
    ...     EdgeDetector,
    ...     ElongatedMaskKernel,
    ...     ElongatedMaskParams,
    ... )
    >>> from image_processing.combination import sum_of_powers
    >>> params = ElongatedMaskParams(
    ...     n_angles=18,
    ...     kernel_half_size=30,
    ...     length_falloff=0.1,
    ...     width_falloff=1.0,
    ... )
    >>> detector = EdgeDetector(
    ...     kernel=ElongatedMaskKernel(params, device="cpu"),
    ...     combine_fn=sum_of_powers(3.0),
    ... )
    >>> detector.detect(torch.rand(3, 128, 128)).shape
    torch.Size([128, 128])
    """

    def __init__(
        self,
        kernel: BaseKernel | None = None,
        combine_fn: CombineFn | None = None,
        normalize: bool = True,
    ) -> None:
        self.kernel: BaseKernel = (
            kernel if kernel is not None else ElongatedMaskKernel()
        )
        self.combine_fn: CombineFn = (
            combine_fn if combine_fn is not None else sum_of_squares
        )
        self.normalize = normalize

    def detect(self, image: torch.Tensor) -> torch.Tensor:
        """Detect edges in a single image.

        Parameters
        ----------
        image : torch.Tensor
            Input image of shape ``(C, H, W)`` or ``(H, W)``.  Grayscale
            inputs ``(H, W)`` are expanded to ``(1, H, W)`` automatically.
            Non-float tensors are cast to ``float32``.

        Returns
        -------
        torch.Tensor
            Edge map of shape ``(H, W)`` on the kernel's device.

        Raises
        ------
        ValueError
            If ``image`` is not 2-D or 3-D.

        Examples
        --------
        >>> import torch
        >>> from image_processing import EdgeDetector
        >>> EdgeDetector().detect(torch.rand(3, 64, 64)).shape
        torch.Size([64, 64])
        """
        if image.dim() == 2:
            image = image.unsqueeze(0)
        if image.dim() != 3:
            raise ValueError(
                f"Expected a 2-D (H, W) or 3-D (C, H, W) tensor, "
                f"got shape {tuple(image.shape)}."
            )

        k = self.kernel.kernels
        image = image.to(k.device)
        if not image.is_floating_point():
            image = image.float()

        n_channels, height, width = image.shape
        n_kernels, k_height, k_width = k.shape

        # Treat colour channels as independent batch items so that a single
        # conv2d call applies all N kernels to each channel simultaneously.
        x = image.view(n_channels, 1, height, width)
        w = k.view(n_kernels, 1, k_height, k_width)

        response = torch_f.conv2d(x, w, padding="same")  # (C, N, H, W)
        edge_map = self.combine_fn(response)  # (H, W)

        if self.normalize and edge_map.max() > 0:
            edge_map = edge_map / edge_map.max()

        return edge_map

    def detect_batch(self, images: list[torch.Tensor]) -> list[torch.Tensor]:
        """Detect edges in a list of images.

        Images may have different spatial sizes; each is processed
        independently via :meth:`detect`.

        Parameters
        ----------
        images : list of torch.Tensor
            Each element is a ``(C, H, W)`` or ``(H, W)`` tensor.

        Returns
        -------
        list of torch.Tensor
            One ``(H, W)`` edge map per input image.

        Examples
        --------
        >>> import torch
        >>> from image_processing import EdgeDetector
        >>> imgs = [torch.rand(3, 64, 64), torch.rand(3, 128, 96)]
        >>> [m.shape for m in EdgeDetector().detect_batch(imgs)]
        [torch.Size([64, 64]), torch.Size([128, 96])]
        """
        return [self.detect(img) for img in images]
