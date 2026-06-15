"""Combination functions for aggregating per-orientation convolution responses.

A combination function receives the full ``(C, N, H, W)`` response tensor
produced by applying ``N`` oriented kernels to a ``C``-channel image and
reduces it to a ``(H, W)`` edge-strength map.

All built-in functions expect the response to have at least two dimensions
``(C, N, ...)``; additional spatial dimensions are preserved.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

# Callable (C, N, H, W) -> (H, W): aggregates multi-channel, multi-orientation
# response tensor into a single-channel edge map.
type CombineFn = Callable[[torch.Tensor], torch.Tensor]


def sum_of_squares(response: torch.Tensor) -> torch.Tensor:
    """Sum squared responses over all colour channels and kernel orientations.

    Parameters
    ----------
    response : torch.Tensor
        Shape ``(C, N, H, W)``.

    Returns
    -------
    torch.Tensor
        Shape ``(H, W)``.

    Examples
    --------
    >>> import torch
    >>> from image_processing.combination import sum_of_squares
    >>> r = torch.randn(3, 10, 64, 64)
    >>> sum_of_squares(r).shape
    torch.Size([64, 64])
    """
    return torch.sum(response**2, dim=(0, 1))


def sum_of_abs(response: torch.Tensor) -> torch.Tensor:
    """Sum absolute responses over all colour channels and kernel orientations.

    Parameters
    ----------
    response : torch.Tensor
        Shape ``(C, N, H, W)``.

    Returns
    -------
    torch.Tensor
        Shape ``(H, W)``.

    Examples
    --------
    >>> import torch
    >>> from image_processing.combination import sum_of_abs
    >>> r = torch.randn(3, 10, 64, 64)
    >>> sum_of_abs(r).shape
    torch.Size([64, 64])
    """
    return torch.sum(torch.abs(response), dim=(0, 1))


def max_abs(response: torch.Tensor) -> torch.Tensor:
    """Return the pixel-wise maximum absolute response over channels and orientations.

    Parameters
    ----------
    response : torch.Tensor
        Shape ``(C, N, H, W)``.

    Returns
    -------
    torch.Tensor
        Shape ``(H, W)``.

    Examples
    --------
    >>> import torch
    >>> from image_processing.combination import max_abs
    >>> r = torch.randn(3, 10, 64, 64)
    >>> max_abs(r).shape
    torch.Size([64, 64])
    """
    return torch.abs(response).amax(dim=(0, 1))


def sum_of_powers(power: float) -> CombineFn:
    """Return a combination function that sums ``|response|^power``.

    Higher powers emphasise strong responses and suppress weak ones, which
    sharpens detected edges at the cost of sensitivity.  ``power=1`` is
    equivalent to :func:`sum_of_abs`; ``power=2`` is equivalent to
    :func:`sum_of_squares`.

    Parameters
    ----------
    power : float
        Exponent applied to the absolute response before summation.

    Returns
    -------
    CombineFn
        A callable with signature ``(C, N, H, W) → (H, W)``.

    Examples
    --------
    >>> import torch
    >>> from image_processing.combination import sum_of_powers
    >>> fn = sum_of_powers(3.0)
    >>> fn(torch.randn(3, 10, 64, 64)).shape
    torch.Size([64, 64])
    """

    def _combine(response: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.abs(response) ** power, dim=(0, 1))

    _combine.__name__ = f"sum_of_powers(power={power})"
    return _combine
