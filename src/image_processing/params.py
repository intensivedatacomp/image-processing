"""Parameter dataclasses for edge detection kernels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaseKernelParams:
    """Base parameters shared by all convolution kernel families.

    Parameters
    ----------
    n_angles : int
        Number of kernel orientations, evenly spaced in ``[0°, 180°)``.
    kernel_half_size : int
        Half-size of the square kernel canvas.  The full kernel will be
        ``(2 * kernel_half_size) x (2 * kernel_half_size)`` pixels.
    """

    n_angles: int = 10
    kernel_half_size: int = 20


@dataclass
class ElongatedMaskParams(BaseKernelParams):
    """Parameters for the elongated directional stripe kernel.

    The base mask is a horizontal stripe whose pixel values decay along its
    length and optionally across its width.  The stripe is rotated to
    ``n_angles`` orientations and anti-symmetrised by subtracting its own
    180° rotation, making each oriented kernel respond to signed intensity
    gradients perpendicular to the stripe direction.

    Parameters
    ----------
    stripe_half_width : int
        Number of pixel rows occupied by the stripe measured from the centre
        row (total stripe width = ``stripe_half_width`` rows).
    stripe_half_length : int
        Half-length of the stripe along its axis.  Column offsets span
        ``-stripe_half_length`` to ``stripe_half_length - 1``.
    length_falloff : float
        Decay coefficient along the stripe axis.  The weight at column offset
        ``j`` is ``1 / (length_falloff * |j| + 1)``.  Set to ``0.0`` for a
        uniform-weight stripe.
    width_falloff : float
        Decay coefficient across the stripe.  The weight at row offset ``i``
        is multiplied by ``1 / (width_falloff * i + 1)``.  Set to ``0.0``
        (default) for uniform weight across all rows.
    """

    stripe_half_width: int = 5
    stripe_half_length: int = 20
    length_falloff: float = 0.05
    width_falloff: float = 0.0
