"""Image processing package for GPU-accelerated edge detection.

Implements the elongated-mask edge detection method from
*Contour-texture separation: part 2* (Antal, 2024) as an extendable,
GPU-aware PyTorch package.

Quick start
-----------
>>> import torch
>>> from image_processing import EdgeDetector
>>> detector = EdgeDetector()                    # uses GPU if available
>>> image = torch.rand(3, 512, 512)              # (C, H, W) float image in [0, 1]
>>> edges = detector.detect(image)               # (H, W) edge map in [0, 1]

Customising the kernel
----------------------
>>> from image_processing import (
...     EdgeDetector,
...     ElongatedMaskKernel,
...     ElongatedMaskParams,
... )
>>> from image_processing.combination import sum_of_powers
>>> params = ElongatedMaskParams(
...     n_angles=18,
...     kernel_half_size=30,
...     stripe_half_width=5,
...     stripe_half_length=30,
...     length_falloff=0.1,
...     width_falloff=1.0,
... )
>>> detector = EdgeDetector(
...     kernel=ElongatedMaskKernel(params),
...     combine_fn=sum_of_powers(3.0),
... )

Extending with a custom kernel
-------------------------------
Subclass :class:`BaseKernel` and pair it with a custom
:class:`BaseKernelParams` dataclass::

    from dataclasses import dataclass
    import torch
    from image_processing import BaseKernel, BaseKernelParams

    @dataclass
    class MyParams(BaseKernelParams):
        sigma: float = 1.0

    class MyKernel(BaseKernel):
        def build(self) -> torch.Tensor:
            p: MyParams = self.params           # type: ignore[assignment]
            # ... build and return (N, kH, kW) tensor on self.device

Public API
----------
- :class:`EdgeDetector` - applies a kernel stack and combines the responses.
- :class:`ElongatedMaskKernel` - rotated anti-symmetric stripe kernels.
- :class:`ElongatedMaskParams` - parameters for :class:`ElongatedMaskKernel`.
- :class:`BaseKernel` - abstract base for custom kernel families.
- :class:`BaseKernelParams` - base dataclass for kernel parameters.
- :data:`CombineFn` - type alias for combination callables.
- :mod:`image_processing.combination` - built-in combination functions.
"""

from .combination import (
    CombineFn as CombineFn,
)
from .combination import (
    max_abs as max_abs,
)
from .combination import (
    sum_of_abs as sum_of_abs,
)
from .combination import (
    sum_of_powers as sum_of_powers,
)
from .combination import (
    sum_of_squares as sum_of_squares,
)
from .detector import EdgeDetector as EdgeDetector
from .kernels import (
    BaseKernel as BaseKernel,
)
from .kernels import (
    ElongatedMaskKernel as ElongatedMaskKernel,
)
from .params import (
    BaseKernelParams as BaseKernelParams,
)
from .params import (
    ElongatedMaskParams as ElongatedMaskParams,
)

__all__ = [
    "BaseKernel",
    "BaseKernelParams",
    "CombineFn",
    "EdgeDetector",
    "ElongatedMaskKernel",
    "ElongatedMaskParams",
    "max_abs",
    "sum_of_abs",
    "sum_of_powers",
    "sum_of_squares",
]
