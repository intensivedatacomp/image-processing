"""Basic edge detection with the image_processing package.

Run with:
    python examples/basic_usage.py
"""

import torch

from image_processing import EdgeDetector, ElongatedMaskKernel, ElongatedMaskParams
from image_processing.combination import (
    max_abs,
    sum_of_abs,
    sum_of_powers,
    sum_of_squares,
)

# ---------------------------------------------------------------------------
# 1. Create a synthetic test image (C, H, W) float32 in [0, 1]
# ---------------------------------------------------------------------------
torch.manual_seed(42)
image = torch.rand(3, 64, 64)
print(f"Input image shape : {image.shape}")

# ---------------------------------------------------------------------------
# 2. Detect edges with default settings (GPU if available, otherwise CPU)
# ---------------------------------------------------------------------------
detector = EdgeDetector()
edges = detector.detect(image)

print(f"Output edge map   : {edges.shape}")
print(f"Value range       : [{edges.min():.4f}, {edges.max():.4f}]")
print(f"Running on device : {detector.kernel.device}")

# ---------------------------------------------------------------------------
# 3. Try different combination functions
# ---------------------------------------------------------------------------
print("\nCombination function comparison:")
print(f"  {'Function':<25} {'max value':>10}")
print(f"  {'-' * 37}")

kernel = ElongatedMaskKernel(device="cpu")  # pin to CPU for determinism

for name, fn in [
    ("sum_of_squares", sum_of_squares),
    ("sum_of_abs", sum_of_abs),
    ("max_abs", max_abs),
    ("sum_of_powers(3.0)", sum_of_powers(3.0)),
]:
    det = EdgeDetector(kernel=kernel, combine_fn=fn, normalize=False)
    out = det.detect(image)
    print(f"  {name:<25} {out.max().item():>10.4f}")

# ---------------------------------------------------------------------------
# 4. Customise kernel parameters (matches the notebook's GPU example)
# ---------------------------------------------------------------------------
params = ElongatedMaskParams(
    n_angles=18,  # finer angular resolution
    kernel_half_size=30,  # larger receptive field (61 x 61 kernel)
    stripe_half_width=5,  # wider stripe
    stripe_half_length=30,
    length_falloff=0.1,  # faster falloff along stripe axis
    width_falloff=1.0,  # add decay across stripe width
)
fine_kernel = ElongatedMaskKernel(params, device="cpu")
fine_detector = EdgeDetector(kernel=fine_kernel)

print(f"\nFine kernel shape : {fine_kernel.kernels.shape}")
edges_fine = fine_detector.detect(image)
print(f"Fine edge map     : {edges_fine.shape}")

# ---------------------------------------------------------------------------
# 5. Batch detection
# ---------------------------------------------------------------------------
images = [torch.rand(3, h, w) for h, w in [(32, 48), (64, 64), (16, 32)]]
maps = fine_detector.detect_batch(images)

print("\nBatch detection:")
for img, m in zip(images, maps, strict=True):
    print(f"  input {tuple(img.shape)} -> edge map {tuple(m.shape)}")
