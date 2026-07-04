# Usage Guide — Image Processing Research Package

This document describes the standardized usage of the Image Processing framework contained in this repository. The implementation is designed for research-grade experiments in GPU-accelerated edge detection based on oriented elongated mask kernels (Antal, 2024: *Contour-texture separation, Part 2*). The API is intentionally minimal, deterministic, and fully compatible with PyTorch execution workflows.

---

## 1. Installation

This project is a Python package managed via `pyproject.toml`.

### Recommended setup (development / research environment)

```bash
git clone <repository_url>
cd image-processing-main

python -m venv venv
source venv/bin/activate   # Linux / macOS
# venv\Scripts\activate    # Windows

pip install -r requirements.txt
pip install -e .
````

### Optional dependencies

For development and testing:

```bash
pip install -r requirements-dev.txt
```

For documentation build:

```bash
pip install -r requirements-docs.txt
```

---

## 2. Core Concept

The system implements a multi-orientation convolutional edge detection pipeline:

1. A stack of oriented kernels is generated (default: elongated mask kernels).
2. Each kernel is applied via 2D convolution over RGB or grayscale images.
3. The resulting response tensor is aggregated into a single edge-strength map.
4. Optional normalization scales output into `[0, 1]`.

All operations are executed using PyTorch and are GPU-compatible.

---

## 3. Basic Usage

### Minimal Example

```python
import torch
from image_processing import EdgeDetector

# Initialize detector (automatically uses GPU if available)
detector = EdgeDetector()

# Input image format: (C, H, W), float32 in [0, 1]
image = torch.rand(3, 512, 512)

# Compute edge map
edges = detector.detect(image)

print(edges.shape)  # torch.Size([512, 512])
```

---

## 4. Expected Input / Output Format

### Input

* Type: `torch.Tensor`
* Shape: `(C, H, W)`
* Channels:

  * 1 = grayscale
  * 3 = RGB
* Value range: `[0.0, 1.0]`
* Device: CPU or CUDA (auto-migrated internally)

### Output

* Type: `torch.Tensor`
* Shape: `(H, W)`
* Value range: `[0.0, 1.0]` (if normalization enabled)

---

## 5. Advanced Configuration

### 5.1 Custom Kernel Configuration

The detector supports full customization of the kernel generation process.

```python
from image_processing import EdgeDetector, ElongatedMaskKernel, ElongatedMaskParams

params = ElongatedMaskParams(
    n_angles=18,
    kernel_half_size=30,
    stripe_half_width=5,
    stripe_half_length=30,
    length_falloff=0.1,
    width_falloff=1.0,
)

kernel = ElongatedMaskKernel(params=params)

detector = EdgeDetector(kernel=kernel)
```

---

### 5.2 Custom Combination Function

By default, responses are aggregated using a sum-of-squares strategy. This can be replaced:

```python
from image_processing import EdgeDetector
from image_processing.combination import sum_of_powers

def custom_combine(x):
    return sum_of_powers(x, p=3)

detector = EdgeDetector(combine_fn=custom_combine)
```

---

### 5.3 Disabling Normalization

```python
detector = EdgeDetector(normalize=False)
```

This returns raw edge intensity values without scaling.

---

## 6. Pipeline Behavior (Technical Specification)

Given an input tensor `I ∈ R^(C×H×W)`:

1. Tensor is cast to `float32`

2. Moved to kernel device (CPU/CUDA)

3. Convolution applied:

   ```
   R[c, n] = conv2d(I[c], K[n])
   ```

4. Response tensor shape:

   ```
   R ∈ R^(C × N × H × W)
   ```

5. Aggregation:

   ```
   E = combine_fn(R)
   ```

6. Optional normalization:

   ```
   E = E / max(E)
   ```

---

## 7. Example Research Workflow

```python
import torch
from image_processing import EdgeDetector

detector = EdgeDetector()

dataset = torch.rand(10, 3, 256, 256)  # example batch

results = []

for i in range(len(dataset)):
    image = dataset[i]
    edges = detector.detect(image)
    results.append(edges)
```

---

## 8. Reproducibility Notes

* All kernel generation is deterministic given identical parameters.
* GPU execution may introduce minor floating-point variance.
* Results are stable across CPU/GPU for identical PyTorch versions.
* Recommended environment: Python ≥ 3.10, PyTorch ≥ 2.0

---

## 9. Performance Considerations

* GPU execution is strongly recommended for `kernel_half_size > 20`
* Larger `n_angles` increases computational complexity linearly
* Memory usage scales with `(C × N × H × W)`
* Use `torch.no_grad()` during inference to reduce overhead

---

## 10. Common Issues

### Shape mismatch

Ensure input is `(C, H, W)` not `(H, W, C)`.

### Device mismatch

All tensors are automatically moved, but manual tensors passed into custom functions must match device.

### Slow execution

Reduce:

* `n_angles`
* kernel size
* input resolution

---

## 11. Summary

This package provides a modular, research-oriented implementation of oriented kernel-based edge detection. It is designed for experimental reproducibility, extensibility of kernel design, and GPU-accelerated processing of image data in scientific workflows.
