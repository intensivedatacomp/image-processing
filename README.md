# Image processing

## Overview

This repository implements a GPU-accelerated image processing framework focused on edge detection using multi-orientation convolutional kernels. The system is designed as a modular research codebase for experimenting with structured filter banks and response aggregation strategies in computer vision.

The core implementation is based on an extendable edge detection pipeline inspired by *Contour-texture separation* methodologies, with emphasis on orientation-aware filtering and differentiable computation using PyTorch.

---

## Scientific Context

Edge detection remains a fundamental problem in computer vision, particularly in separating structural contours from texture-induced noise. This framework explores a kernel-based approach where multiple oriented convolution masks are applied in parallel, and their responses are aggregated into a unified edge-strength representation.

The design prioritizes:
- Orientation sampling in filter banks
- GPU acceleration via PyTorch
- Modular combination functions for response fusion
- Experimentation with structured kernel families (including elongated masks)

---

## Core Features

- Multi-orientation convolutional edge detection
- GPU-aware execution (CUDA or CPU fallback)
- Extensible kernel system
- Pluggable response aggregation strategies
- Research-oriented modular architecture
- Reproducible experimental setup with test coverage

---

## Architecture

The package is structured into four main components:

### 1. `EdgeDetector`
Located in `detector.py`

Responsible for:
- Applying a stack of convolution kernels to input images
- Managing device placement (CPU/GPU)
- Aggregating per-orientation responses into final edge maps

Pipeline:
1. Input tensor normalization and device transfer
2. Kernel response computation (per orientation)
3. Combination of responses into a single edge map

---

### 2. Kernel System
Located in `kernels.py`

Defines the convolutional building blocks:
- `BaseKernel`: abstract kernel interface
- Specialized kernel families (e.g., elongated mask kernels)
- Device resolution utilities
- Conversion between parameter definitions and tensor kernels

Kernels are dynamically generated based on parameter dataclasses.

---

### 3. Parameter Definitions
Located in `params.py`

Defines configuration structures for kernel generation:

- `BaseKernelParams`
  - `n_angles`: number of orientations in `[0°, 180°)`
  - `kernel_half_size`: spatial extent of convolution kernel

These parameters control resolution and angular sampling density.

---

### 4. Response Combination
Located in `combination.py`

Implements functions that aggregate multi-orientation responses:

- Input: tensor of shape `(C, N, H, W)`
- Output: aggregated edge map `(H, W)`

Includes differentiable reduction strategies such as:
- Sum-of-squares aggregation (`sum_of_squares`)
- Additional configurable combination operators

## Commit Workflow

This project uses feature branches and pull requests for development.

### 1. Create a feature branch
```bash
git switch -c feature/your-awesome-feature
```
### 2. Make changes and commit
```bash
git add .
git commit -m "Useful commit message"
```
### 3. Push the branch to remote
```bash
git push --set-upstream origin feature/your-awesome-feature
```
### 4. Open a Pull Request
```bash
xdg-open https://github.com/dcintlab/image-processing/pull/new/feature/your-awesome-feature
```
### 5. After merge (cleanup)
```bash
git switch main
git pull
git branch -d feature/your-awesome-feature #Delete local branch
git push origin --delete feature/your-awesome-feature #Delete remote branch
```
