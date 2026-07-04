# Development Guide

# Overview

This repository contains a research-oriented Python library focused on image processing, with a particular emphasis on convolution-based feature extraction and edge detection techniques. The system is designed for extensibility, reproducibility, and scientific validation. The architecture supports modular experimentation with kernels, detectors, and parameterized image transformation pipelines.

The project is implemented in Python (>= 3.12) and follows a strict package structure under `src/image_processing/`.

---

## Architecture

The codebase is organized into clearly separated modules:

- `kernels.py`
  Defines convolution kernels and kernel generation utilities. This module is responsible for low-level filter construction used in image transformations.

- `detector.py`
  Implements edge detection and feature extraction logic. This is the core computational component of the system, building on convolution operations.

- `combination.py`
  Provides utilities for combining multiple kernels and aggregating detection outputs. Used for ensemble-style filtering and composite feature maps.

- `params.py`
  Contains strongly typed parameter definitions and configuration structures used across the pipeline. Ensures reproducibility and experiment traceability.

- `__init__.py`
  Exposes the public API of the package.

---

## Development Principles

This project follows a research-grade development model:

- Deterministic outputs are preferred over stochastic behavior where possible.
- All transformations should be mathematically explicit and reproducible.
- No implicit global state is allowed in processing pipelines.
- Modules are designed to be independently testable and replaceable.
- Experimental extensions must not break backward compatibility of the core API.

---

## Environment Setup

### Requirements

- Python >= 3.12
- Dependencies defined in:
  - `requirements.txt` (runtime)
  - `requirements-dev.txt` (development)
  - `requirements-docs.txt` (documentation)

### Installation

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -r requirements-dev.txt
pip install -e .
````

---

## Development Workflow

### Branching Strategy

All new features must be developed in isolated branches:

```bash
git switch -c feature/your-feature-name
```

After implementation:

```bash
git add .
git commit -m "Clear and descriptive commit message"
git push --set-upstream origin feature/your-feature-name
```

Pull request creation:

```bash
xdg-open https://github.com/dcintlab/image-processing/pull/new/feature/your-feature-name
```

After merge:

```bash
git switch main
git pull
git branch -d feature/your-feature-name
git branch -d origin/feature/your-feature-name --remote
```

---

## Testing Strategy

The project uses `pytest` for validation.

Test coverage includes:

* Kernel correctness (`test_kernels.py`)
* Detector stability and output shape validation (`test_detector.py`)
* Parameter validation (`test_params.py`)
* Combination logic correctness (`test_combination.py`)

### Running Tests

```bash
pytest
```

Tests must remain deterministic and independent. Any test relying on external state is considered invalid.

---

## Pre-commit and CI

A pre-commit pipeline is enforced via:

* `.pre-commit-config.yaml`
* GitHub Actions workflow under `.github/workflows/pre-commit.yml`

Before pushing changes, developers should ensure:

```bash
pre-commit run --all-files
```

The CI pipeline validates:

* Code formatting consistency
* Linting rules
* Test execution
* Import integrity

## Adding New Features

When introducing a new feature:

1. Define scope clearly in a dedicated branch.
2. Extend or reuse existing kernel/detector abstractions.
3. Add corresponding unit tests.
4. Ensure no regression in existing tests.
5. Update documentation if public API changes.

Example feature workflow:

```text
feature/new-kernel-type → implementation → tests → CI validation → PR → merge
```

---

## Research Orientation

This project is not a generic utility library. It is intended as a controlled experimental environment for image processing research.

All contributions should respect the following constraints:

* Scientific clarity is prioritized over performance optimizations.
* Code must be explainable in academic or research contexts.
* Experimental reproducibility is a primary requirement.
* Any heuristic must be explicitly documented.
