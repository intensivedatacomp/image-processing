# Installation Guide — Image Processing
## Overview

This repository implements a modular image processing framework designed for research-oriented experimentation in classical and kernel-based image analysis methods. The system is structured for reproducibility, extensibility, and integration into scientific workflows.

The following installation procedure ensures a fully functional environment for development, testing, and experimental execution.

## System Requirements

The framework is compatible with modern Linux, macOS, and Windows environments.

### Required:

Python ≥ 3.10

pip ≥ 22.0

git
(Recommended) virtual environment support: venv, conda, or miniforge

### Optional but recommended:

* JupyterLab (for notebook-based experiments)
* pre-commit (for development integrity checks)
## 1. Clone the Repository
```bash
git clone https://github.com/intensivedatacomp/image-processing.git
cd image-processing
```
## 2. Create a Virtual Environment
### Option A — venv (standard Python)
```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
```
### Option B — Conda / Miniforge (recommended for research workflows)
```bash
conda create -n imgproc python=3.11 -y
conda activate imgproc
```
## 3. Install Dependencies
### Core dependencies
```bash
pip install -r requirements.txt
```
### Development dependencies (testing, linting, CI alignment)
```bash 
pip install -r requirements-dev.txt
```
### Documentation dependencies (optional)
```bash
pip install -r requirements-docs.txt
```
## 4. Install the Package in Editable Mode

This step is required for development and module-level imports.
```bash
pip install -e .
```
This enables live synchronization between source code and installed package, which is essential for research iteration cycles.

## 5. Pre-commit Hooks (Recommended for Contributors)

This repository enforces code quality through pre-commit hooks.
```bash
pre-commit install
```
To manually run all hooks:
```bash
pre-commit run --all-files
```
## 6. Running Tests

The project uses pytest for validation of image processing modules.
```bash
pytest
```
For verbose output:
```bash
pytest -v
```
### Test coverage includes:

* Kernel operations
* Detector logic
* Parameter validation
* Combination pipelines

## 7. Running Example Workflows

The repository includes reproducible examples for core functionality.

### Basic usage example
```bash
python examples/basic_usage.py
```
### Custom kernel demonstration
```bash
python examples/custom_kernel.py
```
### Jupyter-based edge detection experiment
```bash
jupyter notebook examples/edge_detection_demo.ipynb
```

## 8. Project Structure
```bash
src/image_processing/
    combination.py
    detector.py
    kernels.py
    params.py
```
### Core modules:

* kernels.py — convolutional and filter kernel definitions
* detector.py — edge and feature detection algorithms
* combination.py — pipeline composition utilities
* params.py — parameter validation and configuration layer

## 9. Development Notes

This framework is designed as a research-grade experimental environment, not a production inference library. As such:

* Deterministic behavior is prioritized where possible
* Functions are modular and independently testable
* Unit tests are required for all algorithmic changes
* CI pipeline enforces formatting and validation via pre-commit
## 10. Troubleshooting
### Import errors

If module imports fail after installation:
```bash
pip install -e .
```
### Pre-commit failing
```bash
pre-commit clean
pre-commit install
```
### Missing dependencies
Ensure environment consistency:
```bash
pip install -U pip setuptools wheel
pip install -r requirements.txt
```
## 11. Scientific Context

### This repository is intended for academic and applied research in:
* Classical image filtering
* Edge detection theory
* Kernel-based transformations
* Compositional image processing pipelines

Reproducibility and transparency are core design principles.
