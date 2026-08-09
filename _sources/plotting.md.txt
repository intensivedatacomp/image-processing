# Plotting Kernels

`image_processing.visualization` provides two functions for inspecting a
kernel bank, plus matching convenience methods on `BaseKernel`. Both work
with any `BaseKernel` subclass (not just `ElongatedMaskKernel`), as long as
its `params` is a dataclass.

---

## 1. All orientations in one grid

```python
from image_processing import ElongatedMaskKernel, plot_kernel_grid

kernel = ElongatedMaskKernel()
plot_kernel_grid(kernel, save_path="kernel_grid.png")
```

Or as a method directly on the kernel:

```python
kernel.plot_all(save_path="kernel_grid.png")
```

This renders every orientation in a `cols`-wide grid (default `6`), each
titled with its angle in degrees, and a shared colorbar. The figure's
`suptitle` also lists the kernel's parameter values, read automatically from
the dataclass fields of `kernel.params`.

**Signature**

```python
plot_kernel_grid(
    kernel: BaseKernel,
    cols: int = 6,
    save_path: str | Path | None = None,
    show: bool = True,
) -> matplotlib.figure.Figure
```

---

## 2. A single orientation

```python
from image_processing import plot_single_kernel

plot_single_kernel(kernel, index=0, save_path="kernel_single.png")
```

Or:

```python
kernel.plot(index=0, save_path="kernel_single.png")
```

Renders one orientation at a larger size, titled with the kernel class name,
angle, and spatial dimensions.

**Signature**

```python
plot_single_kernel(
    kernel: BaseKernel,
    index: int = 0,
    save_path: str | Path | None = None,
    show: bool = True,
) -> matplotlib.figure.Figure

# Raises IndexError if `index` is out of range for the kernel bank.
```

---

## 3. Notes

- Both functions build the kernel bank if it isn't cached yet (via
  `kernel.kernels`), then move it to CPU/NumPy for plotting — the original
  tensor stays on its original device.
- `show=False` skips `plt.show()`, useful for headless/CI runs or when you
  only want to save to disk.
- `save_path` accepts anything Matplotlib's `savefig` accepts (`str` or
  `pathlib.Path`); the file format is inferred from the extension.
- Kernel weights use a diverging colormap (`RdBu_r`) centered at zero, so
  positive and negative weights are easy to tell apart at a glance.

## 4. Example

```python
from image_processing import ElongatedMaskKernel, ElongatedMaskParams

params = ElongatedMaskParams(n_angles=18, kernel_half_size=30)
kernel = ElongatedMaskKernel(params, device="cpu")

kernel.plot_all(save_path="grid.png")      # overview of all 18 orientations
kernel.plot(index=3, save_path="k3.png")   # close-up of the 4th orientation
```
