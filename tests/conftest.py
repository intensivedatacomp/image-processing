"""Setup functionalities for the tests."""

from typing import Tuple

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.utils.data import Dataset


class DummyImageDataset(Dataset[Tuple[Tensor, Tensor]]):  # type: ignore[misc]
    """Small synthetic dataset used for testing."""

    def __init__(
        self,
        n_samples: int = 50,
        image_shape: Tuple[int, int, int] = (1, 32, 32),
        num_classes: int = 3,
    ) -> None:
        """
        Define dummy dataset.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        image_shape : tuple[int, int, int]
            Image tensor shape.
        num_classes : int
            Number of target classes.
        """
        self.n_samples: int = n_samples
        self.image_shape: Tuple[int, int, int] = image_shape
        self.num_classes: int = num_classes

        self.data: Tensor = torch.randn(n_samples, *image_shape)
        self.targets: Tensor = torch.randint(0, num_classes, (n_samples,))

    def __len__(self) -> int:
        """Get number of samples."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Get an image."""
        return self.data[idx], self.targets[idx]


@pytest.fixture  # type: ignore[misc]
def dummy_dataset() -> DummyImageDataset:
    """Fixture returning a small dummy dataset."""
    return DummyImageDataset()


@pytest.fixture  # type: ignore[misc]
def deterministic_labels() -> Tuple[np.ndarray, np.ndarray]:
    """Fixture returning deterministic labels and predictions."""
    labels: np.ndarray = np.array([0, 1, 2, 1, 0])
    preds: np.ndarray = np.array([0, 1, 2, 0, 0])

    return labels, preds
