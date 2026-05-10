"""Testing trainer."""

from typing import Protocol, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from simpletorchtrainer.dataset import ImageFolderDataset
from simpletorchtrainer.results import TrainingResults
from simpletorchtrainer.trainer import Trainer

from .conftest import DummyImageDataset


class TinyModel(nn.Module):  # type: ignore[misc]
    """Very small neural network used for testing."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define forward method."""
        return self.net(x)


class _DatasetLike(Protocol):
    """Protocol matching the interface Trainer expects."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_classes: int


class DummyWrapper:
    """Wrapper mimicking ImageFolderDataset for testing."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_classes: int

    def __init__(self) -> None:
        dataset: Dataset = DummyImageDataset(n_samples=40)

        train, val, test = random_split(dataset, [20, 10, 10])

        self.train_loader = DataLoader(train, batch_size=4)
        self.val_loader = DataLoader(val, batch_size=4)
        self.test_loader = DataLoader(test, batch_size=4)

        self.num_classes = 3


def test_trainer_runs() -> None:
    """Test trainer runs."""
    dataset = DummyWrapper()

    trainer = Trainer(
        model_class=TinyModel,
        num_epochs=2,
    )

    # Cast to satisfy strict Trainer typing
    results = trainer.train(cast(ImageFolderDataset, dataset))

    assert isinstance(results, TrainingResults)

    assert len(results.train_losses) == 2
    assert len(results.val_accuracies) == 2

    assert 0 <= results.test_accuracy <= 1
