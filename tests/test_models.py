"""Testing models."""

import torch

from simpletorchtrainer.models import DefaultCNN


def test_default_cnn_forward() -> None:
    """Ensure model forward pass works."""
    model = DefaultCNN(num_classes=5)

    x = torch.randn(2, 1, 128, 128)

    output = model(x)

    assert output.shape == (2, 5)
