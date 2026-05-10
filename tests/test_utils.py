"""Testing utils."""

import random

import numpy as np
import torch

from simpletorchtrainer.utils import seed_worker, set_seed


def test_seed_reproducibility() -> None:
    """Test that random generators become deterministic."""
    set_seed(42)

    a = np.random.rand()
    b = torch.rand(1).item()

    set_seed(42)

    a2 = np.random.rand()
    b2 = torch.rand(1).item()

    assert a == a2
    assert b == b2


def test_seed_worker() -> None:
    """Test seed_worker() function."""
    set_seed(42)
    seed_worker(0)
    a = random.randint(0, 100)
    set_seed(42)
    seed_worker(0)
    a2 = random.randint(0, 100)

    assert a == a2
