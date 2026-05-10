"""Testing metrics."""

from typing import Tuple

import numpy as np

from simpletorchtrainer.metrics import classification_metrics


def test_classification_metrics(
    deterministic_labels: Tuple[np.ndarray, np.ndarray],
) -> None:
    """Test metric computation."""
    labels, preds = deterministic_labels

    acc, prec, rec, f1 = classification_metrics(labels, preds)

    assert isinstance(acc, float)
    assert isinstance(prec, float)
    assert isinstance(rec, float)
    assert isinstance(f1, float)

    assert 0 <= acc <= 1
    assert 0 <= prec <= 1
    assert 0 <= rec <= 1
    assert 0 <= f1 <= 1
