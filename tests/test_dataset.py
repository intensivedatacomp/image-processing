"""Testing datasets."""

import os
import tempfile

import numpy as np
from PIL import Image

from simpletorchtrainer.dataset import ImageFolderDataset


def create_fake_dataset(root: str) -> None:
    """Create a small fake ImageFolder dataset."""
    classes = ["cat", "dog"]

    for c in classes:
        os.makedirs(os.path.join(root, c))

        for i in range(3):
            img = (np.random.rand(64, 64) * 255).astype(np.uint8)
            image = Image.fromarray(img)

            path = os.path.join(root, c, f"img{i}.png")

            image.save(path)


def test_imagefolder_dataset_loading() -> None:
    """Test folder loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        create_fake_dataset(tmpdir)

        dataset = ImageFolderDataset(tmpdir, batch_size=2)

        assert dataset.num_classes == 2

        train_batch = next(iter(dataset.train_loader))

        images, labels = train_batch

        assert images.shape[0] <= 2
