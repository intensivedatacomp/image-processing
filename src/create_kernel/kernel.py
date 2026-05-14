"""Kernel generation module for edge detection."""

import numpy as np
import PIL.Image
import PIL.ImageOps
import torch
import torchvision


class Kernel:
    """Convolution kernel for edge detection."""

    def __init__(self, device: torch.device | str) -> None:
        self.device = device

    def generate_mask(self) -> torch.Tensor:
        """Generate rotated edge detection masks.

        Returns
        -------
        torch.Tensor
            Tensor of edge detection masks.
        """
        nx2 = 5
        ny2 = 20
        nv2 = 20
        emask = torch.zeros((2 * nv2, 2 * nv2))
        for i in range(0, nx2):
            for j in range(-ny2, ny2):
                emask[nv2 + i, nv2 + j] = 1.0 / (0.05 * np.abs(j) + 1)
        emask_pil = torchvision.transforms.ToPILImage()(emask)

        nmasks = 10
        allmasks = []
        for phi in np.arange(0.0, 180.0, 180.0 / nmasks):
            img = emask_pil.rotate(phi, PIL.Image.NEAREST, expand=0)
            rt = torchvision.transforms.ToTensor()(img)
            rt = rt - torchvision.transforms.ToTensor()(
                PIL.ImageOps.mirror(PIL.ImageOps.flip(img))
            )
            allmasks.append(rt)

        return torch.cat(allmasks).view(len(allmasks), *emask.shape).to(self.device)
