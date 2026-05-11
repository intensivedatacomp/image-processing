"""Kernel generation module for edge detection."""

import numpy as np
import PIL.Image
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
        Nx2 = 5
        Ny2 = 20
        Nv2 = 20
        emask = torch.zeros((2 * Nv2, 2 * Nv2))
        for i in range(0, Nx2):
            for j in range(-Ny2, Ny2):
                emask[Nv2 + i, Nv2 + j] = 1.0 / (0.05 * np.abs(j) + 1)
        emaskPIL = torchvision.transforms.ToPILImage()(emask)

        Nmasks = 10
        allmasks = []
        for phi in np.arange(0.0, 180.0, 180.0 / Nmasks):
            img = emaskPIL.rotate(phi, PIL.Image.NEAREST, expand=0)
            rt = torchvision.transforms.ToTensor()(img)
            rt = rt - torchvision.transforms.ToTensor()(
                PIL.ImageOps.mirror(PIL.ImageOps.flip(img))
            )
            allmasks.append(rt)

        return torch.cat(allmasks).view(len(allmasks), *emask.shape).to(self.device)
