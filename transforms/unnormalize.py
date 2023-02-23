import torch
from torch.nn import Module


class UnNormalize(Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        mean = torch.tensor(self.mean).reshape((1, -1, 1, 1))
        std = torch.tensor(self.std).reshape((1, -1, 1, 1))
        return tensor * std + mean

