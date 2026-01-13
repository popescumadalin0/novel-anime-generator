import torch
from torch import nn


class NoiseInjection(nn.Module):

    def __init__(self):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        if noise is None:
            batch, _, height, width = x.shape
            noise = torch.randn(batch, 1, height, width, device=x.device)

        return x + self.weight * noise