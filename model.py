import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import lru_cache

class WNConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch.nn.utils.weight_norm(self)
        self.forward = super().forward

class GatedResBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class CausalAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class PixelBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class PixelSnail(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

if __name__ == "__main__":
    wn_conv = WNConv2d(1, 8, 3)
    print(wn_conv)
