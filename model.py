import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import lru_cache

# Inherits directly from nn.Conv2d and immediately normalises weights
# TODO: May be worth trying kaiming normal or something
class WNConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch.nn.utils.weight_norm(self)
        self.forward = super().forward

# Causal Convolutions with multiple padding types
class CausalConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding='downright'):
        super().__init__()
        assert padding in ['downright', 'down', 'causal'], "Invalid padding argument [downright, down, causal]"
        if padding == 'downright':
            pad = [kernel_size - 1, 0, kernel_size - 1, 0]
        elif padding in ['down', 'causal']:
            pad = [kernel_size // 2, kernel_size // 2, kernel_size - 1, 0]

        self.causal = kernel_size // 2 if padding == 'causal' else 0
        self.pad = nn.ZeroPad2d(pad)
        self.conv = WNConv2d(in_channel, out_channel, kernel_size, stride=stride)

    def forward(self, x):
        x = self.pad(x)
        if self.causal: # Zero out elements that would be inaccessible during sampling
            self.conv.weight_v.data[:, :, -1, self.causal:].zero_()
        x = self.conv(x)
        return x

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
    x = torch.randn(1, 1, 4, 4)
    cconv1 = CausalConv2d(1, 8, 3, stride=1, padding='downright')
    cconv2 = CausalConv2d(1, 8, 3, stride=1, padding='down')
    cconv3 = CausalConv2d(1, 8, 3, stride=1, padding='causal')
    print(cconv1(x).shape)
    print(cconv2(x).shape)
    print(cconv3(x).shape)
