import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import lru_cache, partial

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
    def __init__(self, in_channel, channel, kernel_size, conv='wnconv2d', dropout=0.1, condition_dim=0):
        super().__init__()

        assert conv in ['wnconv2d', 'causal_downright', 'causal'], "Invalid conv argument [wnconv2d, causal_downright, causal]"
        # partial is amazing! should use more!
        if conv == 'wnconv2d':
            conv_builder = partial(WNConv2d, padding=kernel_size // 2)
        elif conv == 'causal_downright':
            conv_builder = partial(CausalConv2d, padding='downright')
        elif conv == 'causal':
            conv_builder = partial(CausalConv2d, padding='causal')

        self.conv1 = conv_builder(in_channel, channel, kernel_size)
        self.conv2 = conv_builder(channel, in_channel*2, kernel_size)
        self.drop1 = nn.Dropout(dropout)

        if condition_dim > 0:
            self.convc = WNConv2d(condition_dim, in_channel*2, 1, bias=False)
        self.gate = nn.GLU(1) # 0 -> 1 === ReZero -> Residual

    def forward(self, x, c=None):
        y = F.elu(self.conv1(F.elu(x)))
        y = self.drop1(y)
        y = self.conv2(y)

        if c != None:
            y = self.convc(c) + y
        y = self.gate(y) + x
        return y

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
    x = torch.randn(1, 3, 4, 4)
    blk = GatedResBlock(3, 8, 3, conv='wnconv2d', dropout=0.1)
    print(blk(x).shape)
    blk = GatedResBlock(3, 8, 3, conv='causal_downright', dropout=0.1)
    print(blk(x).shape)
    blk = GatedResBlock(3, 8, 3, conv='causal', dropout=0.1)
    print(blk(x).shape)
    blk = GatedResBlock(3, 8, 3, conv='causal', dropout=0.1, condition_dim=1)
    print(blk(x, torch.randn(1,1,4,4)).shape)
