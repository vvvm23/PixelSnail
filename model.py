import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import lru_cache, partial
import math

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
    def __init__(self, q_channel, k_channel, channel, nb_head=8, dropout=0.1):
        super().__init__()
        self.fc_query = nn.utils.weight_norm(nn.Linear(q_channel, channel))
        self.fc_key = nn.utils.weight_norm(nn.Linear(k_channel, channel))
        self.fc_value = nn.utils.weight_norm(nn.Linear(k_channel, channel))
        self.head_dim = channel // nb_head
        self.nb_head = nb_head

        self.drop = nn.Dropout(dropout)

    def forward(self, q, k):
        batch_size, _, height, width = k.shape
        reshape = lambda x: x.view(batch_size, -1, self.nb_head, self.head_dim).transpose(1, 2)

        # Flatten query and key values
        qf = q.view(batch_size, q.shape[1], -1).transpose(1, 2)
        kf = k.view(batch_size, k.shape[1], -1).transpose(1, 2)

        # Apply attention over flatten and return query, key and value
        q = reshape(self.fc_query(qf))
        k = reshape(self.fc_key(kf)).transpose(2, 3)
        v = reshape(self.fc_value(kf))

        # Apply causal masking to resulting attention vector
        attn = torch.matmul(q, k) / math.sqrt(self.head_dim)
        mask, start_mask = causal_mask(height*width)
        mask, start_mask = mask.type_as(q), start_mask.type_as(q)

        attn = attn.masked_fill(mask == 0, -1e4)
        attn = torch.softmax(attn, 3) * start_mask
        attn = self.drop(attn)

        # Query value using final attention probabilities
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch_size, height, width, self.head_dim*self.nb_head).permute(0, 3, 1, 2)
        return out

# Ported from numpy to torch. lru_cache for memoisation of masks
@lru_cache(maxsize=64)
def causal_mask(size):
    shape = [size, size]
    mask = torch.triu(torch.ones(shape), diagonal=1).to(torch.uint8).t()
    start_mask = torch.ones(size).to(torch.float32)
    start_mask[0] = 0

    return (
        mask.unsqueeze(0),
        start_mask.unsqueeze(1),
    )

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
    attn = CausalAttention(3, 3, 8)
    q = torch.randn(1, 3, 8, 8)
    k = torch.randn(1, 3, 8, 8)
    print(attn(q, k).shape)
