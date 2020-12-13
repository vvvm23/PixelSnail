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
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]

        if padding == 'downright':
            pad = [kernel_size[1] - 1, 0, kernel_size[0] - 1, 0]
        elif padding in ['down', 'causal']:
            pad = [kernel_size[1] // 2, kernel_size[1] // 2, kernel_size[0] - 1, 0]

        self.causal = kernel_size[1] // 2 if padding == 'causal' else 0
        self.pad = nn.ZeroPad2d(pad)
        self.conv = WNConv2d(in_channel, out_channel, kernel_size, stride=stride)

    def forward(self, x):
        x = self.pad(x)
        if self.causal: # Zero out elements that would be inaccessible during sampling
            self.conv.weight_v.data[:, :, -1, self.causal:].zero_()
        x = self.conv(x)
        return x

class GatedResBlock(nn.Module):
    def __init__(self, in_channel, channel, kernel_size, conv='wnconv2d', dropout=0.1, condition_dim=0, aux_channels=0):
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

        if aux_channels > 0:
            self.aux_conv = WNConv2d(aux_channels, channel, 1)

        if condition_dim > 0:
            self.convc = WNConv2d(condition_dim, in_channel*2, 1, bias=False)
        self.gate = nn.GLU(1) # 0 -> 1 === ReZero -> Residual

    def forward(self, x, a=None, c=None):
        y = self.conv1(F.elu(x))

        if a != None:
            y = y + self.aux_conv(F.elu(a))
        y = F.elu(y)

        y = self.drop1(y)
        y = self.conv2(y)

        if c != None:
            y = self.convc(c) + y
        y = self.gate(y) + x
        return y

# TODO: Potentially add some linear attention if time allows
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
    def __init__(self, in_channel, channel, kernel_size, nb_res_blocks, dropout=0.1, condition_dim=0):
        super().__init__()
        self.res_blks = nn.ModuleList([GatedResBlock(in_channel, channel, kernel_size, conv='causal', dropout=dropout, condition_dim=condition_dim) for _ in range(nb_res_blocks)])
        self.k_blk = GatedResBlock(in_channel*2 + 2, in_channel, 1, dropout=dropout)
        self.q_blk = GatedResBlock(in_channel + 2, in_channel, 1, dropout=dropout)

        self.attn = CausalAttention(in_channel + 2, in_channel*2 + 2, in_channel // 2, dropout=dropout)

        self.out_blk = GatedResBlock(in_channel, in_channel, 1, dropout=dropout, aux_channels=in_channel // 2)

    def forward(self, x, bg, c=None):
        y = x
        for blk in self.res_blks:
            y = blk(y, c=c)

        k = self.k_blk(torch.cat([x, y, bg], 1))
        q = self.q_blk(torch.cat([y, bg], 1))
        y = self.out_blk(y, a=self.attn(q, k))

        return y

class PixelSnail(nn.Module):
    def __init__(self, shape, nb_class, channel, kernel_size, nb_pixel_block, nb_res_block, res_channel, dropout=0.1, nb_cond_res_block=0, cond_res_channel=0, cond_res_kernel=3, nb_out_res_block=0):
        super().__init__()
        height, width = shape
        self.nb_class = nb_class

        assert kernel_size % 2, "Kernel size must be odd"

        self.horz_conv = CausalConv2d(nb_class, channel, [kernel_size // 2, kernel_size], padding='down')
        self.vert_conv = CausalConv2d(nb_class, channel, [(kernel_size+1) // 2, kernel_size // 2], padding='downright')

        coord_x = (torch.arange(height).float() - height / 2) / height
        coord_x = coord_x.view(1, 1, height, 1).expand(1, 1, height, width)
        coord_y = (torch.arange(width).float() - width / 2) / width
        coord_y = coord_y.view(1, 1, 1, width).expand(1, 1, height, width)
        self.register_buffer('bg', torch.cat([coord_x, coord_y], 1))

        self.blks = nn.ModuleList([PixelBlock(channel, res_channel, kernel_size, nb_res_block, dropout=dropout, condition_dim=cond_res_channel) for _ in range(nb_pixel_block)])
        
        if nb_cond_res_block > 0:
            cond_net = [WNConv2d(nb_class, cond_res_channel, cond_res_kernel, padding=cond_res_kernel // 2)]
            cond_net.extend([GatedResBlock(cond_res_channel, cond_res_channel, cond_res_kernel) for _ in range(nb_cond_res_block)])
            self.cond_net = nn.Sequential(*cond_net)

        out = []
        for _ in range(nb_out_res_block):
            out.append(GatedResBlock(channel, res_channel, 1))
        out.append(nn.ELU(inplace=True))
        out.append(WNConv2d(channel, nb_class, 1))

        self.out = nn.Sequential(*out)

        self.shift_down = lambda x, size=1: F.pad(x, [0,0,size,0])[:, :, :x.shape[2], :]
        self.shift_right = lambda x, size=1: F.pad(x, [size,0,0,0])[:, :, :, :x.shape[3]]

    # cache is used to increase speed of sampling
    def forward(self, x, c=None, cache=None):
        if cache is None:
            cache = {}
        batch, height, width = x.shape
        x = F.one_hot(x, self.nb_class).permute(0, 3, 1, 2).type_as(self.bg)
        
        horz = self.shift_down(self.horz_conv(x))
        vert = self.shift_right(self.vert_conv(x))
        y = horz + vert

        bg = self.bg[:, :, :height, :].expand(batch, 2, height, width)

        if c != None:
            if 'c' in cache:
                c = cache['c']
                c = c[:, :, :height, :]
            else:
                c = F.one_hot(c, self.nb_class).permute(0,3,1,2).type_as(self.bg)
                c = self.cond_net(c)
                c = F.interpolate(c, scale_factor=4) # TODO: Could have some tranposed Conv2d instead
                cache['condition'] = c.detach().clone()
                c = c[:, :, :height, :]

        for blk in self.blks:
            y = blk(y, bg, c=c)
        y = self.out(y)
        return y, cache

if __name__ == "__main__":
    ps = PixelSnail([6, 6], 512, 64, 7, 7, 2, 32, cond_res_channel=32, nb_out_res_block=5)
    x = torch.LongTensor(1, 6, 6).random_(0, 255)
    print(ps(x)[0].shape)

    x = torch.LongTensor(1, 24, 24).random_(0, 255)
    c = torch.LongTensor(1, 6, 6).random_(0, 255)
    ps = PixelSnail([24, 24], 512, 64, 7, 7, 2, 32, nb_cond_res_block=3, cond_res_channel=32, nb_out_res_block=5)
    print(ps(x, c=c)[0].shape)
