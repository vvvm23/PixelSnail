import torch
import torch.nn
import torch.nn.functional as F

from functools import lru_cache

class PixelSnail(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

if __name__ == "__main__":
    print("Aloha, World!")
