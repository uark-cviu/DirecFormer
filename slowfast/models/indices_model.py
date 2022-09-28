import math
from functools import partial
import torch
import torch.nn as nn


class IndicesHead(nn.Module):
    def __init__(self, dim_in, dim_out, pool_size, stride=1, padding=0, upsample=1) -> None:
        super(IndicesHead, self).__init__()
        self.dim_out = dim_out
        self.pool = nn.AvgPool3d(pool_size, stride=stride, padding=padding)
        self.linear = nn.Linear(dim_in, dim_out**upsample)
        self.head = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.pool(x)
        # (N, C, T, H, W) -> (N, T, C, H, W) -> (N*T, C, H, W)
        n, c, t, _, _ = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(n*t, c)
        x = self.linear(x)
        # (N*T, C) -> (N, T, C)
        x = x.reshape(n, self.dim_out, -1)
        x = self.head(x)
        return x
