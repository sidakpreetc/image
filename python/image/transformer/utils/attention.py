import torch
import torch.nn as nn
import numpy as np


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // self.num_heads
        assert dim % self.num_heads == 0, f"Dimensions {dim} should be divisible by number of heads {num_heads}"

        self.dk = self.head_dim ** (-1 / 2)
        
        self.query_projection = nn.Linear(dim, self.head_dim * num_heads, bias=False)
        self.key_projection = nn.Linear(dim, self.head_dim * num_heads, bias=False)
        self.value_projection = nn.Linear(dim, self.head_dim * num_heads, bias=False)

        self.softmax = nn.Softmax()
        self.final = nn.Linear(num_heads*self.head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        energy: torch.Tensor = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk

        if mask:
            energy = energy.masked_fill(mask, -np.inf)

        attention = self.softmax(energy)
        out = torch.einsum("... i j , ... j d -> ... i d", attention, value)
        out = self.final(out)
        return out
