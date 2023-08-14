import torch
import torch.nn as nn
from .attention import MultiHeadSelfAttention
from .mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int=8, dropout: float=0.1, linear_dim: int=1024):
        super().__init__()

        self.mhsa = MultiHeadSelfAttention(dim=dim, heads=heads)

        self.dropout = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, linear_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None):

        out = self.mhsa(x, mask)
        out = self.dropout(x) + x
        out = self.norm_1(out)

        out = self.mlp(out) + x
        out = self.norm_2(out)

        return out
    

class TransformerEncoder(nn.Module):
    def __init__(self, dim: int, linear_dim: int, layers: int, heads: int=8, dropout=0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(dim, heads, dropout, linear_dim) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
