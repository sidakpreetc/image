import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, dim: int, internal_dim: int) -> None:
        super().__init__()
        
        self.mlp_layers = nn.Sequential(
            nn.Linear(dim, internal_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(internal_dim, dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        x = self.mlp_layers(x)
        return x
