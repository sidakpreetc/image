import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from .utils.transformer import TransformerEncoder


class ViT(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_dim, classification=True, num_classes=1):
        super().__init__()

        self.patch_dim = patch_dim
        self.classification = classification
        self.num_tokens = (img_dim // patch_dim) ** 2
        self.token_dim = in_channels * (patch_dim ** 2)

        self.projection = nn.Linear(self.token_dim, embedding_dim)
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.dropout = nn.Dropout(0.1)
        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num)

        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x: torch.Tensor):
        patches = rearrange(x,
                  'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                  patch_x=self.patch_dim, patch_y=self.patch_dim)
        batch_size, tokens, _ = patches.shape
        projections = self.projection(patches)
        token = repeat(
            self.cls_token,
            'b ... -> (b batch_size) ...',
            batch_size=batch_size
        )
        patches = torch.cat([token, projections], dim=1)
        patches += self.embedding[:tokens+1, :]

        x = self.dropout(patches)
        x = self.transformer(x)
        x = self.mlp_head(x[:, 0, :] if self.classification else x[:, 1:, :])
        return x
