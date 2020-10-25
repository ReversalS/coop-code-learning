# `base.py` ensembles temporary versions of encoder bases for contrast

from .c2v import Code2VecEncoder
from .astnn import ASTnnEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelWithProjection(nn.Module):
    """
    Combine encoder with a projection head
    TODO: try different projection head settings
    """
    def __init__(self, base_encoder, out_dim):
        super(ModelWithProjection, self).__init__()
        self.base_encoder = base_encoder
        self.proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim)
            # nn.ReLU(inplace=True),
            # nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        x = self.base_encoder(x)
        return self.proj(x)
