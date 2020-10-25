# `base.py` ensembles temporary versions of encoder bases for contrast

from .set import SetTransformer, MAB, SAB, ISAB, PMA
from .c2v import Code2VecEncoder
from .astnn import ASTnnEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSetEncoder(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SimpleSetEncoder, self).__init__()
        self.encoder = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
            PMA(dim_hidden, num_heads, num_outputs, ln=ln)
        )
    
    def forward(self, x):
        return self.encoder(x)


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
        return self.proj(self.base_encoder(x))