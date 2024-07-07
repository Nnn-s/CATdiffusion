from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

from ldm.modules.diffusionmodules.util import checkpoint
from ldm.modules.attention import *






class Fusiontransformer(nn.Module):

    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 use_checkpoint=True):
        super().__init__()
        inner_dim = n_heads * d_head
        self.in_channels = in_channels
        self.norm = Normalize(context_dim)
        self.proj_in = nn.Linear(context_dim,inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=in_channels,
                                    checkpoint=use_checkpoint)
                for _ in range(depth)]
        )




        self.proj_out = zero_module(nn.Linear(inner_dim,context_dim))


    def forward(self, x,context=None):
        b, c, h, w = x.shape
        context_in = context
        context = self.norm(context)
        context = self.proj_in(context)
        #x == 1,257,1024
        for block in self.transformer_blocks:
            context = block(context,x)
        context = self.proj_out(context)

        return context + context_in
