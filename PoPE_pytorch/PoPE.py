import torch
from torch.nn import Module

import torch.nn.functional as F

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

def apply_pope_to_qk(pope, q, k):
    return q, k

class PoPE(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

    def forward(self, q, k):
        return q, k
