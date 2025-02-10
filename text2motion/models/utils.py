import torch.nn as nn
import torch
from torch.nn import GroupNorm
class NormalizationBlock(nn.Module):
    def __init__(self, dim, num_groups=8):
        super().__init__()
        self.norm = nn.Sequential(
            GroupNorm(num_groups, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x):
        return self.norm(x)
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    Used for final output layer to stabilize training.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module 