import torch.nn as nn
import torch
from stylization import StylizationBlock
from switch_moe import SwitchMoELayer
class MultiBranchFFN(nn.Module):
    def __init__(self, latent_dim: int, ffn_dim: int,
                 num_branches: int = 4, dropout: float = 0.1,
                 time_embed_dim: int = 512):
        super().__init__()
        self.num_branches = num_branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, latent_dim)
            ) for _ in range(num_branches)
        ])
        
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        out = 0
        for branch in self.branches:
            out += branch(x)
        out = out / self.num_branches
        out = x + self.proj_out(out, emb)
        return out

class MoEMultiBranchFFN(nn.Module):
    def __init__(self, latent_dim: int, ffn_dim: int,
                 num_experts: int = 8,
                 num_branches: int = 2,
                 dropout: float = 0.1,
                 time_embed_dim: int = 512):
        super().__init__()
        self.num_branches = num_branches
        self.branches = nn.ModuleList()

        for _ in range(num_branches):
            branch = nn.ModuleDict({
                "layernorm": nn.LayerNorm(latent_dim),
                # We'll do top-2 gating inside SwitchMoELayer
                "moe": SwitchMoELayer(latent_dim, ffn_dim, num_experts=num_experts),
                "drop": nn.Dropout(dropout),
            })
            self.branches.append(branch)
        
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        out = 0
        for b in self.branches:
            h = b["layernorm"](x)
            h = b["moe"](h)
            h = b["drop"](h)
            out += h
        out = out / self.num_branches
        out = x + self.proj_out(out, emb)
        return out
