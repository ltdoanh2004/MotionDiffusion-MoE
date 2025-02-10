import torch.nn as nn
import torch

class GatedFusion(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj_time = nn.Linear(embed_dim, embed_dim)
        self.proj_text = nn.Linear(embed_dim, embed_dim)
        self.sigmoid = nn.Sigmoid()
        self.post_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, time_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        t = self.proj_time(time_emb)
        x = self.proj_text(text_emb)
        gating = self.sigmoid(t + x)
        fused = gating * t + (1 - gating) * x
        fused = self.post_mlp(fused)
        return fused
