import torch
import torch.nn as nn   

class StylizationBlock(nn.Module):
    def __init__(self, latent_dim: int, time_embed_dim: int, dropout: float):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        B, T, D = h.shape
        if emb.shape[-1] != self.time_embed_dim:
            emb_proj = nn.Linear(emb.shape[-1], self.time_embed_dim).to(emb.device)
            emb = emb_proj(emb)

        emb_out = self.emb_layers(emb).unsqueeze(1)
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h