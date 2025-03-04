import math
import torch
import torch.nn as nn
class LearnableTimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int, max_period: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def _sinusoidal_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) 
            * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
            / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        sin_emb = self._sinusoidal_embedding(timesteps, self.embed_dim)
        out = self.mlp(sin_emb)
        return out



class StochasticDepth(nn.Module):
    def __init__(self, module: nn.Module, survival_prob: float = 1.0):
        super().__init__()
        self.module = module
        self.survival_prob = survival_prob

    def forward(self, *args, **kwargs):
        if not self.training or self.survival_prob == 1.0:
            return self.module(*args, **kwargs)
        if torch.rand(1).item() < self.survival_prob:
            return self.module(*args, **kwargs)
        else:
            if len(args) > 0 and isinstance(args[0], torch.Tensor):
                return args[0]
            return None