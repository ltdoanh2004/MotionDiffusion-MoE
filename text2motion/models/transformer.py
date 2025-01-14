import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from transformers import AutoModel, AutoTokenizer

# ------------------------------------------------------------------------
# 1) Basic Utility and Helper Functions
# ------------------------------------------------------------------------

def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


# ------------------------------------------------------------------------
# 2) SwitchMoELayer with Top-2 Gating + Simple Load Balancing
# ------------------------------------------------------------------------
class SwitchMoELayer(nn.Module):
    """
    Enhanced Switch Transformer MoE layer with:
      - Top-2 gating
      - Simple load balancing
      - Thematic experts (foot, arm, torso, head, etc.) as an example

    We keep the same signature but internally do top-2 gating.
    """
    def __init__(self, input_dim, hidden_dim, num_experts=8):
        super().__init__()
        self.num_experts = num_experts
        # e.g., 4 "thematic" experts, or more:
        # We'll still rely on 'num_experts' though. In a real scenario,
        # you could name them specifically.
        self.gate = nn.Linear(input_dim, num_experts)

        # Create experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(num_experts)
        ])
        
        # Basic gating init
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        # For load balancing, track how many tokens each expert sees
        # (very simple approach).
        self.register_buffer("expert_usage", torch.zeros(num_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        Returns: (B, T, D) output
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # [B*T, D]
        
        # 1) Gating
        logits = self.gate(x_flat)  # [B*T, num_experts]
        probs = F.softmax(logits, dim=1)
        # top-2 gating
        top2_vals, top2_idx = torch.topk(probs, k=2, dim=1)  # each is [B*T, 2]
        
        # 2) Compute partial load balancing
        # We sum how many tokens go to each expert (based on top-1), for illustration
        top1_idx = top2_idx[:, 0]
        for e_idx in range(self.num_experts):
            usage_count = (top1_idx == e_idx).sum().item()
            self.expert_usage[e_idx] += usage_count

        # 3) Merge outputs from top-2 experts
        output = torch.zeros_like(x_flat)  # [B*T, D]

        for expert_idx in range(self.num_experts):
            # Mask for top-2 routing
            mask = (top2_idx == expert_idx)  # [B*T, 2]
            if not mask.any():
                continue
            # gather tokens that route to this expert
            # For each token that has expert_idx in top2_idx,
            # we add: top2_vals * expert(...) to 'output'.
            # Because top2_idx has shape [B*T, 2], let's do it carefully:

            # Flatten mask to [B*T], where True if top-2 includes expert_idx
            mask_any = mask.any(dim=1)
            if mask_any.any():
                expert_input = x_flat[mask_any]  # the tokens going to this expert
                # pass through the expert
                expert_output = self.experts[expert_idx](expert_input)
                
                # We need to scale by top2_vals. But each token might route with a different weight
                # we find which column belongs to expert_idx
                col = (top2_idx[mask_any] == expert_idx).nonzero()[:,1]  # either 0 or 1
                # gather the corresponding val
                val = top2_vals[mask_any, col]  # [#(mask_any)]
                val = val.unsqueeze(-1)  # shape [num_routed, 1]

                # Add to output
                output[mask_any] += val * expert_output
        
        return output.view(B, T, D)


# ------------------------------------------------------------------------
# 3) StylizationBlock (unchanged, except comments)
# ------------------------------------------------------------------------
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


# ------------------------------------------------------------------------
# 4) LearnableTimeEmbedding (unchanged, except comments)
# ------------------------------------------------------------------------
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


# ------------------------------------------------------------------------
# 5) GatedFusion (unchanged)
# ------------------------------------------------------------------------
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


# ------------------------------------------------------------------------
# 6) MultiBranchFFN (unchanged)
# ------------------------------------------------------------------------
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


# ------------------------------------------------------------------------
# 6B) MoEMultiBranchFFN (unchanged except now top-2 gating is in SwitchMoELayer)
# ------------------------------------------------------------------------
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


# ------------------------------------------------------------------------
# 7) StochasticDepth (unchanged)
# ------------------------------------------------------------------------
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


# ------------------------------------------------------------------------
# 8) GatedCrossAttention (unchanged, but let's add an "adaptive gating factor")
# ------------------------------------------------------------------------
class LinearTemporalCrossAttention(nn.Module):
    def __init__(self, seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

        # adaptive gating factor
        self.adaptive_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, xf, emb):
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head

        # same old approach
        q = F.softmax(self.query(self.norm(x)).view(B, T, H, -1), dim=-1)
        k = F.softmax(self.key(self.text_norm(xf)).view(B, N, H, -1), dim=1)
        v = self.value(self.text_norm(xf)).view(B, N, H, -1)

        attention = torch.einsum('bnhd,bnhl->bhdl', k, v)  # [B,H,D,D]
        y = torch.einsum('bnhd,bhdl->bnhl', q, attention).reshape(B, T, D)

        # adaptive gate
        alpha = torch.sigmoid(self.adaptive_gate)  # in [0,1]
        out_attn = x + alpha * self.proj_out(y, emb)
        return out_attn

class GatedCrossAttention(nn.Module):
    def __init__(self, seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.base_ca = LinearTemporalCrossAttention(seq_len, latent_dim,
                                                    text_latent_dim, num_head,
                                                    dropout, time_embed_dim)
        # We keep a separate gating param if we want
        self.gate = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, x, xf, emb):
        ca_out = self.base_ca(x, xf, emb)
        alpha = torch.sigmoid(self.gate).view(1, 1, -1)
        return x + alpha * (ca_out - x)


# ------------------------------------------------------------------------
# 9) PerformerSelfAttention (unchanged)
# ------------------------------------------------------------------------
class FastAttention(nn.Module):
    def __init__(self, dim, num_features=256, ortho=True, head_dim=None, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_features = num_features
        self.ortho = ortho
        self.head_dim = head_dim if head_dim is not None else dim
        self.projection_matrix = None
        self.eps = eps
        print(f"FastAttention initialized with head_dim={self.head_dim}, num_features={num_features}")
        
        # Single normalization layer for each head dimension
        self.norm = nn.LayerNorm(self.head_dim)
        
    def _create_projection(self, device):
        if self.ortho:
            projection_matrix = torch.randn(self.head_dim, self.num_features, device=device)
            q, _ = torch.linalg.qr(projection_matrix, mode='reduced')
            projection_matrix = q
        else:
            projection_matrix = torch.randn(self.head_dim, self.num_features, device=device)
        projection_matrix = F.normalize(projection_matrix, dim=0) * (self.head_dim ** -0.25)
        return projection_matrix
    
    def forward(self, q, k, v, mask=None):
        B, H, T, D = q.shape
        
        # Create or move projection matrix to correct device
        if self.projection_matrix is None:
            self.projection_matrix = self._create_projection(q.device)
        elif self.projection_matrix.device != q.device:
            self.projection_matrix = self.projection_matrix.to(q.device)

        # Normalize each head separately
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        # Reshape to [B*H, T, D] for normalization
        q = q.transpose(1, 2).reshape(B*H*T, D)
        k = k.transpose(1, 2).reshape(B*H*T, D)
        v = v.transpose(1, 2).reshape(B*H*T, D)
        
        # Apply normalization
        q = self.norm(q).reshape(B, T, H, D).permute(0, 2, 1, 3)
        k = self.norm(k).reshape(B, T, H, D).permute(0, 2, 1, 3)
        v = self.norm(v).reshape(B, T, H, D).permute(0, 2, 1, 3)

        # Additional stabilization
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        # Project with stable scaling
        q_proj = torch.exp(torch.clamp(
            torch.einsum('bhtn,nm->bhtm', q, self.projection_matrix),
            min=-15, max=15
        )) * 0.1
        
        k_proj = torch.exp(torch.clamp(
            torch.einsum('bhtn,nm->bhtm', k, self.projection_matrix),
            min=-15, max=15
        )) * 0.1
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.to(q.dtype)  # Convert mask to same dtype
            if mask.dim() == 3:  # [B, T, 1]
                mask = mask.squeeze(-1).unsqueeze(1)  # [B, 1, T]
            k_proj = k_proj * mask.unsqueeze(-1)
        
        # Compute attention with stable intermediate values
        kv = torch.einsum('bhtm,bhtn->bhmn', k_proj, v) * 0.1
        qkv = torch.einsum('bhtm,bhmn->bhtn', q_proj, kv) * 0.1
        
        # Compute denominator with stability
        denominator = torch.einsum('bhtm,bhtm->bht', q_proj, k_proj).unsqueeze(-1)
        denominator = denominator.clamp(min=self.eps)
        
        # Final output with normalization
        output = qkv / denominator
        
        # Reshape and normalize final output
        output = output.transpose(1, 2).reshape(B*H*T, D)
        output = self.norm(output)
        output = output.reshape(B, T, H, D).permute(0, 2, 1, 3)
        
        return output

class PerformerSelfAttention(nn.Module):
    def __init__(self, seq_len, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.head_dim = latent_dim // num_head
        assert latent_dim % num_head == 0, "latent_dim must be divisible by num_head"
        
        # Pre-normalization
        self.pre_norm = nn.LayerNorm(latent_dim)
        self.post_norm = nn.LayerNorm(latent_dim)
        
        # Main attention components
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        self.fast_attention = FastAttention(
            dim=latent_dim, 
            head_dim=self.head_dim, 
            num_features=256
        )
        
        # Output projection
        self.proj_out = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Stylization block
        self.style_block = StylizationBlock(latent_dim, time_embed_dim, dropout)
        
        # Initialize weights with smaller values
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_normal_(p, gain=0.1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H = self.num_head
        
        # Pre-normalization
        h = self.pre_norm(x)
        
        # Project and reshape
        q = self.query(h)
        k = self.key(h)
        v = self.value(h)
        
        # Clip gradients
        for tensor in [q, k, v]:
            if tensor.requires_grad:
                tensor.register_hook(lambda grad: torch.clamp(grad, -1, 1))
        
        # Reshape with scaling - use reshape instead of view
        q = q.reshape(B, T, H, self.head_dim).permute(0, 2, 1, 3) * 0.1
        k = k.reshape(B, T, H, self.head_dim).permute(0, 2, 1, 3) * 0.1
        v = v.reshape(B, T, H, self.head_dim).permute(0, 2, 1, 3) * 0.1
        
        # Apply attention
        attn_out = self.fast_attention(q, k, v, mask=src_mask)
        attn_out = self.attn_dropout(attn_out)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, D)
        
        # Output projection
        attn_out = self.proj_out(attn_out)
        attn_out = self.proj_dropout(attn_out)
        
        # Post-normalization
        attn_out = self.post_norm(attn_out)
        
        # Scale output
        attn_out = F.normalize(attn_out, dim=-1) * (D ** 0.5)
        
        # Apply stylization
        style_out = self.style_block(attn_out, emb)
        
        # Final skip connection with scaling
        y = x + 0.1 * style_out
        return y


# ------------------------------------------------------------------------
# 10) DualSelfAttentionBlock (unchanged)
# ------------------------------------------------------------------------
class DualSelfAttentionBlock(nn.Module):
    def __init__(self, seq_len, latent_dim, num_head, dropout, time_embed_dim,
                 local_window_size=16):
        super().__init__()
        # Pre and post norms for the block
        self.pre_norm = nn.LayerNorm(latent_dim)
        self.post_norm = nn.LayerNorm(latent_dim)
        
        # Attention modules
        self.local_attn = PerformerSelfAttention(
            seq_len, latent_dim, num_head, dropout, time_embed_dim
        )
        self.global_attn = PerformerSelfAttention(
            seq_len, latent_dim, num_head, dropout, time_embed_dim
        )
        
        # Additional skip connection
        self.skip_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Dropout(dropout),
            nn.GELU()
        )
        
    def forward(self, x, emb, src_mask):
        # Pre-norm
        h = self.pre_norm(x)
        
        # Local attention path
        local_out = self.local_attn(h, emb, src_mask)
        
        # Global attention path
        global_out = self.global_attn(local_out, emb, src_mask)
        
        # Skip connection with projection
        skip = self.skip_proj(x)
        
        # Combine paths with scaled residual
        out = skip + 0.1 * global_out
        
        # Post-norm
        out = self.post_norm(out)
        return out


# ------------------------------------------------------------------------
# 11) MemoryEfficientCrossAttentionBlock for stable diffusion (unchanged)
#    We'll keep as is from prior code if we want. We'll skip redefinition for brevity.
#    But we can add it if we want more cross-attn. 
#    We'll add "layer-wise cross-attn" next.
# ------------------------------------------------------------------------
class MemoryEfficientCrossAttentionBlock(nn.Module):
    """
    Simple chunk-based cross-attention to reduce memory usage.
    We'll integrate it into the final layer or multiple layers.
    """
    def __init__(self, latent_dim, text_latent_dim, num_heads, chunk_size=256, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.chunk_size = chunk_size
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(latent_dim, latent_dim)
        self.key   = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.out   = nn.Linear(latent_dim, latent_dim)

        self.dropout = nn.Dropout(dropout)
        # small FFN after cross-attn
        self.ffn = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim*4),
            nn.GELU(),
            nn.Linear(latent_dim*4, latent_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, xf, mask=None):
        B, Tx, D = x.shape
        Txf = xf.shape[1]

        q = self.query(x).view(B, Tx, self.num_heads, self.head_dim)
        k = self.key(xf).view(B, Txf, self.num_heads, self.head_dim)
        v = self.value(xf).view(B, Txf, self.num_heads, self.head_dim)

        q = q.permute(0,2,1,3)
        k = k.permute(0,2,1,3)
        v = v.permute(0,2,1,3)

        out = torch.zeros_like(q)
        for start in range(0, Tx, self.chunk_size):
            end = min(start + self.chunk_size, Tx)
            q_chunk = q[:,:,start:end,:]
            attn_scores = torch.einsum('bhqd,bhkd->bhqk', q_chunk*self.scale, k)
            if mask is not None:
                attn_scores += mask[:,None,start:end,:]
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            out_chunk = torch.einsum('bhqk,bhkd->bhqd', attn_probs, v)
            out[:,:,start:end,:] = out_chunk

        out = out.permute(0,2,1,3).reshape(B, Tx, D)
        out = self.out(out)

        # small residual FFN
        out = out + self.ffn(out)
        return x + out


# ------------------------------------------------------------------------
# 12) MoEExtendedDecoderLayer (modified to add "layer-wise memory eff cross-attn")
# ------------------------------------------------------------------------
class MoEExtendedDecoderLayer(nn.Module):
    """
    We combine:
      - DualSelfAttentionBlock
      - GatedCrossAttention
      - MoEMultiBranchFFN
      - We ALSO add a memory-efficient cross-attn block in the middle or end,
        so that text info is injected more times => "layer-wise cross-attn"
    """
    def __init__(self,
                 seq_len, latent_dim, text_latent_dim,
                 num_head, dropout, time_embed_dim, ffn_dim,
                 moe_num_experts=8,
                 chunk_size=256):
        super().__init__()
        self.dual_self_attn = DualSelfAttentionBlock(
            seq_len, latent_dim, num_head, dropout, time_embed_dim
        )
        self.cross_attn = GatedCrossAttention(
            seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim
        )
        self.ffn = MoEMultiBranchFFN(
            latent_dim=latent_dim,
            ffn_dim=ffn_dim,
            num_experts=moe_num_experts,
            num_branches=2,
            dropout=dropout,
            time_embed_dim=time_embed_dim
        )
        # 2nd cross-attn for layer-wise injection
        self.sd_cross_attn = MemoryEfficientCrossAttentionBlock(
            latent_dim=latent_dim,
            text_latent_dim=text_latent_dim,
            num_heads=num_head,
            chunk_size=chunk_size,
            dropout=dropout
        )

    def forward(self, x, xf, emb, src_mask):
        # local+global self-attn
        x = self.dual_self_attn(x, emb, src_mask)
        # first cross-attn
        x = self.cross_attn(x, xf, emb)
        # MoE FFN
        x = self.ffn(x, emb)
        # second memory-efficient cross-attn
        x = self.sd_cross_attn(x, xf, mask=None)
        return x


# ------------------------------------------------------------------------
# 13) EnhancedTextEncoder (unchanged)
# ------------------------------------------------------------------------
class EnhancedTextEncoder(nn.Module):
    def __init__(self, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.model_name = "microsoft/deberta-v3-large"
        self.bert = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.proj = nn.Sequential(
            nn.LayerNorm(self.bert.config.hidden_size),
            nn.Linear(self.bert.config.hidden_size, output_dim),
            nn.Dropout(dropout),
            nn.GELU()
        )
        self.num_prompt_tokens = 8
        self.prompt_tokens = nn.Parameter(
            torch.randn(1, self.num_prompt_tokens, self.bert.config.hidden_size)
        )

    def forward(self, text: List[str], device: torch.device):
        inputs = self.tokenizer(
            text, padding=True, truncation=True, max_length=77,
            return_tensors="pt"
        ).to(device)

        batch_size = len(text)
        prompts = self.prompt_tokens.repeat(batch_size, 1, 1)

        outputs = self.bert(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            return_dict=True
        )
        hidden_states = outputs.last_hidden_state
        hidden_states = torch.cat([prompts, hidden_states], dim=1)
        projected = self.proj(hidden_states)
        
        pooled = torch.mean(projected, dim=1)
        return pooled, projected


# ------------------------------------------------------------------------
# 14) Multi-Scale (U-Net–like) MotionTransformer
# ------------------------------------------------------------------------
class MotionTransformer(nn.Module):
    """
    We wrap your original approach with:
      - Downsample stage (coarse motion)
      - Intermediate MoE blocks
      - Upsample stage
    We keep MoEExtendedDecoderLayer but call it at multiple scales.
    """
    def __init__(self,
                 input_feats: int,
                 num_frames: int = 60,
                 latent_dim: int = 512,
                 ff_size: int = 1024,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 text_latent_dim: int = 256,
                 moe_num_experts: int = 4,
                 model_size: str = "small",
                 chunk_size: int = 256,
                 **kwargs):
        super().__init__()
        if model_size == "big":
            latent_dim *= 2
            ff_size *= 2
            text_latent_dim *= 2
            print("[INFO] Using 'big' model configuration: doubled latent_dim/ff_size")

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.moe_num_experts = moe_num_experts
        self.chunk_size = chunk_size

        self.time_embed_dim = latent_dim*4
        self.sequence_embedding = nn.Parameter(torch.randn(num_frames, latent_dim))

        # Time + text embed
        self.learnable_time_embed = LearnableTimeEmbedding(latent_dim)
        self.gated_fusion = GatedFusion(embed_dim=latent_dim)
        self.text_encoder = EnhancedTextEncoder(output_dim=text_latent_dim, dropout=dropout)
        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.time_proj = nn.Linear(self.time_embed_dim, latent_dim)

        # Input embed
        self.joint_embed = nn.Linear(input_feats, latent_dim)

        # U-Net style down/upsample
        # We'll do 2-level: a coarse scale at T/2, then full scale.
        # For demonstration, we do a simple 1D conv approach
        self.downsample = nn.Conv1d(latent_dim, latent_dim, kernel_size=2, stride=2)
        self.upsample = nn.ConvTranspose1d(latent_dim, latent_dim, kernel_size=2, stride=2)

        # We keep multiple MoEExtendedDecoderLayers, applying at each scale
        survival_probs = torch.linspace(1.0, 0.8, steps=num_layers)
        self.decoder_blocks_low = nn.ModuleList()
        self.decoder_blocks_high = nn.ModuleList()
        for i in range(num_layers):
            block_low = MoEExtendedDecoderLayer(
                seq_len=num_frames//2,
                latent_dim=latent_dim,
                text_latent_dim=text_latent_dim,
                num_head=num_heads,
                dropout=dropout,
                time_embed_dim=self.time_embed_dim,
                ffn_dim=ff_size,
                moe_num_experts=moe_num_experts,
                chunk_size=chunk_size
            )
            block_high = MoEExtendedDecoderLayer(
                seq_len=num_frames,
                latent_dim=latent_dim,
                text_latent_dim=text_latent_dim,
                num_head=num_heads,
                dropout=dropout,
                time_embed_dim=self.time_embed_dim,
                ffn_dim=ff_size,
                moe_num_experts=moe_num_experts,
                chunk_size=chunk_size
            )
            self.decoder_blocks_low.append(StochasticDepth(block_low, survival_prob=survival_probs[i].item()))
            self.decoder_blocks_high.append(StochasticDepth(block_high, survival_prob=survival_probs[i].item()))

        # final out
        self.out = zero_module(nn.Linear(latent_dim, input_feats))


    def encode_text(self, text: List[str], device: torch.device):
        return self.text_encoder(text, device)

    def generate_src_mask(self, T: int, length: torch.Tensor) -> torch.Tensor:
        B = len(length)
        mask = torch.ones(B, T, device=length.device)
        for i in range(B):
            mask[i, length[i]:] = 0
        return mask

    def forward(self,
                x: torch.Tensor,
                timesteps: torch.Tensor,
                length: torch.Tensor,
                text: Optional[List[str]]=None) -> torch.Tensor:
        """
        Multi-scale forward pass:
         1) embed input
         2) downsample (coarse scale)
         3) pass MoE blocks at coarse scale
         4) upsample back to full scale
         5) pass MoE blocks at full scale
        """
        device = x.device
        B, T, D = x.shape

        # 1) text encode
        xf_proj, xf_out = self.encode_text(text, device)
        if xf_proj.shape[-1] != self.latent_dim:
            text_proj = nn.Linear(xf_proj.shape[-1], self.latent_dim).to(device)
            xf_proj = text_proj(xf_proj)

        # 2) fuse time + text
        time_emb = self.learnable_time_embed(timesteps)
        time_emb_expanded = self.time_embed(time_emb)
        time_emb_proj = self.time_proj(time_emb_expanded)
        fused_emb = self.gated_fusion(time_emb_proj, xf_proj)

        # 3) embed motion
        h = self.joint_embed(x)
        # add sequence embedding
        h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :]

        # src mask
        src_mask = self.generate_src_mask(T, length).unsqueeze(-1)

        # => shape [B, latent_dim, T]
        h_for_down = h.permute(0,2,1)
        # downsample
        h_low = self.downsample(h_for_down)  # [B, latent_dim, T/2]
        # revert shape to [B,T/2, latent_dim]
        Tlow = h_low.shape[-1]
        h_low = h_low.permute(0,2,1)

        # pass through coarse MoE blocks
        # We'll produce a smaller src_mask for T/2
        length_low = (length/2).long()
        src_mask_low = self.generate_src_mask(Tlow, length_low).unsqueeze(-1)
        for block in self.decoder_blocks_low:
            h_low = block(h_low, xf_out, fused_emb, src_mask_low)

        # upsample
        h_up = h_low.permute(0,2,1)  # [B,latent_dim,T/2]
        h_up = self.upsample(h_up)   # => [B,latent_dim,T] if stride=2
        h_up = h_up.permute(0,2,1)   # [B,T,latent_dim]

        # combine skip connection from original h
        # (like a U-Net skip)
        h_combined = h_up + h  # skip

        # pass full-scale blocks
        for block in self.decoder_blocks_high:
            h_combined = block(h_combined, xf_out, fused_emb, src_mask)

        # final out
        output = self.out(h_combined)
        return output


# ------------------------------------------------------------------------
# Usage Example
# ------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotionTransformer(
        input_feats=66,
        num_frames=60,
        latent_dim=128,
        ff_size=256,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        text_latent_dim=128,
        moe_num_experts=4,
        model_size="big",   # e.g., double dims
        chunk_size=256
    ).to(device)

    B = 2
    T = 60
    x = torch.randn(B, T, 66, device=device)
    timesteps = torch.randint(0, 1000, (B,), device=device)
    length = torch.tensor([T, T], dtype=torch.long, device=device)
    text_prompts = ["person dancing quickly", "person walking slowly"]

    output = model(x, timesteps, length, text_prompts)
    print("Output shape:", output.shape)  # [B, T, 66]
