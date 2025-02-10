import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from torch.nn.utils import weight_norm
from torch.nn import GroupNorm

from switch_moe import SwitchMoELayer
from gate import GatedFusion
from fast_attention import DualSelfAttentionBlock, GatedCrossAttention, MemoryEfficientCrossAttentionBlock
from time import LearnableTimeEmbedding, StochasticDepth
from multi_branch import MoEMultiBranchFFN
from text_encoder import TextEncoder
from utils import zero_module, NormalizationBlock

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




class EnhancedStylizationBlock(nn.Module):
    def __init__(self, latent_dim, time_embed_dim, dropout, num_groups=8):
        super().__init__()
        self.group_norm = GroupNorm(num_groups, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)
        
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            weight_norm(nn.Linear(time_embed_dim, 2 * latent_dim)),
            nn.Dropout(p=dropout)
        )
        
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            weight_norm(nn.Linear(latent_dim, latent_dim)),
            GroupNorm(num_groups, latent_dim)
        )

    def forward(self, h, emb):
        # Apply dual normalization
        h = self.group_norm(h)
        h = self.layer_norm(h)
        
        emb_out = self.emb_layers(emb).unsqueeze(1)
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        
        h = h * (1 + scale) + shift
        h = self.out_layers(h)
        return h

class EnhancedMoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=8, num_groups=8):
        super().__init__()
        self.num_experts = num_experts
        self.input_norm = NormalizationBlock(input_dim, num_groups)
        
        self.gate = weight_norm(nn.Linear(input_dim, num_experts))
        self.experts = nn.ModuleList([
            nn.Sequential(
                GroupNorm(num_groups, input_dim),
                weight_norm(nn.Linear(input_dim, hidden_dim)),
                nn.GELU(),
                nn.Dropout(0.1),
                weight_norm(nn.Linear(hidden_dim, input_dim)),
                GroupNorm(num_groups, input_dim)
            ) for _ in range(num_experts)
        ])
        
        # Initialize gates
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x):
        B, T, D = x.shape
        
        # Apply input normalization
        x = self.input_norm(x)
        
        # Compute gates
        gates = F.softmax(self.gate(x), dim=-1)
        
        # Expert computation with residual
        expert_outputs = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)
            expert_outputs += gates[..., i:i+1] * expert_output
            
        return expert_outputs

class EnhancedTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.1, num_groups=8):
        super().__init__()
        self.norm1 = NormalizationBlock(dim, num_groups)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm2 = NormalizationBlock(dim, num_groups)
        self.mlp = nn.Sequential(
            weight_norm(nn.Linear(dim, dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            weight_norm(nn.Linear(dim * mlp_ratio, dim)),
            GroupNorm(num_groups, dim)
        )
        
    def forward(self, x):
        # First normalization and attention
        normed = self.norm1(x)
        attn_output, _ = self.attn(normed, normed, normed)
        x = x + attn_output
        
        # Second normalization and MLP
        normed = self.norm2(x)
        mlp_output = self.mlp(normed)
        x = x + mlp_output
        
        return x

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
        self.text_encoder = TextEncoder(output_dim=text_latent_dim, dropout=dropout)
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

  
    def reset_all_moe_counters(self, model):
        for module in model.modules():
            if isinstance(module, SwitchMoELayer):
                module._reset_moe_counters()
                
    def get_total_moe_loss(self, model, moe_coef=0.01):
        total_moe_loss = 0.0
        for module in model.modules():
            if isinstance(module, SwitchMoELayer):
                total_moe_loss += module.get_load_balancing_loss()
        return moe_coef * total_moe_loss
    
    def get_moe_loss(self, model):
        moe_loss = 0
        for module in model.modules():
                # You could also do something like hasattr(module, "get_load_balancing_loss")
                # if you prefer a more general check.
                if isinstance(module, SwitchMoELayer):
                    moe_loss += module.get_load_balancing_loss()
        return moe_loss
    
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
            text_proj = nn.Linear(xf_proj.shape[-1], self.latent_dim).to(device)  # Ensure it's on the correct device
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
