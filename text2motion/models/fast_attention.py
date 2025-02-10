import torch
import torch.nn as nn
from stylization import StylizationBlock
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
            mask = mask.to(q.device)
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
