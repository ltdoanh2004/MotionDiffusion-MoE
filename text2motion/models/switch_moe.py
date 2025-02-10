import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwitchMoELayer(nn.Module):
    """
    Enhanced Switch Transformer MoE layer with:
      - Top-2 gating
      - Simple load balancing
    """
    def __init__(self, input_dim, hidden_dim, num_experts=8):
        super().__init__()
        self.num_experts = num_experts
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

        # Track usage (number of tokens that picked expert e as top-1)
        self.register_buffer("expert_usage", torch.zeros(num_experts))
        # Track importance (sum of gate probabilities that go to e)
        self.register_buffer("expert_importance", torch.zeros(num_experts))

    def _reset_moe_counters(self):
        """
        Reset counters at the start of each forward or each training iteration.
        This is optional but often helpful so that the counters don't keep growing forever.
        """
        self.expert_usage.zero_()
        self.expert_importance.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        Returns: (B, T, D) output
        """
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)  # [B*T, D]
        
        # 1) Gating
        logits = self.gate(x_flat)  # [B*T, num_experts]
        probs = F.softmax(logits, dim=1)  # [B*T, num_experts]

        # ---- top-2 gating ----
        top2_vals, top2_idx = torch.topk(probs, k=2, dim=1)  # each is [B*T, 2]

        # 2) Update usage & importance
        #
        #    - "usage" is how many tokens pick an expert as top-1
        #    - "importance" is the sum of the probabilities (here, sum of top-2 or full? 
        #      But commonly we sum whichever probabilities lead to actual routing)
        #
        #    Because these are buffers, we must be sure to do the updates in a no-grad block
        #    or use in-place modifications that won't break autograd. Typically, usage is
        #    purely counters. "importance" can also be kept outside the gradient path.
        #

        # top1 assignments
        top1_idx = top2_idx[:, 0]  # (B*T,)
        with torch.no_grad():
            # Count how many times each expert was top-1
            for e_idx in range(self.num_experts):
                usage_count = (top1_idx == e_idx).sum()
                self.expert_usage[e_idx] += usage_count.item()

            # For importance, we add whichever probabilities contributed
            # in top-2 gating. (Alternatively, you might add all probs from 'probs'.)
            for e_idx in range(self.num_experts):
                # gather the probability for e_idx from top2
                # mask where top2_idx == e_idx
                mask_any = (top2_idx == e_idx).any(dim=1)  # [B*T]
                if mask_any.any():
                    # For each token that routes e_idx as top-1 or top-2,
                    # add the corresponding top2_vals
                    # figure out if e_idx is in the 0th or 1st column
                    rowcols = torch.nonzero(top2_idx[mask_any] == e_idx)
                    # rowcols is shape [num_routed, 2], second dim is which column
                    # rowcols[:, 1] is in {0,1}, giving us the correct top2_val
                    vals = top2_vals[mask_any, rowcols[:,1]]
                    self.expert_importance[e_idx] += vals.sum().item()

        # 3) Merge outputs from top-2 experts
        output = torch.zeros_like(x_flat)

        for expert_idx in range(self.num_experts):
            mask = (top2_idx == expert_idx)  # [B*T, 2]
            if not mask.any():
                continue
            mask_any = mask.any(dim=1)
            if mask_any.any():
                expert_input = x_flat[mask_any]
                expert_output = self.experts[expert_idx](expert_input)

                # find which column belongs to expert_idx
                col = (top2_idx[mask_any] == expert_idx).nonzero()[:,1]
                val = top2_vals[mask_any, col].unsqueeze(-1)  # scale factor
                output[mask_any] += val * expert_output
        
        return output.view(B, T, D)

    def get_load_balancing_loss(self, epsilon: float = 1e-8) -> torch.Tensor:
        """
        Compute a scalar load-balancing loss that penalizes large mismatch
        between 'expert_usage' (actual top-1 counts) and 'expert_importance' (sum of gating probs).
        
        We first convert each to a fraction of total usage or total importance.
        Then we measure how well they align by computing the dot product:
            sum_{e} fraction_usage[e] * fraction_importance[e]
        That dot product is in [0,1], and is 1 if usage fraction == importance fraction.
        
        A common approach is to *maximize* that dot product, 
        or equivalently minimize something like: (1 - dot_product).
        We multiply by num_experts so that, in the ideal case, the loss is 0
        when dot_product=1, and in the worst case is ~ num_experts if they are disjoint.
        
        You can adapt or scale this based on your preference.
        """
        # usage fractions
        total_usage = self.expert_usage.sum().clamp_min(epsilon)
        fraction_usage = self.expert_usage / total_usage

        # importance fractions
        total_importance = self.expert_importance.sum().clamp_min(epsilon)
        fraction_importance = self.expert_importance / total_importance

        # dot product in [0..1]
        alignment = (fraction_usage * fraction_importance).sum()

        # One common form:  load_balance_loss = num_experts * (1.0 - alignment)
        # If alignment=1, loss=0 => perfectly balanced
        # If alignment=0, loss=num_experts => severely unbalanced
        load_balance_loss = self.num_experts * (1.0 - alignment)
        return load_balance_loss
    

