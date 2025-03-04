import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from transformers import AutoModel, AutoTokenizer
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
        hidden_states = torch.cat([prompts.to(device), hidden_states], dim=1)
        projected = self.proj(hidden_states)
        
        pooled = torch.mean(projected, dim=1)
        return pooled, projected