# models/text_encoder.py
import torch
import torch.nn as nn
from .components import TransformerBlock
from ..utils.initialization import rms_norm

class TextEncoder(nn.Module):
    """Simple text encoder using pretrained embeddings"""
    def __init__(self, vocab_size=50257, hidden_size=256, max_text_len=77):
        super().__init__()
        self.max_text_len = max_text_len
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_text_len, hidden_size)

        # Text transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, 8, causal=False)
            for _ in range(3)
        ])

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, text_tokens):
        B, T = text_tokens.shape

        # Token and position embeddings
        text_emb = self.embedding(text_tokens)
        pos_ids = torch.arange(T, device=text_tokens.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(pos_ids)

        x = text_emb + pos_emb

        # Process through transformer layers
        for layer in self.layers:
            x = layer(x)

        return self.norm(x)