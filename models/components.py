# models/components.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple
from ..utils.initialization import rms_norm

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embeddings correctly"""
    # q, k: (B, T, nh, hs)
    # cos, sin: (T, hs//2)
    B, T, nh, hs = q.shape

    # Expand cos, sin to (B, T, nh, hs//2)
    cos = cos.unsqueeze(0).unsqueeze(2).expand(B, -1, nh, -1)
    sin = sin.unsqueeze(0).unsqueeze(2).expand(B, -1, nh, -1)

    # Split into pairs
    q0 = q[..., :hs//2]
    q1 = q[..., hs//2:]
    k0 = k[..., :hs//2]
    k1 = k[..., hs//2:]

    # Apply rotation
    q0_rot = q0 * cos - q1 * sin
    q1_rot = q0 * sin + q1 * cos
    q_embed = torch.cat([q0_rot, q1_rot], dim=-1)

    k0_rot = k0 * cos - k1 * sin
    k1_rot = k0 * sin + k1 * cos
    k_embed = torch.cat([k0_rot, k1_rot], dim=-1)

    return q_embed, k_embed

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with proper dimension handling"""
    def __init__(self, hidden_size, num_heads, causal=True):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.causal = causal

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, cos=None, sin=None):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape: (B, T, num_heads, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        # Apply rotary embeddings
        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Transpose for attention: (B, num_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores.masked_fill_(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape back: (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class SwiGLU(nn.Module):
    """SwiGLU Feed Forward Network"""
    def __init__(self, hidden_size, expansion=2.0):
        super().__init__()
        inner_dim = int(hidden_size * expansion)
        self.w1 = nn.Linear(hidden_size, inner_dim, bias=False)
        self.w2 = nn.Linear(hidden_size, inner_dim, bias=False)
        self.w3 = nn.Linear(inner_dim, hidden_size, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class TransformerBlock(nn.Module):
    """Transformer block with post-norm"""
    def __init__(self, hidden_size, num_heads, expansion=2.0, causal=True):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_size, num_heads, causal)
        self.ffn = SwiGLU(hidden_size, expansion)

    def forward(self, x, cos=None, sin=None):
        x = rms_norm(x + self.attn(x, cos, sin))
        x = rms_norm(x + self.ffn(x))
        return x

class CrossAttention(nn.Module):
    """Cross attention for text-to-video conditioning"""
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, video_features, text_features):
        B, T_v, C = video_features.shape
        _, T_t, _ = text_features.shape

        q = self.q_proj(video_features).view(B, T_v, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(text_features).view(B, T_t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(text_features).view(B, T_t, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T_v, C)
        return self.out_proj(out)