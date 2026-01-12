# models/hrm.py
import torch
import torch.nn as nn
from .components import CrossAttention
from .text_encoder import TextEncoder
from .video_processing import VideoTokenizer, VideoDetokenizer
from ..utils.initialization import rms_norm, trunc_normal_init_

class HRMLayer(nn.Module):
    """Hierarchical Reasoning Layer with text conditioning"""
    def __init__(self, hidden_size, num_heads, expansion=2.0):
        super().__init__()
        from .components import TransformerBlock
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, expansion, causal=True)
            for _ in range(2)
        ])

        # Cross attention for text conditioning
        self.cross_attn = CrossAttention(hidden_size, num_heads)

    def forward(self, hidden_states, input_injection, text_features=None, cos=None, sin=None):
        # Input injection
        hidden_states = hidden_states + input_injection

        # Cross attention with text
        if text_features is not None:
            hidden_states = rms_norm(hidden_states + self.cross_attn(hidden_states, text_features))

        # Self attention blocks
        from .components import TransformerBlock
        for block in self.blocks:
            hidden_states = block(hidden_states, cos, sin)

        return hidden_states

class TextToVideoHRM(nn.Module):
    """Text-to-Video HRM optimized for T4 GPU"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Text processing
        self.text_encoder = TextEncoder(
            vocab_size=config['vocab_size'],
            hidden_size=config['hidden_size'],
            max_text_len=config['max_text_len']
        )

        # Video processing
        self.video_tokenizer = VideoTokenizer(
            in_channels=3,
            hidden_dim=config['hidden_size'],
            patch_size=(2, 4, 4)
        )
        self.video_detokenizer = VideoDetokenizer(
            hidden_dim=config['hidden_size'],
            out_channels=3,
            output_shape=(config['frames'], config['height'], config['width'])
        )

        # Compute actual seq_len from tokenizer
        dummy_input = torch.zeros(1, 3, config['frames'], config['height'], config['width'])
        with torch.no_grad():
            sample_tokens = self.video_tokenizer(dummy_input)
        actual_seq_len = sample_tokens.shape[1]
        self.config['video_seq_len'] = actual_seq_len

        # Position embeddings for video tokens
        from .components import RotaryEmbedding
        self.rope = RotaryEmbedding(
            config['hidden_size'] // config['num_heads'],
            max_seq_len=actual_seq_len
        )

        # Hierarchical reasoning layers
        self.H_layer = HRMLayer(
            config['hidden_size'],
            config['num_heads'],
            config['expansion']
        )
        self.L_layer = HRMLayer(
            config['hidden_size'],
            config['num_heads'],
            config['expansion']
        )

        # Initial states
        self.register_buffer('H_init', torch.randn(1, 1, config['hidden_size']) * 0.02)
        self.register_buffer('L_init', torch.randn(1, 1, config['hidden_size']) * 0.02)

        # Output processing
        self.output_norm = nn.LayerNorm(config['hidden_size'])
        self.halt_predictor = nn.Linear(config['hidden_size'], 1)

    def forward(self, text_tokens, target_video=None, max_steps=3):
        batch_size = text_tokens.shape[0]

        # Encode text
        text_features = self.text_encoder(text_tokens)

        # Initialize video tokens (for generation, start with noise or learned embeddings)
        if target_video is not None:
            # Training mode: use target video
            video_tokens = self.video_tokenizer(target_video)
        else:
            # Generation mode: start with learned embeddings or noise
            seq_len = self.config['video_seq_len']
            video_tokens = torch.randn(batch_size, seq_len, self.config['hidden_size'],
                                     device=text_tokens.device) * 0.02

        seq_len = video_tokens.shape[1]

        # Initialize carry states
        z_H = self.H_init.expand(batch_size, seq_len, -1)
        z_L = self.L_init.expand(batch_size, seq_len, -1)

        # Get rotary embeddings
        cos, sin = self.rope(seq_len, video_tokens.device)

        # Hierarchical reasoning with text conditioning
        halt_scores = None
        step = 0
        for step in range(max_steps):
            # L-level reasoning (fine-grained) with text conditioning
            z_L = self.L_layer(z_L, z_H + video_tokens, text_features, cos, sin)

            # H-level reasoning (coarse-grained) with text conditioning
            z_H = self.H_layer(z_H, z_L, text_features, cos, sin)

            # Early stopping check
            if step > 0:
                halt_scores = torch.sigmoid(self.halt_predictor(z_H.mean(dim=1)))
                if (halt_scores > 0.5).all():
                    break

        # Generate final video
        video_features = self.output_norm(z_H)
        output_video = self.video_detokenizer(video_features)

        return {
            'generated_video': output_video,
            'reasoning_steps': step + 1,
            'halt_scores': halt_scores,
            'text_features': text_features,
            'video_features': video_features
        }