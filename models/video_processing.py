# models/video_processing.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoTokenizer(nn.Module):
    """3D CNN-based video tokenizer with fixed input format"""
    def __init__(self, in_channels=3, hidden_dim=256, patch_size=(2, 4, 4)):
        super().__init__()
        self.patch_size = patch_size

        # Fixed 3D convolution expecting (B, C, T, H, W)
        self.conv1 = nn.Conv3d(in_channels, hidden_dim//4, 3, 1, 1)
        self.conv2 = nn.Conv3d(hidden_dim//4, hidden_dim//2, patch_size, patch_size)
        self.conv3 = nn.Conv3d(hidden_dim//2, hidden_dim, 3, 1, 1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Ensure input is (B, C, T, H, W)
        if x.dim() == 5 and x.shape[1] != 3:
            # If shape is wrong, permute to correct format
            if x.shape[2] == 3:  # (B, T, C, H, W) -> (B, C, T, H, W)
                x = x.permute(0, 2, 1, 3, 4)

        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = self.conv3(x)

        # Flatten to sequence format: (B, seq_len, hidden_dim)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T*H*W, C)
        return self.norm(x)

class VideoDetokenizer(nn.Module):
    """3D CNN-based video detokenizer"""
    def __init__(self, hidden_dim=256, out_channels=3, output_shape=(8, 32, 32)):
        super().__init__()
        self.output_shape = output_shape
        self.hidden_dim = hidden_dim

        # Calculate tokens per dimension after encoding
        T, H, W = output_shape
        self.token_T = T // 2  # Reduced by patch_size[0]
        self.token_H = H // 4  # Reduced by patch_size[1]
        self.token_W = W // 4  # Reduced by patch_size[2]

        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.conv1 = nn.ConvTranspose3d(hidden_dim, hidden_dim//2, 3, 1, 1)
        self.conv2 = nn.ConvTranspose3d(hidden_dim//2, hidden_dim//4, (2, 4, 4), (2, 4, 4))
        self.conv3 = nn.ConvTranspose3d(hidden_dim//4, out_channels, 3, 1, 1)

    def forward(self, x):
        # x: (B, seq_len, hidden_dim)
        B, seq_len, C = x.shape

        x = self.proj(x)
        # Reshape to 3D volume
        x = x.reshape(B, self.token_T, self.token_H, self.token_W, C)
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)

        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = torch.tanh(self.conv3(x))

        # Interpolate to exact output shape to handle deconv sizing
        x = F.interpolate(x, size=(self.output_shape[0], self.output_shape[1], self.output_shape[2]),
                          mode='trilinear', align_corners=False)

        return x