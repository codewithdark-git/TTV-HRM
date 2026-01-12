# utils/initialization.py
import torch
import math

def trunc_normal_init_(tensor, std=0.02):
    """Initialize tensor with truncated normal distribution"""
    with torch.no_grad():
        tensor.normal_(0, std)
        tensor.clamp_(-2*std, 2*std)
    return tensor

def rms_norm(x, variance_epsilon=1e-5):
    """RMS Layer Normalization"""
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + variance_epsilon)