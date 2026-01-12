# utils/device.py
import torch

def set_device():
    """Set device to T4 GPU if available"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device