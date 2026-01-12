# utils/evaluation.py
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F

def compute_fid(real_videos, generated_videos, device, num_frames=8):
    """
    Compute FID between real and generated videos by pooling frames.
    """
    print("Computing FID score...")
    fid = FrechetInceptionDistance(feature=2048)
    fid.to(device)

    # Frame transformation - starts from PIL Image, outputs uint8 tensor for FID
    frame_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).round().byte()),
    ])

    # Update FID with real videos (all frames)
    with torch.no_grad():
        for video in real_videos:
            # video shape: (B, C, T, H, W)
            B, C, T, H, W = video.shape
            for b in range(B):
                for t in range(num_frames):
                    frame = video[b, :, t, :, :]  # Extracts (C, H, W)
                    # Denormalize from [-1,1] to [0,1]
                    frame_denorm = (frame + 1) / 2
                    # Ensure frame is (C, H, W), squeeze to remove potential singleton dims, then permute to (H, W, C) and convert to NumPy for ToPILImage
                    frame = frame_denorm.squeeze().permute(1, 2, 0).cpu().numpy() # Permute to (H, W, C) and convert to NumPy
                    frame = np.clip(frame, 0, 1)
                    frame_pil = transforms.ToPILImage()(frame) # Convert NumPy array to PIL Image
                    frame_processed = frame_transform(frame_pil).to(device) # Apply the rest of the transform
                    fid.update(frame_processed.unsqueeze(0), real=True) # Add batch dimension (1, C, H, W)

    # Update with generated
    with torch.no_grad():
        for video in generated_videos:
            # video shape: (B, C, T, H, W) from the eval loop
            B, C, T, H, W = video.shape
            for b in range(B):
                for t in range(T):
                    frame = video[b, :, t, :, :]  # Extracts (C, H, W)
                    # Denormalize from [-1,1] to [0,1]
                    frame_denorm = (frame + 1) / 2
                    # Ensure frame is (C, H, W), squeeze to remove potential singleton dims, then permute to (H, W, C) and convert to NumPy for ToPILImage
                    frame = frame_denorm.squeeze().permute(1, 2, 0).cpu().numpy() # Permute to (H, W, C) and convert to NumPy
                    frame = np.clip(frame, 0, 1)
                    frame_pil = transforms.ToPILImage()(frame) # Convert NumPy array to PIL Image
                    frame_processed = frame_transform(frame_pil).to(device) # Apply the rest of the transform
                    fid.update(frame_processed.unsqueeze(0), real=False) # Add batch dimension (1, C, H, W)

    fid_score = fid.compute()
    print(f"FID computation completed. Score: {fid_score.item():.4f}")
    return fid_score.item()