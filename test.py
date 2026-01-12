# test.py
import torch
import imageio
from .models.hrm import TextToVideoHRM
from .data.tokenizer import ProperTokenizer
from .utils.device import set_device

def test_text_to_video_generation(model, device):
    """Test text-to-video generation"""
    print("Testing text-to-video generation...")
    model.eval()

    # Test prompts
    test_texts = [
        "a red ball moving left to right",
        "a blue square rotating clockwise"
    ]

    tokenizer = ProperTokenizer()

    with torch.no_grad():
        for i, text in enumerate(test_texts):
            print(f"\nGenerating video for: '{text}'")

            text_tokens = tokenizer.encode(text).unsqueeze(0).to(device)

            # Generate video (no target video provided)
            outputs = model(text_tokens, target_video=None)

            generated_video = outputs['generated_video']
            print(f"Generated video shape: {generated_video.shape}")
            print(f"Video value range: [{generated_video.min():.3f}, {generated_video.max():.3f}]")
            print(f"Reasoning steps used: {outputs['reasoning_steps']}")

            # Save the generated video as GIF for human readability
            # Denormalize from [-1,1] to [0,255] uint8
            generated_video = (generated_video + 1) / 2 * 255
            generated_video = generated_video.clamp(0, 255).byte()

            # Permute to (T, H, W, C) for video writing
            generated_video = generated_video.squeeze(0).permute(1, 2, 3, 0)  # From (1, C, T, H, W) -> (T, H, W, C)

            # Prepare frames as numpy arrays
            frames = [generated_video[t].cpu().numpy() for t in range(generated_video.shape[0])]

            # FPS for the video (arbitrary, e.g., 8 FPS for 8 frames), but for GIF it's duration
            duration = 8  # total duration in seconds, so FPS = len(frames)/duration

            # Save path as GIF
            save_path = f'generated_video_{i+1}_{text.replace(" ", "_")[:20]}.gif'
            print(f"Saving generated video to: {save_path}")

            # Write video using imageio as GIF
            try:
                imageio.mimsave(save_path, frames, format='GIF', duration=duration / len(frames))
                print(f"GIF saved successfully: {save_path}")
            except Exception as e:
                print(f"Failed to save GIF: {e}")