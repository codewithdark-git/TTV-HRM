# main.py
from train import train_text_to_video
from test import test_text_to_video_generation
from utils.device import set_device
import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    print("=== Text-to-Video HRM Training on T4 GPU ===")

    # Train model with extended params
    model = train_text_to_video(
        max_samples=50,
        batch_size=2,
        epochs=3,
        lr=1e-4,
        checkpoint_dir='./checkpoints',
        from_hf=False,  # Set to True and provide repo_id/hf_token to resume from HF
        repo_id="XCollab/TTV-HRM",  # Update with your HF repo
        hf_token=os.getenv('HF_TOKEN'),  # Your HF token
        save_every=1,
        eval_ratio=0.1,
        gradient_accumulation_steps=1,
        max_training_time_hours=4
    )

    # Test generation
    device = set_device()
    test_text_to_video_generation(model, device)

    print("\n=== Training Complete ===")
    print("Text-to-Video HRM successfully trained!")
    print("Generated videos saved as GIF files in the current directory for viewing!")