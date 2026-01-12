# train.py
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from .models.hrm import TextToVideoHRM
from .data.dataset import TextVideoDataset
from .utils.device import set_device
from .utils.checkpoint import save_and_push_checkpoint, load_checkpoint
from .utils.evaluation import compute_fid

def train_text_to_video(
    max_samples=50,
    batch_size=2,
    epochs=3,
    lr=1e-4,
    checkpoint_dir='./checkpoints',
    from_hf=False,
    repo_id="codewithdark/TTV-HRM",
    hf_token=None,
    save_every=1,
    eval_ratio=0.1,
    gradient_accumulation_steps=1,
    max_training_time_hours=4
):
    """
    Optimized training function for text-to-video model with checkpoints and FID evaluation.
    """
    start_time = time.time()
    max_training_time_seconds = max_training_time_hours * 3600

    device = set_device()

    # T4-optimized configuration
    config = {
        'hidden_size': 256,
        'num_heads': 8,
        'expansion': 2.0,
        'vocab_size': 50257,
        'max_text_len': 77,
        'video_seq_len': 256,  # Matches tokenizer output for (8,32,32)
        'frames': 8,
        'height': 32,
        'width': 32,
        'max_steps': 3,
    }

    print(f"Configuration: {config}")

    # Initialize model
    model = TextToVideoHRM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} (~{total_params/1e6:.1f}M)")

    # Dataset and dataloader
    full_dataset = TextVideoDataset(max_samples=max_samples, num_frames=config['frames'], height=config['height'], width=config['width'])

    # Split into train and eval
    eval_size = int(len(full_dataset) * eval_ratio)
    train_size = len(full_dataset) - eval_size
    train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [train_size, eval_size])
    print(f"Dataset split - Training samples: {len(train_dataset)}, Evaluation samples: {len(eval_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size // gradient_accumulation_steps, shuffle=True, num_workers=0)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Train dataloader batches per epoch: {len(train_dataloader)}, Eval dataloader batches: {len(eval_dataloader)}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Scheduler
    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Resume from checkpoint if available
    starting_epoch = 0
    best_fid = float('inf')

    if from_hf:
        model, optimizer, scheduler, starting_epoch, _ = load_checkpoint(
            model, optimizer, scheduler,
            path_or_repo=repo_id,
            from_hf=True
        )
    elif os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        if checkpoint_files:
            def get_epoch_key_local(filename):
                parts = filename.split('_')
                if len(parts) > 1 and parts[-1].split('.')[0].isdigit():
                    return int(parts[-1].split('.')[0])
                return 0

            latest_checkpoint = sorted(checkpoint_files, key=get_epoch_key_local)[-1]
            print(f"Resuming from local checkpoint in directory: {checkpoint_dir}")
            model, optimizer, scheduler, starting_epoch, _ = load_checkpoint(
                model, optimizer, scheduler,
                path_or_repo=checkpoint_dir,
                from_hf=False
            )

    # Training loop
    print(f"Starting training from epoch {starting_epoch + 1}")
    model.train()
    print("Starting text-to-video training...")

    from .data.tokenizer import ProperTokenizer
    tokenizer = ProperTokenizer()  # For saving

    for epoch in range(starting_epoch, epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Check time limit
        elapsed_time = time.time() - start_time
        if elapsed_time > max_training_time_seconds * 0.9:
            print(f"Approaching time limit. Saving checkpoint and ending training.")
            save_and_push_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=0.0,  # Placeholder
                tokenizer=tokenizer,
                repo_id=repo_id,
                hf_token=hf_token,
                path=checkpoint_dir
            )
            break

        # Train
        epoch_loss = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training")):
            text_tokens = batch['text_tokens'].to(device)
            target_video = batch['video'].to(device)

            # Forward pass
            outputs = model(text_tokens, target_video)
            predicted_video = outputs['generated_video']
            loss = F.mse_loss(predicted_video, target_video)

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()

            epoch_loss += loss.item() * gradient_accumulation_steps

            # Update every accumulation steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} training loss: {avg_loss:.4f}")

        # Evaluate with FID
        model.eval()
        real_videos = []
        generated_videos = []
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluation"):
                text_tokens = batch['text_tokens'].to(device)
                target_video = batch['video'].to(device)
                real_videos.append(target_video.cpu())

                outputs = model(text_tokens, target_video=None)  # Generate
                generated = outputs['generated_video']
                generated_videos.append(generated.cpu())

        fid_score = compute_fid(real_videos, generated_videos, device)
        print(f"Epoch {epoch+1} FID: {fid_score:.4f}")

        # Save best model based on FID
        if fid_score < best_fid:
            best_fid = fid_score
            print(f"New best FID: {best_fid:.4f}. Saving best checkpoint.")
            save_and_push_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch+1,
                loss=avg_loss,
                tokenizer=tokenizer,
                repo_id=repo_id,
                hf_token=hf_token,
                path=checkpoint_dir
            )

        # Save periodically
        if (epoch + 1) % save_every == 0:
            print(f"Saving periodic checkpoint for epoch {epoch+1}.")
            save_and_push_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch+1,
                loss=avg_loss,
                tokenizer=tokenizer,
                repo_id=repo_id,
                hf_token=hf_token,
                path=checkpoint_dir
            )

        # Time check
        elapsed_time = time.time() - start_time
        remaining_time = max_training_time_seconds - elapsed_time
        print(f"Elapsed: {elapsed_time/3600:.2f}h, Remaining: {remaining_time/3600:.2f}h")

        if remaining_time < 1800:  # <30 min
            print("Less than 30 min left. Saving and stopping.")
            save_and_push_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch+1,
                loss=avg_loss,
                tokenizer=tokenizer,
                repo_id=repo_id,
                hf_token=hf_token,
                path=checkpoint_dir
            )
            break

        model.train()

    print("Training completed successfully!")
    return model