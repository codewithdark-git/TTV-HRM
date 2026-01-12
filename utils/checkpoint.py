# utils/checkpoint.py
import os
import torch
from safetensors.torch import save_file as save_safetensors, load_file as load_safetensors
from huggingface_hub import upload_file, upload_folder, hf_hub_download, list_repo_files

def save_and_push_checkpoint(model, optimizer, scheduler, epoch, loss, tokenizer, repo_id, hf_token, path='./checkpoint', use_safetensors=True):
    os.makedirs(path, exist_ok=True)

    # Save model checkpoint
    if use_safetensors:
        model_filename = f"model_epoch_{epoch}.safetensors"
        model_path = os.path.join(path, model_filename)
        save_safetensors(model.state_dict(), model_path)

        meta_filename = f"checkpoint_meta_epoch_{epoch}.pt"
        meta_path = os.path.join(path, meta_filename)
        torch.save({
            'epoch': epoch,
            'loss': loss,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None
        }, meta_path)

        print(f"Model (safetensors) saved to: {model_path}")
        print(f"Metadata saved to: {meta_path}")
    else:
        model_filename = f"model_checkpoint_epoch_{epoch}.pt"
        model_path = os.path.join(path, model_filename)
        torch.save({
            'epoch': epoch,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None
        }, model_path)
        print(f"Full checkpoint (.pt) saved to: {model_path}")

    # Save tokenizer
    tokenizer_path = os.path.join(path, "tokenizer")
    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Tokenizer saved to: {tokenizer_path}")

    # Upload files to HF if token provided
    if hf_token:
        print(f"Uploading checkpoint to Hugging Face repo: {repo_id}")
        hf_checkpoint_dir = "checkpoints"
        for upload_path in [model_path, meta_path] if use_safetensors else [model_path]:
            if os.path.exists(upload_path):
                try:
                    path_in_repo = os.path.join(hf_checkpoint_dir, os.path.basename(upload_path))
                    upload_file(
                        path_or_fileobj=upload_path,
                        path_in_repo=path_in_repo,
                        repo_id=repo_id,
                        repo_type="model",
                        token=hf_token
                    )
                    print(f"Uploaded {path_in_repo} to {repo_id}")
                except Exception as e:
                    print(f"Failed to upload {os.path.basename(upload_path)}: {e}")

        # Upload tokenizer folder to root
        try:
            upload_folder(
                folder_path=tokenizer_path,
                repo_id=repo_id,
                repo_type="model",
                token=hf_token
            )
            print(f"Uploaded tokenizer folder to {repo_id}")
        except Exception as e:
            print(f"Failed to upload tokenizer: {e}")
    else:
        print("No HF token provided. Skipping upload to Hugging Face.")

def load_checkpoint(model, optimizer, scheduler, path_or_repo, from_hf=True):
    try:
        if from_hf:
            print(f"Loading checkpoint from Hugging Face repo: {path_or_repo}")
            all_files = list_repo_files(path_or_repo, repo_type="model")
            # Filter for checkpoint files in checkpoints/ dir
            files = [f for f in all_files if f.startswith("checkpoints/")]
        else:
            print(f"Loading checkpoint from local path: {path_or_repo}")
            if not os.path.isdir(path_or_repo):
                raise FileNotFoundError(f"Provided path is not a directory: {path_or_repo}")
            files = os.listdir(path_or_repo)

        # Prioritize .pt, then .safetensors
        def get_epoch_key(filename):
            fname = filename.replace("checkpoints/", "") if from_hf else filename
            parts = fname.split('_')
            if len(parts) > 1 and parts[-1].split('.')[0].isdigit():
                return int(parts[-1].split('.')[0])
            return 0

        pt_files = sorted([f for f in files if f.endswith(".pt")], key=get_epoch_key)
        st_files = sorted([f for f in files if f.endswith(".safetensors")], key=get_epoch_key)

        if pt_files:
            latest = pt_files[-1]
            ext = ".pt"
        elif st_files:
            latest = st_files[-1]
            ext = ".safetensors"
        else:
            raise FileNotFoundError("No valid checkpoint found")

        print(f"Loading latest checkpoint: {latest}")

        if from_hf:
            checkpoint_path = hf_hub_download(repo_id=path_or_repo, filename=latest, repo_type="model")
        else:
            checkpoint_path = os.path.join(path_or_repo, latest)

        if ext == ".pt":
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch = checkpoint.get('epoch', 0)
            loss = checkpoint.get('loss', 0.0)

        else:  # .safetensors
            model.load_state_dict(load_safetensors(checkpoint_path))

            # Load metadata separately
            base_name = latest.replace("checkpoints/", "") if from_hf else latest
            meta_name = base_name.replace("model_epoch", "checkpoint_meta_epoch").replace(".safetensors", ".pt")
            if from_hf:
                meta_filename = f"checkpoints/{meta_name}"
                meta_path = hf_hub_download(repo_id=path_or_repo, filename=meta_filename, repo_type="model")
            else:
                meta_path = os.path.join(path_or_repo, meta_name)

            meta = torch.load(meta_path, map_location="cpu")
            optimizer.load_state_dict(meta['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in meta:
                scheduler.load_state_dict(meta['scheduler_state_dict'])
            epoch = meta.get('epoch', 0)
            loss = meta.get('loss', 0.0)

        print(f"Successfully loaded checkpoint for epoch {epoch} with loss {loss:.4f}")
        return model, optimizer, scheduler, int(epoch), float(loss)
    except FileNotFoundError as e:
        print(f"No valid checkpoint found: {e}. Starting from scratch.")
        return model, optimizer, scheduler, 0, 0.0
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting from scratch.")
        return model, optimizer, scheduler, 0, 0.0