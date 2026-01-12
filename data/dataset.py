# data/dataset.py
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import warnings
import os
from decord import VideoReader, cpu
import torchvision.transforms as transforms
import itertools
import numpy as np
from .tokenizer import ProperTokenizer

warnings.filterwarnings("ignore")

class TextVideoDataset(Dataset):
    def __init__(self, max_samples=50, num_frames=8, height=32, width=32, cache_dir='./video_cache'):
        print(f"Loading FineVideo dataset with streaming (max_samples={max_samples})...")
        self.hf_dataset = load_dataset("HuggingFaceFV/finevideo", split="train", streaming=True)
        self.tokenizer = ProperTokenizer()
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        # Take only max_samples for efficiency
        self.samples = list(itertools.islice(self.hf_dataset, max_samples))
        print(f"Loaded {len(self.samples)} samples from FineVideo dataset.")

    def _load_video(self, mp4_bytes, original_filename):
        key = original_filename.replace('/', '_').replace('\\', '_')  # Sanitize filename
        cache_path = os.path.join(self.cache_dir, f"{key}.pt")
        if os.path.exists(cache_path):
            return torch.load(cache_path)

        video_path = os.path.join(self.cache_dir, f"{key}.mp4")
        with open(video_path, 'wb') as f:
            f.write(mp4_bytes)

        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)

            if total_frames == 0:
                raise ValueError("No frames in video")

            # Uniformly sample num_frames
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = vr.get_batch(frame_indices).asnumpy()

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.height, self.width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

            video_frames = [transform(frame) for frame in frames]
            video = torch.stack(video_frames)  # T, C, H, W
            video = video.permute(1, 0, 2, 3)  # C, T, H, W

            torch.save(video, cache_path)
            return video

        except Exception as e:
            print(f"Error loading video {original_filename}: {e}")
            # Fallback to zero tensor for training continuity
            return torch.zeros(3, self.num_frames, self.height, self.width)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['json']['content_metadata']['description']
        video = self._load_video(sample['mp4'], sample['json']['original_video_filename'])
        text_tokens = self.tokenizer.encode(text)

        return {
            'text_tokens': text_tokens,
            'video': video,
            'text': text
        }