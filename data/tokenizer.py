# data/tokenizer.py
from transformers import GPT2Tokenizer
import torch

class ProperTokenizer:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, text, max_length=77):
        if isinstance(text, str):
            encoding = self.tokenizer(text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
            return encoding['input_ids'].squeeze(0)
        else:
            # If already tokens
            return torch.tensor(text[:max_length])

    def save_pretrained(self, save_directory):
        self.tokenizer.save_pretrained(save_directory)