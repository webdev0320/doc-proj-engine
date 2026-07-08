from datasets import load_dataset
from torch.utils.data import DataLoader
from train import DonutDataset, processor, task_prompt, model
import torch

print('Loading dataset...')
dataset = load_dataset('imagefolder', data_dir='./data', split='train')
train_dataset = DonutDataset(dataset, processor, task_prompt)
loader = DataLoader(train_dataset, batch_size=1)

print('Getting first batch...')
batch = next(iter(loader))
print(f"Batch pixel_values shape: {batch['pixel_values'].shape}")
print(f"Batch labels shape: {batch['labels'].shape}")

print('Running forward pass...')
outputs = model(pixel_values=batch['pixel_values'], labels=batch['labels'])
print(f"Loss: {outputs.loss.item()}")
