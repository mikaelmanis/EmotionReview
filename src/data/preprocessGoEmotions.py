import torch
from datasets import load_dataset

# Load GoEmotions
goemotions = load_dataset("go_emotions")

# Function to prepare multi-hot encoding
def encode_labels(example):
    labels = torch.zeros(28)  # 28 emotions
    labels[example['labels']] = 1
    example['labels'] = labels.tolist()
    return example

# Apply it
encoded_goemotions = goemotions.map(encode_labels)
