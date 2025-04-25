import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from datasets import load_dataset, DatasetDict
from src.models.emotionMapping import emotionSentiment
import pandas as pd

def load_goemotions():
    dataset = load_dataset("go_emotions")
    return dataset

# Maps GoEmotion emotion labels to sentiment
def map_labels_to_sentiment(example):
    sentiments = set()
    for label in example["labels"]:
        sentiment = emotionSentiment.get(label, "neutral")
        sentiments.add(sentiment)

    if "positive" in sentiments and "negative" not in sentiments:
        return "positive"
    elif "negative" in sentiments and "positive" not in sentiments:
        return "negative"
    elif "positive" in sentiments and "negative" in sentiments:
        return "mixed"
    else:
        return "neutral"

def preprocess_goemotions(dataset_split):
    # Ensure we're mapping individual examples
    return dataset_split.map(
        lambda x: {"sentiment": map_labels_to_sentiment(x)},
        batched=False
    )

if __name__ == "__main__":
    dataset = load_goemotions()
    processed = {}

    for split in ["train", "validation", "test"]:
        processed[split] = preprocess_goemotions(dataset[split])
        print(f"{split}: {processed[split].num_rows} samples")
        print(processed[split][0])  # print first item to confirm

if __name__ == "__main__":
    dataset = load_goemotions()
    processed = {}

    for split in ["train", "validation", "test"]:
        processed[split] = preprocess_goemotions(dataset[split])
        print(f"{split}: {processed[split].num_rows} samples")
        print(processed[split][0])

    # Wrap in DatasetDict
    processed_dataset = DatasetDict(processed)

    # Save it to a folder
    save_path = os.path.join(os.path.dirname(__file__), "../../data/processed/goemotions_sentiment")
    processed_dataset.save_to_disk(save_path)
    print(f"✅ Saved processed dataset to {save_path}")

