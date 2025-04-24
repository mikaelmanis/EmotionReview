# src/data/prep_goemotions.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from datasets import load_dataset
import pandas as pd
from src.models.emotionMapping import emotionSentiment, emotionID

def load_goemotions():
    dataset = load_dataset("go_emotions")
    return dataset

def map_labels_to_sentiment(example):
    # Map each emotion ID to sentiment polarity
    sentiments = set()
    for label in example["labels"]:
        sentiment = emotionSentiment.get(label, "neutral")
        sentiments.add(sentiment)
    # Return most dominant class
    if "positive" in sentiments and "negative" not in sentiments:
        return "positive"
    elif "negative" in sentiments and "positive" not in sentiments:
        return "negative"
    elif "positive" in sentiments and "negative" in sentiments:
        return "mixed"
    else:
        return "neutral"

def preprocess_goemotions(dataset):
    # Apply the mapping function to each sample
    dataset = dataset.map(lambda x: {"sentiment": map_labels_to_sentiment(x)})
    return dataset

if __name__ == "__main__":
    dataset = load_goemotions()
    processed = {}
    for split in ["train", "validation", "test"]:
        processed[split] = preprocess_goemotions(dataset[split])
        print(f"{split}: {processed[split].num_rows} samples")
        
        print(processed[split][0])  # print one sample

