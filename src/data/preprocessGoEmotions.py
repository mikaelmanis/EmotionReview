from datasets import load_dataset, DatasetDict
from transformers import BertTokenizer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from render.app.emotionMapping import emotionSentiment


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def load_goemotions():
    return load_dataset("go_emotions")

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

def preprocess_goemotions(dataset):
    dataset = dataset.map(lambda x: {**x, "sentiment": map_labels_to_sentiment(x)}, batched=False)
    dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), batched=True)
    return dataset

if __name__ == "__main__":
    raw_dataset = load_goemotions()
    processed = {split: preprocess_goemotions(raw_dataset[split]) for split in ["train", "validation", "test"]}
    processed_dataset = DatasetDict(processed)

    save_path = os.path.join(os.path.dirname(__file__), "../../data/processed/goemotions_sentiment")
    processed_dataset.save_to_disk(save_path)
    print(f"✅ Saved to {save_path}")
