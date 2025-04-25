from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer
from datasets import load_from_disk
import torch
import os

# Load tokenizer and dataset
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = load_from_disk("data/processed/imdb_dataset")  # adjust if saved elsewhere

# Initialize model: 2 labels for positive/negative
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Training args
training_args = TrainingArguments(
    output_dir="models/sentiment",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="logs/sentiment",
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

# Train!
if __name__ == "__main__":
    trainer.train()
    trainer.save_model("models/sentiment")
    print("✅ Sentiment model saved to models/sentiment")
