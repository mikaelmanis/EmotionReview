from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
import torch
import os

# Load tokenizer and dataset
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = load_from_disk("data/processed/goemotions_sentiment")  # update if needed

# Update label field: GoEmotions uses multilabel format (list of ints)
# You need to convert it to a binary vector of size 28
def one_hot_encode_labels(example):
    label_vector = [0] * 28
    for label in example["labels"]:
        label_vector[label] = 1
    example["labels"] = label_vector
    return example

dataset = dataset.map(one_hot_encode_labels)

# Format dataset for PyTorch
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Initialize model for 28-label multilabel classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=28, problem_type="multi_label_classification")

# Training args
training_args = TrainingArguments(
    output_dir="models/emotion",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="logs/emotion",
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
)

# Train!
if __name__ == "__main__":
    trainer.train()
    trainer.save_model("models/emotion")
    print("✅ Emotion model saved to models/emotion")
