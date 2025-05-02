from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
import torch
from torch.nn import BCEWithLogitsLoss

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = load_from_disk("data/processed/goemotions_sentiment") 

# Encode labels
def one_hot_encode_labels(example):
    label_vector = [0] * 28
    for label in example["labels"]:
        label_vector[label] = 1
    example["labels"] = [float(x) for x in label_vector] 
    return example

dataset = dataset.map(one_hot_encode_labels)

# Initializeing model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=28, 
    problem_type="multi_label_classification"
).to(device)

def compute_loss(outputs, labels, **kwargs):
    logits = outputs.logits
    labels = labels.float()

    loss_fn = BCEWithLogitsLoss()
    loss = loss_fn(logits, labels)

    return loss


# Training arguments
training_args = TrainingArguments(
    output_dir="models/emotion",
    save_strategy="epoch",
    eval_strategy="epoch", 
    learning_rate=2e-5,
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="logs/emotion",
    load_best_model_at_end=True,
    logging_steps=200, 
    report_to="tensorboard",  
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_loss_func=compute_loss,  
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model("models/emotion")
