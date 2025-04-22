from datasets import load_dataset, load_from_disk
from transformers import BertTokenizer

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
imdbDataset = load_dataset("imdb")
'''
def clean_text(text):
    return text.strip().replace('\n', ' ')

dataset = imdbDataset.map(lambda x: {"text": clean_text(x["text"])})
# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
# Set the format of the dataset to PyTorch tensors
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])
tokenized_datasets.save_to_disk("imdb_dataset")
'''
# Load the tokenized dataset
loaded_dataset = load_from_disk("imdb_dataset")
train_dataset = loaded_dataset["train"][0]
decoded_text = tokenizer.decode(train_dataset['input_ids'])
print(decoded_text)
# Check the first example
print(decoded_text)




