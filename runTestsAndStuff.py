from datasets import load_dataset, load_from_disk
from transformers import BertTokenizer

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the tokenized dataset
loaded_dataset = load_from_disk("data/processed/imdb_dataset")
train_dataset = loaded_dataset["train"][0]
decoded_text = tokenizer.decode(train_dataset['input_ids'])
print(decoded_text)
# Check the first example
print(decoded_text)




