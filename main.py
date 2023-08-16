from load_data import load_data
from preprocess_data import tokenize_and_pad
from train_transformer import train_model
import torch
from datasets import load_dataset

data = load_dataset("squad_v2")

# Convert sentences to tokenized and padded format
texts = tokenize_and_pad(data['text'].tolist(), max_len=100)
labels = data['label'].tolist()

# Convert to PyTorch tensors
texts_tensor = torch.tensor(texts, dtype=torch.long)
labels_tensor = torch.tensor(labels, dtype=torch.long)

train_model(texts_tensor, labels_tensor)

