import torch

def save_preprocessed_data(tokenized_data, path="./preprocessed_data.pt"):
    torch.save(tokenized_data, path)

from transformers import AutoTokenizer

def preprocess_data(contexts, questions, model_name, max_len):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize and pad context-question pairs
    tokenized_data = tokenizer(contexts, questions, padding='max_length', max_length=max_len, truncation=True, return_tensors="pt")
    
    return tokenized_data
