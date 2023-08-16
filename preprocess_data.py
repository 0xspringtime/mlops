from transformers import AutoTokenizer

def preprocess_data(data, model_name, max_len):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize and pad
    tokenized_data = tokenizer(data, padding='max_length', max_length=max_len, truncation=True, return_tensors="pt")
    
    return tokenized_data["input_ids"]
