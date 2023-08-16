from transformers import BertTokenizer

def preprocess_data(data):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized_data = tokenizer(data['text'], truncation=True, padding=True, max_length=256)
    return tokenized_data

if __name__ == "__main__":
    import pickle
    with open('loaded_data.pkl', 'rb') as f:
        data = pickle.load(f)
    preprocessed_data = preprocess_data(data)
    with open('preprocessed_data.pkl', 'wb') as f:
        pickle.dump(preprocessed_data, f)

