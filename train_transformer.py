from transformers import BertForSequenceClassification, Trainer, TrainingArguments

def train_model(preprocessed_data):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        num_train_epochs=1,
        logging_dir='./logs',
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=preprocessed_data,
    )
    trainer.train()
    model.save_pretrained("model/")

if __name__ == "__main__":
    import pickle
    with open('preprocessed_data.pkl', 'rb') as f:
        preprocessed_data = pickle.load(f)
    train_model(preprocessed_data)

