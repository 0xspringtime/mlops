# train_transformer.py
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer

def train_model(preprocessed_data, model_name, train_args):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2) # Assuming binary classification. Modify if different.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_data, labels = preprocessed_data

    dataset = list(zip(tokenized_data['input_ids'], tokenized_data['attention_mask'], labels))
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained("./saved_model/")

