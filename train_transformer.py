# train_transformer.py
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer

def train_model(tokenized_data, model_name, train_args):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2) # This needs to change for SQuAD
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # No separate labels in this case, as SQuAD isn't a classification task
    dataset = list(zip(tokenized_data['input_ids'], tokenized_data['attention_mask']))

    # Modify the trainer to work without explicit labels (labels are in tokenized_data if provided)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,  # This needs adjustment for SQuAD
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained("./saved_model/")

