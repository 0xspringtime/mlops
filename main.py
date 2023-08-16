# main.py
from load_data import load_data
from preprocess_data import preprocess_data
from train_transformer import train_model
from transformers import TrainingArguments
from datasets import load_dataset

model_name = "bert-base-uncased"  # Replace with desired model

data = load_data(squad_v2)
preprocessed_data = preprocess_data(data, model_name, 128)

training_args = TrainingArguments(
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
)

train_model(preprocessed_data, model_name, training_args)

