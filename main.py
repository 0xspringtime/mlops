# main.py
from load_data import load_data
from preprocess_data import preprocess_data
from train_transformer import train_model

data = load_data()
preprocessed_data = preprocess_data(data)
train_model(preprocessed_data)
