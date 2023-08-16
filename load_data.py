from datasets import load_dataset

def load_data(path=None):
    dataset = load_dataset('imdb')
    # For simplicity, let's just use the train set.
    return dataset['train']

if __name__ == "__main__":
    data = load_data()
    import pickle
    with open('loaded_data.pkl', 'wb') as f:
        pickle.dump(data, f)

