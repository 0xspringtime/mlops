from datasets import load_dataset

def load_data(dataset_name, split='train'):
    """
    Load a dataset from HuggingFace's datasets.

    Args:
    - dataset_name (str): The name of the dataset to load from HuggingFace's datasets.
    - split (str, optional): The split of the dataset to load ('train', 'validation', 'test', etc.). Defaults to 'train'.

    Returns:
    - Dataset: The loaded dataset.
    """
    data = load_dataset(dataset_name, split=split)
    return data

