from datasets import load_dataset

def load_data(dataset_name, split='train'):
    """
    Load a dataset from HuggingFace's datasets and extract contexts and questions.

    Args:
    - dataset_name (str): The name of the dataset to load from HuggingFace's datasets.
    - split (str, optional): The split of the dataset to load ('train', 'validation', 'test', etc.). Defaults to 'train'.

    Returns:
    - Tuple[List[str], List[str]]: A tuple containing a list of contexts and a list of questions.
    """
    data = load_dataset(dataset_name, split=split)
    contexts = [item['context'] for item in data]
    questions = [item['question'] for item in data]
    return contexts, questions

