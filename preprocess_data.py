import torch

def preprocess_data(data, model_name, max_len):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    examples = []
    for question, context, answer in zip(data['question'], data['context'], data['answers']):
        encoding = tokenizer(question, context, padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")
        # Find the start and end token position of the answer in the context
        start_idx = context.find(answer['text'][0])
        end_idx = start_idx + len(answer['text'][0])
        start_token = len(tokenizer(context[:start_idx])['input_ids'])
        end_token = start_token + len(tokenizer(answer['text'][0])['input_ids']) - 1

        examples.append({
            'input_ids': encoding['input_ids'].squeeze(0), 
            'attention_mask': encoding['attention_mask'].squeeze(0), 
            'start_token': start_token, 
            'end_token': end_token
        })
    
    return examples


