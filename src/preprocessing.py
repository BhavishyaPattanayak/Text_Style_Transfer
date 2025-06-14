def tokenize_data(dataset, tokenizer, max_input_length=64, max_target_length=64):
    def preprocess(example):
        model_input = tokenizer(example['input_text'], max_length=max_input_length, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(example['target_text'], max_length=max_target_length, truncation=True)
        model_input['labels'] = labels['input_ids']
        return model_input
    return dataset.map(preprocess, batched=True)
