from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from textstylex.src.preprocessing import tokenize_data

def train_model(tokenizer, model, dataset):
    tokenized_dataset = tokenize_data(dataset, tokenizer)

    args = TrainingArguments(
        output_dir="/content/textstylex/outputs",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        logging_steps=50,
        save_strategy="no",
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained("/content/textstylex/saved_model/")
    tokenizer.save_pretrained("/content/textstylex/saved_model/")
