# Text Style Transfer – Shakespearean Rewriting with T5

This project demonstrates an end-to-end text style transfer system that converts modern English into Shakespearean English using a fine-tuned T5 model.


## Project Structure

```plaintext
textstylex/
  ├── data/
  │   ├── modern.txt                  -> Modern English sentences
  │   ├── shakespeare.txt            -> Parallel Shakespearean sentences
  │   └── shakespeare_dataset.csv    -> Combined CSV (input_text, target_text, style)
  ├── src/
  │   ├── Load_data.py               -> Loads the CSV and converts to HuggingFace Dataset
  │   ├── preprocessing.py           -> Tokenization and preprocessing for T5
  │   └── train.py                   -> Fine-tunes the T5 model on the dataset
  ├── outputs/                       -> Training logs and checkpoints (ignored in .gitignore)
  ├── main.py                        -> Training entry point
  └── saved_model/                   -> Saved model weights (~230MB, excluded from GitHub)

```
## Dataset & Training Setup

- Total examples: ~18,395 sentence pairs
- Model: t5-small
- Training time: ~45 min per 5k samples on GPU


## Training the Model


Run the training pipeline:

    python main.py

This will:
- Load the preprocessed dataset
- Tokenize input-target pairs with "shakespeare: ..." prefix
- Fine-tune the T5 model
- Save the model and tokenizer to /saved_model


## Inference Example


from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("textstylex/saved_model")
model = T5ForConditionalGeneration.from_pretrained("textstylex/saved_model")

def generate_style(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    output_ids = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(generate_style("shakespeare: I am feeling tired."))


## Sample Predictions


| Input                                   | Output                              |
|----------------------------------------|-------------------------------------|
| shakespeare: I’m sorry                 | I do beseech thee, forgive me       |
| shakespeare: Let’s go                  | Come, let us away                   |
| shakespeare: Thank you                 | I thank thee heartily               |


## What's Not Included


- The `saved_model/` directory is excluded from GitHub (230MB+)
- Re-run training to reproduce results


## Setup Dependencies


    pip install transformers datasets


