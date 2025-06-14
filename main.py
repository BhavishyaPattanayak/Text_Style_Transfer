import sys
sys.path.append('/content')

from transformers import T5Tokenizer, T5ForConditionalGeneration
from textstylex.src.Load_data import load_data
from textstylex.src.train import train_model

def main():
    data_path = "/content/textstylex/data/shakespeare_dataset.csv"
    dataset = load_data(data_path)

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
  
    print("🔧 Training model...")
    train_model(tokenizer, model, dataset)

if __name__ == "__main__":
    main()

