from datasets import load_dataset, Dataset
import pandas as pd
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return Dataset.from_pandas(df)
