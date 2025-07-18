import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def save(df, path):
    df.to_csv(path, index=False)
    