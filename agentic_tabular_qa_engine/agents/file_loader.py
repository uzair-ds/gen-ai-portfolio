import pandas as pd

def load_file(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".xlsx"):
        return pd.read_excel(path)
    raise ValueError("Unsupported file type")
