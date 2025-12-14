def clean_dataframe(df):
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    return df
