def detect_source(source: str) -> str:
    if source.startswith("postgres") or source.startswith("mysql"):
        return "sql"
    if source.endswith(".csv"):
        return "csv"
    if source.endswith(".xlsx"):
        return "excel"
    raise ValueError("Unsupported data source")
