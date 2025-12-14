from typing import TypedDict, Dict, Any
import pandas as pd

class DataAgentState(TypedDict, total=False):
    user_query: str
    data_source: str
    source_type: str

    schema: Dict[str, Any]
    sql_plan: Dict[str, Any]

    raw_df: pd.DataFrame
    cleaned_df: pd.DataFrame

    answer: str
