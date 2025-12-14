import pandas as pd
from sqlalchemy import create_engine

def execute_sql_plan(plan: dict, db_url: str) -> pd.DataFrame:
    engine = create_engine(db_url)

    # VERY simplified example
    query = f"""
    SELECT *
    FROM {plan['tables'][0]}
    JOIN {plan['tables'][1]}
    ON {plan['join']}
    """

    return pd.read_sql(query, engine)
