from langchain_openai import ChatOpenAI
import json

llm = ChatOpenAI(temperature=0)

def generate_sql_plan(user_query: str, schema: dict) -> dict:
    prompt = f"""
You are a SQL planning agent.
Schema:
{schema}

User request:
{user_query}

Return a SAFE SQL PLAN in JSON.
No raw SQL.
"""
    response = llm.invoke(prompt).content
    return json.loads(response)
