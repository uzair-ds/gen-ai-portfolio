from langchain_openai import ChatOpenAI
import json

llm = ChatOpenAI(temperature=0)

def reflect_sql_plan(plan: dict, schema: dict) -> dict:
    prompt = f"""
You are a SQL reflection agent.
Review the SQL plan below for correctness and safety.

Schema:
{schema}

SQL PLAN:
{plan}

Return a corrected JSON plan if needed.
"""
    response = llm.invoke(prompt).content
    return json.loads(response)
