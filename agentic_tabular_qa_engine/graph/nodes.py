from agents.source_router import detect_source
from agents.file_loader import load_file
from agents.sql_planner_agent import generate_sql_plan
from agents.sql_reflection_agent import reflect_sql_plan
from agents.sql_executor import execute_sql_plan
from agents.cleaning_agent import clean_dataframe
from agents.entity_resolution_agent import resolve_entities
from agents.qa_agent import answer_question
from utils.sql_validator import validate_sql_plan


def source_router_node(state):
    return {"source_type": detect_source(state["data_source"])}


def load_data_node(state):
    if state["source_type"] == "sql":
        return state
    df = load_file(state["data_source"])
    return {"raw_df": df}


def sql_flow_node(state):
    plan = generate_sql_plan(state["user_query"], state["schema"])
    plan = reflect_sql_plan(plan, state["schema"])
    validate_sql_plan(plan, state["schema"])

    df = execute_sql_plan(plan, state["data_source"])
    return {"raw_df": df}


def cleaning_node(state):
    df = clean_dataframe(state["raw_df"])
    df = resolve_entities(df)
    return {"cleaned_df": df}


def qa_node(state):
    answer = answer_question(state["cleaned_df"], state["user_query"])
    return {"answer": answer}
