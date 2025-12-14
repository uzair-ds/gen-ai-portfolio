from langgraph.graph import StateGraph, END
from graph.state import DataAgentState
from graph.nodes import *

def build_graph():
    g = StateGraph(DataAgentState)

    g.add_node("source_router", source_router_node)
    g.add_node("load_data", load_data_node)
    g.add_node("sql_flow", sql_flow_node)
    g.add_node("clean", cleaning_node)
    g.add_node("qa", qa_node)

    g.set_entry_point("source_router")

    g.add_edge("source_router", "load_data")
    g.add_edge("load_data", "sql_flow")
    g.add_edge("sql_flow", "clean")
    g.add_edge("clean", "qa")
    g.add_edge("qa", END)

    return g.compile()
