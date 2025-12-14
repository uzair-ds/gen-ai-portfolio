from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.nodes import (
    intent_router_node,
    route_intent,
    summarization_node,
    retrieval_node,
    guardrail_node,
    answer_node,
    citation_node,
)


def build_graph():
    """
    Builds and compiles the LangGraph execution graph
    for document Q&A and summarization.
    """

    graph = StateGraph(AgentState)

    # -------------------------
    # Nodes (must return dict)
    # -------------------------
    graph.add_node("intent_router", intent_router_node)
    graph.add_node("summarize", summarization_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("answer", answer_node)
    graph.add_node("cite", citation_node)

    # -------------------------
    # Entry point
    # -------------------------
    graph.set_entry_point("intent_router")

    # -------------------------
    # Conditional routing
    # (router function returns a string)
    # -------------------------
    graph.add_conditional_edges(
        "intent_router",
        route_intent,
        {
            "summarize": "summarize",
            "qa": "retrieve",
        },
    )

    # -------------------------
    # Q&A flow
    # -------------------------
    graph.add_edge("retrieve", "guardrail")
    graph.add_edge("guardrail", "answer")
    graph.add_edge("answer", "cite")
    graph.add_edge("cite", END)

    # -------------------------
    # Summarization flow
    # -------------------------
    graph.add_edge("summarize", END)

    return graph.compile()
