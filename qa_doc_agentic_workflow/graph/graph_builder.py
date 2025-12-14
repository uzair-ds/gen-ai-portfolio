from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.nodes import (
    intent_router_node,
    route_intent,

    # Summarization
    summarization_node,

    # Q&A + Self-RAG
    retrieval_node,
    answer_node,
    self_rag_node,
    retrieve_again_node,
    citation_node,
)


def build_graph():
    """
    Builds and compiles the LangGraph execution graph
    with Self-RAG for grounded Q&A and safe summarization.
    """

    graph = StateGraph(AgentState)

    # -------------------------------------------------
    # Nodes (ALL nodes must return dicts)
    # -------------------------------------------------
    graph.add_node("intent_router", intent_router_node)

    # Summarization
    graph.add_node("summarize", summarization_node)

    # Q&A + Self-RAG
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("answer", answer_node)
    graph.add_node("self_rag", self_rag_node)
    graph.add_node("retrieve_again", retrieve_again_node)
    graph.add_node("cite", citation_node)

    # -------------------------------------------------
    # Entry point
    # -------------------------------------------------
    graph.set_entry_point("intent_router")

    # -------------------------------------------------
    # Intent-based routing
    # -------------------------------------------------
    graph.add_conditional_edges(
        "intent_router",
        route_intent,
        {
            "summarize": "summarize",
            "qa": "retrieve",
        },
    )

    # -------------------------------------------------
    # Summarization flow
    # -------------------------------------------------
    graph.add_edge("summarize", END)

    # -------------------------------------------------
    # Q&A + Self-RAG flow
    # -------------------------------------------------
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "self_rag")

    graph.add_conditional_edges(
    "self_rag",
    lambda state: state["self_rag_decision"],
    {
        "sufficient": "cite",
        "insufficient": "retrieve_again",
        "unsupported": END,
    },
)

    graph.add_edge("retrieve_again", "answer")
    graph.add_edge("cite", END)

    # -------------------------------------------------
    # Compile graph
    # -------------------------------------------------
    return graph.compile()
