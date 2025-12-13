"""
In LangGraph, the graph state is the shared memory (a dictionary) that flows through the nodes in a computation graph. It represents the evolving data as the graph executes. 
Each node in the graph reads from and writes to this state.
"""

from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]

"""
Think of Graph State As:
A structured dictionary that keeps track of:
Inputs (e.g., user question)
Intermediate results (e.g., search results, LLM outputs)
Control flags (e.g., whether to run a web search)
Final outputs
"""