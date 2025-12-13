"""
This node is going to simply take the question and 
take the documents from our state and simply run the chain.
"""

from typing import Any, Dict
 
from graph_.chains.generation import generation_chain
from graph_.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}