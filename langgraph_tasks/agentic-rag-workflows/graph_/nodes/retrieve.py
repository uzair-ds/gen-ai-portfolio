"""
This node get the state and it is going to extract the question that user has asked.
And it is going to retrive the relevant documents for that state.
It is going to use vector store semantic search.
After this node we should update state documents to hold the relevant documents from our vector store.


"""

from typing import Any, Dict
from graph_.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question} # It is updated the retrived documents from Vector DB and original question
