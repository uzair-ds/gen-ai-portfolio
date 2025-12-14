from typing import TypedDict, List
from langchain.schema import Document

class AgentState(TypedDict, total=False):
    question: str
    docs: List[Document]
    context_docs: List[Document]
    answer: str
    references: List[str]
