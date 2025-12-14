from typing import TypedDict, List
from langchain_core.documents import Document

class AgentState(TypedDict, total=False):
    question: str
    docs: List[Document]
    context_docs: List[Document]
    answer: str
    references: List[str]
    self_rag_decision: str
