from agents.ingestion_agent import ingest_document
from agents.retrieval_agent import retrieve_context
from agents.answer_agent import generate_answer
from agents.summarization_agent import summarize_document
from agents.citation_agent import extract_citations
from agents.guardrail_agent import guardrail_check

from config import DOCUMENT_PATH, TOP_K, SIMILARITY_THRESHOLD
from utils.vector_store import load_vector_store

# -------- Intent Router Node (returns dict ONLY) --------
def intent_router_node(state):
    return state


# -------- Routing Function (returns string) --------
def route_intent(state):
    q = state["question"].lower()
    if any(k in q for k in ["summarize", "summary", "overview"]):
        return "summarize"
    return "qa"


# -------- Summarization Node --------
def summarization_node(state):
    docs = ingest_document(DOCUMENT_PATH)
    summary = summarize_document(docs)

    return {
        "answer": summary,
        "references": [DOCUMENT_PATH],
    }


# -------- Retrieval Node --------
def retrieval_node(state):
    db = load_vector_store()
    context_docs = retrieve_context(
        db,
        state["question"],
        TOP_K,
        SIMILARITY_THRESHOLD,
    )
    return {"context_docs": context_docs}


# -------- Guardrail Node --------
def guardrail_node(state):
    if not guardrail_check(state["context_docs"]):
        return {
            "answer": "I do not have enough information in the provided document.",
            "references": [],
        }
    return state


# -------- Answer Node --------
def answer_node(state):
    answer = generate_answer(
        state["context_docs"],
        state["question"],
    )
    return {"answer": answer}


# -------- Citation Node --------
def citation_node(state):
    citations = extract_citations(state["context_docs"])
    return {"references": citations}
