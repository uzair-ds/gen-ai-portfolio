from dotenv import load_dotenv
load_dotenv()

import os

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from agents.ingestion_agent import ingest_document
from agents.retrieval_agent import retrieve_context
from agents.answer_agent import generate_answer
from agents.citation_agent import extract_citations
from agents.guardrail_agent import guardrail_check
from agents.summarization_agent import summarize_document

from config import (
    DOCUMENT_PATH,
    VECTOR_STORE_PATH,
    TOP_K,
    SIMILARITY_THRESHOLD,
    HASH_FILE_PATH
)

from file_hash import compute_file_hash


def build_vector_store():
    print("📄 Building / refreshing vector store...")

    docs = ingest_document(DOCUMENT_PATH)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    db.save_local(VECTOR_STORE_PATH)

    # Save document hash
    doc_hash = compute_file_hash(DOCUMENT_PATH)
    with open(HASH_FILE_PATH, "w") as f:
        f.write(doc_hash)

    print("✅ Vector store is up to date.")


def vector_store_is_stale() -> bool:
    if not os.path.exists(HASH_FILE_PATH):
        return True

    stored_hash = open(HASH_FILE_PATH).read().strip()
    current_hash = compute_file_hash(DOCUMENT_PATH)

    return stored_hash != current_hash


def load_vector_store():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

def is_summarization_query(question: str) -> bool:
    keywords = ["summarize", "summary", "overview"]
    return any(k in question.lower() for k in keywords)

def answer_question(question: str):
    # ---- 1. ROUTE SUMMARIZATION QUERIES FIRST ----
    if is_summarization_query(question):
        docs = ingest_document(DOCUMENT_PATH)
        summary = summarize_document(docs)
        return {
            "answer": summary,
            "references": [DOCUMENT_PATH]
        }

    # ---- 2. OTHERWISE: NORMAL RAG Q&A FLOW ----
    db = load_vector_store()

    context_docs = retrieve_context(
        db,
        question,
        TOP_K,
        SIMILARITY_THRESHOLD
    )

    if not guardrail_check(context_docs):
        return {
            "answer": "I do not have enough information in the provided document.",
            "references": []
        }

    answer = generate_answer(context_docs, question)
    citations = extract_citations(context_docs)

    return {
        "answer": answer,
        "references": citations
    }




if __name__ == "__main__":
    if (
        not os.path.exists(VECTOR_STORE_PATH)
        or vector_store_is_stale()
    ):
        build_vector_store()
    else:
        print("✅ Vector store already up to date.")

    while True:
        question = input("\nAsk a question (type 'exit' to quit): ")
        if question.lower() == "exit":
            break

        response = answer_question(question)

        print("\n🧠 Answer:")
        print(response["answer"])

        print("\n📚 References:")
        for ref in response["references"]:
            print("-", ref)
