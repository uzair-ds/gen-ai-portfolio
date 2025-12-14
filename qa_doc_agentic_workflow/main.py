from dotenv import load_dotenv
load_dotenv()

import os

from graph.graph_builder import build_graph
from utils.file_hash import compute_file_hash
from utils.vector_store import load_vector_store

from config import (
    DOCUMENT_PATH,
    VECTOR_STORE_PATH,
    HASH_FILE_PATH,
)

from agents.ingestion_agent import ingest_document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# Vector store build / refresh logic
# --------------------------------------------------

def build_vector_store():
    print("📄 Building / refreshing vector store...")

    docs = ingest_document(DOCUMENT_PATH)
    embeddings = OpenAIEmbeddings()

    db = FAISS.from_documents(docs, embeddings)
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    db.save_local(VECTOR_STORE_PATH)

    # Persist document hash
    with open(HASH_FILE_PATH, "w") as f:
        f.write(compute_file_hash(DOCUMENT_PATH))

    print("✅ Vector store ready.")


def vector_store_is_stale() -> bool:
    if not os.path.exists(HASH_FILE_PATH):
        return True

    stored_hash = open(HASH_FILE_PATH).read().strip()
    current_hash = compute_file_hash(DOCUMENT_PATH)

    return stored_hash != current_hash


# --------------------------------------------------
# Application Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    # Ensure vector store is up to date
    if (
        not os.path.exists(VECTOR_STORE_PATH)
        or vector_store_is_stale()
    ):
        build_vector_store()
    else:
        print("✅ Vector store already up to date.")

    # Build LangGraph app
    app = build_graph()

    # Interactive CLI loop
    while True:
        question = input("\nAsk a question (type 'exit'): ")
        if question.lower() == "exit":
            break

        result = app.invoke(
            {
                "question": question
            }
        )

        print("\n🧠 Answer:")
        print(result.get("answer", ""))

        print("\n📚 References:")
        for ref in result.get("references", []):
            print("-", ref)
