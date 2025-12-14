import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_document(path: str):
    """
    Load document based on file extension.
    Supported formats: PDF, TXT, DOCX
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext == ".txt":
        loader = TextLoader(path, encoding="utf-8")
    elif ext == ".docx":
        loader = UnstructuredWordDocumentLoader(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return loader.load()


def ingest_document(path: str):
    """
    Ingests PDF / TXT / DOCX documents and applies
    chunking suitable for RAG over long-form text.
    """

    pages = load_document(path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=180,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
        ],
    )

    chunks = splitter.split_documents(pages)

    # Ensure consistent metadata
    for chunk in chunks:
        chunk.metadata["source"] = path
        chunk.metadata.setdefault("page", "N/A")

    return chunks
