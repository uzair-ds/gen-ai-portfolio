from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def ingest_document(path: str):
    """
    Ingests a scientific PDF and applies section-aware,
    overlapping chunking suitable for RAG over research papers.
    """

    loader = PyPDFLoader(path)
    pages = loader.load()

    # Filter out references section (usually starts after "References")
    filtered_pages = []
    for page in pages:
        text = page.page_content.lower()
        if "references" in text and page.metadata.get("page", 0) > 5:
            break
        filtered_pages.append(page)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,          # larger chunks for scientific text
        chunk_overlap=180,       # preserve experiment continuity
        separators=[
            "\n\n",              # paragraphs
            "\n",
            ". ",
            " ",
        ],
    )

    chunks = splitter.split_documents(filtered_pages)

    # Add explicit metadata for traceability
    for chunk in chunks:
        chunk.metadata["source"] = path
        chunk.metadata.setdefault("section", "Main Text")

    return chunks
