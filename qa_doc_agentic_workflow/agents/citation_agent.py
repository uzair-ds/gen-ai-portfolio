def extract_citations(context_docs):
    citations = set()

    for doc in context_docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        citations.add(f"{source} - Page {page}")

    return list(citations)