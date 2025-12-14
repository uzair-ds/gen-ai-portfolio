from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    temperature=0,
    model="gpt-4o-mini"  # safe + cheaper for summaries
)

CHUNK_SUMMARY_PROMPT = """
You are a summarization agent.
Summarize the following text concisely.
Use ONLY the provided content.
"""

FINAL_SUMMARY_PROMPT = """
You are a document summarization agent.
Combine the following partial summaries into a coherent,
high-level summary of the document.
Do not add external knowledge.
"""


def summarize_chunks(docs, max_chunks=20):
    """
    Map step: summarize chunks individually
    """
    summaries = []

    # Limit number of chunks to avoid excessive cost
    docs = docs[:max_chunks]

    for doc in docs:
        prompt = f"""
{CHUNK_SUMMARY_PROMPT}

TEXT:
{doc.page_content}
"""
        summary = llm.invoke(prompt).content
        summaries.append(summary)

    return summaries


def summarize_document(docs):
    """
    Reduce step: summarize summaries
    """
    chunk_summaries = summarize_chunks(docs)

    combined = "\n\n".join(chunk_summaries)

    prompt = f"""
{FINAL_SUMMARY_PROMPT}

PARTIAL SUMMARIES:
{combined}
"""
    return llm.invoke(prompt).content