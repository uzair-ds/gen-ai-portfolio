from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    temperature=0,
    model="gpt-4o-mini"
)

SELF_RAG_PROMPT = """
You are a self-evaluation agent.

Question:
{question}

Answer:
{answer}

Retrieved Context:
{context}

Decide ONE of the following:
- "sufficient" → Answer is well-supported by context
- "insufficient" → Answer may need more context
- "unsupported" → Answer is not supported at all

Respond with ONLY one word:
sufficient | insufficient | unsupported
"""

def self_rag_check(question, answer, context_docs):
    context = "\n\n".join(d.page_content for d in context_docs)

    prompt = SELF_RAG_PROMPT.format(
        question=question,
        answer=answer,
        context=context
    )

    decision = llm.invoke(prompt).content.strip().lower()
    return decision
