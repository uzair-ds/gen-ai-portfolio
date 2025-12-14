from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    temperature=0,
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY") # or gpt-3.5-turbo if needed
)

SYSTEM_PROMPT = """
You are a document-grounded AI assistant.
You must answer ONLY using the provided document context.
If the answer is not explicitly present, respond with:
"I do not have enough information in the provided document."
"""

def generate_answer(context_docs, question):
    context = "\n\n".join([doc.page_content for doc in context_docs])

    prompt = f"""
{SYSTEM_PROMPT}

CONTEXT:
{context}

QUESTION:
{question}
"""

    return llm.invoke(prompt).content
