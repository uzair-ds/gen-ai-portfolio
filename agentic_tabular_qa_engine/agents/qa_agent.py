from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

def answer_question(df, question: str) -> str:
    prompt = f"""
You are a data analyst.
Answer the question using the table below.

TABLE:
{df.head(20)}

QUESTION:
{question}
"""
    return llm.invoke(prompt).content
