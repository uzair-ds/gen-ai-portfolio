"""
So for that we're going to be writing a retrieval grader chain, 
which is going to use structured output
from our LLM and turning it into a Pydantic object that will have 
the information whether this document
is relevant or not. And if the document is not relevant, 
we want to filter it out and keep only the documents which are
relevant to the question.And if not all documents are relevant.
So this means that at least one document is not relevant to our query.
Then we want to mark the web search flag to be true.
"""
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
load_dotenv()
import os

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,api_key = os.getenv("OPENAI_API_KEY")
)


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """
    You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    """
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

