"""
The generation chain is the final step of your graph.
It runs after all retrieval, filtering, and optional search.
It combines everything to generate a high-quality, accurate answer.
"""
from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
load_dotenv()

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,api_key = os.getenv("OPENAI_API_KEY")
)
prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()
