"""
we'll implement the web searching node.
We'll be leveraging the Tavelli search engine, which is highly optimized to downstream the search results
into an LM based application.
"""

from typing import Any, Dict

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_tavily import TavilySearch

from graph_.state import GraphState

load_dotenv()
web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    if "documents" in state: # if the route to web search in first time then give error
        documents = state["documents"]
        
    tavily_results = web_search_tool.invoke({"query": question})["results"]
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_result)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}
