"""
# grade documents - 
And here we're going to write our node implementation of grading all the documents.
So to decide whether we want to filter them out or to keep them in.
So we're going to define a function which will receive the state.
And in that state we're going to have already the fetched documents.
We're going to iterate through all the documents.
And our greater chain is going to decide for each document whether it's relevant or not.
If it's not relevant, we're going to filter it out.
And finally, if we have found any document that's not relevant, we're going to change the web searching
flag to be true so we can go and later on search for that query.
So we'll have additional information since not all of the documents are relevant for us.
"""

from typing import Any, Dict

from graph_.chains.retrieval_grader import retrieval_grader
from graph_.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}