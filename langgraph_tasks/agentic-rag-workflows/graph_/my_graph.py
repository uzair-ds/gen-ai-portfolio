import sys
import os

# Add the root directory of your project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from graph_.constants import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph_.nodes import generate, grade_documents, retrieve, web_search
from graph_.state import GraphState

load_dotenv()

def decide_to_generate(state):
    """
    This function, decide_to_generate(state), 
    is a decision-making step in your LangGraph workflow. 
    It checks whether the current state indicates that additional 
    web search is needed before generating an answer.
    """
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


# Define Graph 
workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

"""
“After the GRADE_DOCUMENTS node runs, 
choose the next node dynamically by evaluating the function decide_to_generate. 
Depending on what it returns, either go to WEBSEARCH or GENERATE.”
"""
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="rag_graph.png")