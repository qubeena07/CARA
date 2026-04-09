from langgraph.graph import StateGraph, END
from .state import GraphState
from .nodes import retrieve_node, grade_node, rewrite_node, generate_node, hallucination_check_node
from .config import MAX_RETRIES


def should_rewrite(state: GraphState) -> str:
    graded = state.get("graded_documents", [])
    relevant = [d for d in graded if d.get("grade") == "relevant"]
    if not relevant and state.get("retry_count", 0) < MAX_RETRIES:
        return "rewrite"
    return "generate"


def should_retry_generation(state: GraphState) -> str:
    if state.get("hallucination_detected") and state.get("retry_count", 0) < MAX_RETRIES:
        return "rewrite"
    return "end"


def build_graph() -> StateGraph:
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("hallucination_check", hallucination_check_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges("grade", should_rewrite, {"rewrite": "rewrite", "generate": "generate"})
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", "hallucination_check")
    workflow.add_conditional_edges("hallucination_check", should_retry_generation, {"rewrite": "rewrite", "end": END})

    return workflow.compile()


graph = build_graph()
