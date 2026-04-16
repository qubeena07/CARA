from langgraph.graph import StateGraph, END

from .state import GraphState
from .nodes import (
    retrieve_node,
    grade_documents_node,
    rewrite_query_node,
    generate_node,
    hallucination_check_node
)
from .config import MAX_RETRIES, MAX_REGENERATIONS, MIN_RELEVANT_DOCS


# ── Routing functions ─────────────────────────────────────────────────────────

def route_after_grading(state: GraphState) -> str:
    """
    Called after grade_documents_node completes.
    Decides: generate now, or rewrite and try again?

    Logic:
    - If enough relevant docs found → generate
    - If no relevant docs AND still have retries left → rewrite
    - If no relevant docs AND out of retries → generate anyway (will return "cannot answer")
    """
    docs = state["retrieved_documents"]
    relevant_docs = [d for d in docs if d.grade == "relevant"]
    retry_count = state.get("retry_count", 0)

    if len(relevant_docs) >= MIN_RELEVANT_DOCS:
        return "generate"
    elif retry_count < MAX_RETRIES:
        return "rewrite_query"
    else:
        # Out of retries — generate will handle the empty case gracefully
        return "generate"


def route_after_hallucination_check(state: GraphState) -> str:
    """
    Called after hallucination_check_node completes.
    Decides: accept the answer, or regenerate?

    Logic:
    - If hallucination detected AND still have regeneration budget → regenerate
    - If hallucination detected AND out of budget → accept anyway (log the issue)
    - If no hallucination → end
    """
    hallucinating = state.get("hallucination_detected", False)
    regen_count = state.get("regeneration_count", 0)

    if hallucinating and regen_count <= MAX_REGENERATIONS:
        return "generate"
    else:
        return END


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(db):
    """
    Compiles the LangGraph state machine.

    The db (database session) is captured in the closure for the retrieve node.
    This is why retrieve_node takes db as a second argument — LangGraph nodes
    only receive the state, so we use a lambda to inject the db session.
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("retrieve", lambda s: retrieve_node(s, db))
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("hallucination_check", hallucination_check_node)

    # Entry point — first node to run
    workflow.set_entry_point("retrieve")

    # Fixed edges (always go to this next node)
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("rewrite_query", "retrieve")    # after rewriting, retrieve again

    # Conditional edges (routing function decides next node)
    workflow.add_conditional_edges(
        source="grade_documents",
        path=route_after_grading,
        path_map={
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        }
    )

    workflow.add_edge("generate", "hallucination_check")

    workflow.add_conditional_edges(
        source="hallucination_check",
        path=route_after_hallucination_check,
        path_map={
            "generate": "generate",
            END: END,
        }
    )

    return workflow.compile()
