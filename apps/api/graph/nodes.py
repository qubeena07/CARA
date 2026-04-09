from .state import GraphState


def retrieve_node(state: GraphState) -> GraphState:
    """Retrieve documents from vector store."""
    # TODO: implement retrieval via ingestion.retriever
    state["documents"] = []
    return state


def grade_node(state: GraphState) -> GraphState:
    """Grade retrieved documents for relevance."""
    # TODO: implement grading with Gemini
    state["graded_documents"] = state["documents"]
    return state


def rewrite_node(state: GraphState) -> GraphState:
    """Rewrite query if no relevant documents found."""
    state["rewritten_query"] = state["query"]
    state["retry_count"] = state.get("retry_count", 0) + 1
    return state


def generate_node(state: GraphState) -> GraphState:
    """Generate answer from graded documents."""
    # TODO: implement generation with Gemini
    state["answer"] = ""
    state["confidence"] = 0.0
    return state


def hallucination_check_node(state: GraphState) -> GraphState:
    """Check answer for hallucinations."""
    state["hallucination_detected"] = False
    return state
