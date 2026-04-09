from typing import TypedDict, List, Optional


class GraphState(TypedDict):
    query: str
    query_id: str
    session_id: Optional[str]
    documents: List[dict]
    graded_documents: List[dict]
    answer: Optional[str]
    confidence: Optional[float]
    retry_count: int
    hallucination_detected: bool
    rewritten_query: Optional[str]
