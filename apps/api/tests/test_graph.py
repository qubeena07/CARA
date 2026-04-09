import pytest
from graph.state import GraphState


def make_state(**kwargs) -> GraphState:
    defaults = {
        "query": "test query",
        "query_id": "test-id",
        "session_id": None,
        "documents": [],
        "graded_documents": [],
        "answer": None,
        "confidence": None,
        "retry_count": 0,
        "hallucination_detected": False,
        "rewritten_query": None,
    }
    defaults.update(kwargs)
    return defaults


def test_initial_state():
    state = make_state()
    assert state["query"] == "test query"
    assert state["retry_count"] == 0
    assert not state["hallucination_detected"]


def test_state_with_documents():
    docs = [{"id": "1", "content": "hello", "source": "test.pdf", "grade": "relevant"}]
    state = make_state(documents=docs)
    assert len(state["documents"]) == 1
