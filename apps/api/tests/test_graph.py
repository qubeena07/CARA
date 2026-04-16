"""
Integration test for the full LangGraph pipeline.

Mocks all external I/O (LLM calls, retriever) so the test runs without
a database or API keys. Verifies the graph wires nodes correctly and
that state accumulates as expected.
"""
import sys
import os
import json
import types
from unittest.mock import MagicMock, patch

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
# Allow imports like `from graph.xxx` when running from apps/api/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── fixtures ──────────────────────────────────────────────────────────────────

def make_llm_response(content: str) -> MagicMock:
    """Minimal AIMessage-like mock."""
    msg = MagicMock()
    msg.content = content
    return msg


def make_retrieved_doc(idx: int, score: float = 0.9):
    """RetrievedDocument-like object returned by the retriever."""
    doc = MagicMock()
    doc.id = f"doc-{idx}"
    doc.content = f"Content of document {idx}."
    doc.source = f"file_{idx}.pdf"
    doc.score = score
    return doc


# ── the test ──────────────────────────────────────────────────────────────────

@patch("ingestion.retriever.embed_texts")
@patch("ingestion.retriever.cohere.ClientV2")
@patch("graph.nodes.ChatGoogleGenerativeAI")
def test_happy_path(MockLLMClass, MockCohere, mock_embed):
    """
    Happy path: retriever returns 3 docs, grader marks 2 relevant,
    generator produces an answer, hallucination checker says clean.

    Expected audit_log entries: retrieve, grade, generate, hallucination_check (4).
    The spec says 5 but routing decisions are not nodes — they produce no log entries.
    """

    # ── mock embed so retriever doesn't need a real model ─────────────────────
    mock_embed.return_value = [[0.1] * 768]

    # ── mock Cohere reranker to raise so retriever falls back to cosine ───────
    mock_cohere_instance = MagicMock()
    mock_cohere_instance.rerank.side_effect = Exception("mocked — use fallback")
    MockCohere.return_value = mock_cohere_instance

    # ── mock DB session ───────────────────────────────────────────────────────
    db = MagicMock()
    row1 = MagicMock(id="doc-1", content="Content of document 1.", source="file_1.pdf", score=0.9)
    row2 = MagicMock(id="doc-2", content="Content of document 2.", source="file_2.pdf", score=0.8)
    row3 = MagicMock(id="doc-3", content="Content of document 3.", source="file_3.pdf", score=0.3)
    db.execute.return_value.fetchall.return_value = [row1, row2, row3]

    # ── mock LLM responses ────────────────────────────────────────────────────
    # grade_documents_node calls llm.invoke() once per document.
    # hallucination_check_node calls llm.invoke() once.
    # We set up a single mock instance returned by both LLM constructors.
    llm_instance = MagicMock()

    grade_relevant   = make_llm_response(json.dumps({"grade": "relevant",   "reason": "on-topic", "reasoning": "yes"}))
    grade_irrelevant = make_llm_response(json.dumps({"grade": "irrelevant", "reason": "off-topic", "reasoning": "no"}))
    hallucination_ok = make_llm_response(json.dumps({
        "hallucinating": False,
        "confidence": 0.95,
        "explanation": "All claims are grounded in the provided sources.",
        "claims_checked": []
    }))

    # invoke call order: grade doc1, grade doc2, grade doc3, hallucination check
    llm_instance.invoke.side_effect = [
        grade_relevant,
        grade_relevant,
        grade_irrelevant,
        hallucination_ok,
    ]

    # generate_node uses a separate LLM instance (llm_generate); mock it too
    llm_generate_instance = MagicMock()
    llm_generate_instance.invoke.return_value = make_llm_response(
        "The answer is 42. [Source: file_1.pdf]"
    )

    # Both ChatGoogleGenerativeAI() calls return our mocks in order
    MockLLMClass.side_effect = [llm_instance, llm_generate_instance]

    # ── build and run the graph ───────────────────────────────────────────────
    from graph.graph import build_graph

    graph = build_graph(db)

    initial_state = {
        "question": "What is the answer?",
        "session_id": "sess-test",
        "query_id": "q-test",
        "retrieved_documents": [],
        "rewritten_question": None,
        "retry_count": 0,
        "regeneration_count": 0,
        "generation": None,
        "hallucination_detected": False,
        "confidence_score": 0.0,
        "audit_log": [],
        "web_search_needed": False,
    }

    final_state = graph.invoke(initial_state)

    # ── assertions ────────────────────────────────────────────────────────────

    # Generation must be non-empty
    assert final_state["generation"], "generation should be non-empty"
    assert "42" in final_state["generation"], "generation should contain mocked answer"

    # Audit log must have exactly 4 entries (one per node that ran)
    audit_steps = [entry["step"] for entry in final_state["audit_log"]]
    assert audit_steps == ["retrieve", "grade", "generate", "hallucination_check"], (
        f"Unexpected audit steps: {audit_steps}"
    )

    # 2 of 3 docs should be graded relevant
    relevant = [d for d in final_state["retrieved_documents"] if d.grade == "relevant"]
    assert len(relevant) == 2, f"Expected 2 relevant docs, got {len(relevant)}"

    # No hallucination detected
    assert not final_state["hallucination_detected"]
    assert final_state["confidence_score"] == pytest.approx(0.95)

    # No retries triggered
    assert final_state["retry_count"] == 0
    assert final_state["regeneration_count"] == 0
