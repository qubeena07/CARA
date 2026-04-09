import operator
from typing import TypedDict, Annotated, Optional, List
from dataclasses import dataclass, field


@dataclass
class GradedDocument:
    """A retrieved document with its relevance grade."""
    id: str
    content: str
    source: str
    relevance_score: float      # from Cohere reranker (0.0 to 1.0)
    grade: str = "ungraded"    # "relevant", "irrelevant", "ungraded"
    grade_reason: str = ""     # chain-of-thought explanation from the grader


class GraphState(TypedDict):
    """
    The shared state passed between all LangGraph nodes.
    Every field here is readable and writable by every node.

    Fields marked Annotated[list, operator.add] ACCUMULATE (append).
    All other fields REPLACE on update.
    """

    # ── Input ──────────────────────────────────────────────
    question: str              # the user's original question (never modified)
    session_id: str
    query_id: str

    # ── Retrieval ───────────────────────────────────────────
    retrieved_documents: List[GradedDocument]   # starts empty, filled by retrieve node
    rewritten_question: Optional[str]           # filled by rewrite node if triggered

    # ── Loop counters ───────────────────────────────────────
    retry_count: int           # how many rewrite+retrieve loops have happened
    regeneration_count: int    # how many times we've regenerated the answer

    # ── Generation ──────────────────────────────────────────
    generation: Optional[str]           # the LLM's answer text
    hallucination_detected: bool        # True if hallucination check found problems
    confidence_score: float             # 0.0–1.0 from hallucination checker

    # ── Audit trail ─────────────────────────────────────────
    # Annotated with operator.add means updates APPEND to this list, not replace it
    # Each node adds its own entry. The full list is the complete reasoning trace.
    audit_log: Annotated[List[dict], operator.add]

    # ── Flags ───────────────────────────────────────────────
    web_search_needed: bool    # True if all retrieval attempts failed (no relevant docs)
