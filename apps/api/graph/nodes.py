import json
import os
from datetime import datetime
from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

from .state import GraphState, GradedDocument
from .config import (
    FAST_LLM_MODEL, GENERATE_LLM_MODEL,
    TOP_K_RETRIEVE, TOP_K_RERANK
)
from .prompts import GRADE_PROMPT, REWRITE_PROMPT, GENERATE_PROMPT, HALLUCINATION_PROMPT
from ingestion.retriever import retrieve_and_rerank


# ── LLM instances (lazy) ─────────────────────────────────────────────────────
# Initialized on first use so the module can be imported without a key present.
# Once initialized, reused for all subsequent calls (no per-request overhead).
_llm_fast = None
_llm_generate = None


def get_llm_fast() -> ChatGoogleGenerativeAI:
    global _llm_fast
    if _llm_fast is None:
        # temperature=0: deterministic. Used for grading, rewriting, hallucination check.
        _llm_fast = ChatGoogleGenerativeAI(model=FAST_LLM_MODEL, temperature=0)
    return _llm_fast


def get_llm_generate() -> ChatGoogleGenerativeAI:
    global _llm_generate
    if _llm_generate is None:
        # temperature=0.3: slight creativity. Used for answer generation.
        _llm_generate = ChatGoogleGenerativeAI(model=GENERATE_LLM_MODEL, temperature=0.3)
    return _llm_generate


# ── Helper ───────────────────────────────────────────────────────────────────
def make_audit_entry(step: str, message: str, data: dict = None) -> dict:
    """Create a structured audit log entry. Every node calls this."""
    return {
        "step": step,
        "message": message,
        "data": data or {},
        "timestamp": datetime.utcnow().isoformat()
    }


def safe_json_parse(text: str, fallback: dict) -> dict:
    """Parse JSON from LLM output with multiple fallback strategies."""
    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: find JSON object in surrounding text
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        pass

    # Strategy 3: return fallback
    print(f"JSON parse failed for: {text[:200]}...")
    return fallback


# ── NODE 1: Retrieve ─────────────────────────────────────────────────────────
def retrieve_node(state: GraphState, db) -> dict:
    """
    Retrieves relevant document chunks from pgvector.
    Uses the rewritten question if available (after a rewrite loop),
    otherwise uses the original question.

    Why check rewritten_question first: after a failed retrieval, the graph
    rewrites the query and loops back here. We must use the new query,
    not the original one.
    """
    question = state.get("rewritten_question") or state["question"]

    docs = retrieve_and_rerank(
        question=question,
        db=db,
        top_k_retrieve=TOP_K_RETRIEVE,
        top_k_rerank=TOP_K_RERANK
    )

    graded_docs = [
        GradedDocument(
            id=d.id,
            content=d.content,
            source=d.source,
            relevance_score=d.score,
            grade="ungraded",
            grade_reason=""
        )
        for d in docs
    ]

    return {
        "retrieved_documents": graded_docs,
        "audit_log": [make_audit_entry(
            step="retrieve",
            message=f"Retrieved {len(graded_docs)} documents",
            data={
                "question_used": question,
                "is_rewritten": bool(state.get("rewritten_question")),
                "doc_count": len(graded_docs),
                "sources": list(set(d.source for d in graded_docs))
            }
        )]
    }


# ── NODE 2: Grade documents ───────────────────────────────────────────────────
def grade_documents_node(state: GraphState) -> dict:
    """
    Grades each retrieved document as relevant or irrelevant.
    Uses chain-of-thought prompting for accuracy.

    Why grade each document separately: a batch grading call would have the
    model grade all documents at once, which causes cross-contamination — the
    model's judgment of document 1 influences its judgment of document 2.
    Separate calls are slower but more accurate.
    """
    docs = state["retrieved_documents"]
    question = state.get("rewritten_question") or state["question"]

    graded_docs = []
    grade_results = []

    for doc in docs:
        response = get_llm_fast().invoke([
            SystemMessage(content=GRADE_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nDocument chunk:\n{doc.content}")
        ])

        parsed = safe_json_parse(
            response.content,
            fallback={"grade": "irrelevant", "reason": "parse error", "reasoning": ""}
        )

        grade = parsed.get("grade", "irrelevant")
        reason = parsed.get("reason", "")
        reasoning = parsed.get("reasoning", "")

        graded_doc = GradedDocument(
            id=doc.id,
            content=doc.content,
            source=doc.source,
            relevance_score=doc.relevance_score,
            grade=grade,
            grade_reason=reason
        )
        graded_docs.append(graded_doc)
        grade_results.append({
            "source": doc.source,
            "grade": grade,
            "reason": reason,
            "reasoning": reasoning
        })

    relevant_count = sum(1 for d in graded_docs if d.grade == "relevant")

    return {
        "retrieved_documents": graded_docs,
        "audit_log": [make_audit_entry(
            step="grade",
            message=f"{relevant_count} of {len(graded_docs)} documents graded relevant",
            data={
                "grades": grade_results,
                "relevant_count": relevant_count,
                "total_count": len(graded_docs)
            }
        )]
    }


# ── NODE 3: Rewrite query ─────────────────────────────────────────────────────
def rewrite_query_node(state: GraphState) -> dict:
    """
    Rewrites the user's question into retrieval-optimized form.
    Called when all retrieved documents were graded irrelevant.

    Why increment retry_count here: the counter tracks how many full
    rewrite+retrieve cycles have happened. The routing logic uses this
    to stop looping after MAX_RETRIES.
    """
    original_question = state["question"]
    retry_count = state.get("retry_count", 0)

    response = get_llm_fast().invoke([
        SystemMessage(content=REWRITE_PROMPT),
        HumanMessage(content=f"Original question: {original_question}")
    ])

    rewritten = response.content.strip().strip('"').strip("'")

    return {
        "rewritten_question": rewritten,
        "retry_count": retry_count + 1,
        "audit_log": [make_audit_entry(
            step="rewrite",
            message=f"Query rewritten (attempt {retry_count + 1})",
            data={
                "original": original_question,
                "rewritten": rewritten,
                "attempt": retry_count + 1
            }
        )]
    }


# ── NODE 4: Generate answer ───────────────────────────────────────────────────
def generate_node(state: GraphState) -> dict:
    """
    Generates the answer from relevant documents.

    Why filter to only relevant documents: irrelevant documents add noise.
    Including them in the prompt increases the chance the LLM picks up
    irrelevant information and weaves it into the answer.

    Why include source attribution in the context: the generate prompt
    instructs the model to cite sources inline. For that to work, the model
    needs to know which text came from which file.
    """
    question = state.get("rewritten_question") or state["question"]
    relevant_docs = [d for d in state["retrieved_documents"] if d.grade == "relevant"]
    regeneration_count = state.get("regeneration_count", 0)

    if not relevant_docs:
        return {
            "generation": "I could not find relevant information in the provided documents to answer this question. Please try rephrasing your question or upload documents that cover this topic.",
            "audit_log": [make_audit_entry(
                step="generate",
                message="No relevant documents found — returned inability response",
                data={"relevant_doc_count": 0}
            )]
        }

    # Build the context string with source attribution
    context_parts = []
    for doc in relevant_docs:
        context_parts.append(f"[Source: {doc.source}]\n{doc.content}")
    context = "\n\n---\n\n".join(context_parts)

    regen_instruction = ""
    if regeneration_count > 0:
        regen_instruction = f"\n\nIMPORTANT: Your previous answer contained claims not supported by the sources. This is attempt {regeneration_count + 1}. Stay strictly within the provided sources. Do not add any information not explicitly stated in the sources."

    response = get_llm_generate().invoke([
        SystemMessage(content=GENERATE_PROMPT + regen_instruction),
        HumanMessage(content=f"Question: {question}\n\nSource documents:\n\n{context}")
    ])

    return {
        "generation": response.content,
        "audit_log": [make_audit_entry(
            step="generate",
            message=f"Answer generated from {len(relevant_docs)} relevant documents" +
                    (f" (regeneration attempt {regeneration_count})" if regeneration_count > 0 else ""),
            data={
                "relevant_doc_count": len(relevant_docs),
                "sources_used": [d.source for d in relevant_docs],
                "answer_length": len(response.content),
                "is_regeneration": regeneration_count > 0
            }
        )]
    }


# ── NODE 5: Hallucination check ───────────────────────────────────────────────
def hallucination_check_node(state: GraphState) -> dict:
    """
    Verifies the generated answer is grounded in the retrieved sources.

    Why this is a separate LLM call (not part of the generate call):
    The generation model is "in character" as a helpful assistant — it tends
    to rationalize its own outputs. A separate call with a different system
    prompt (auditor role) is more likely to catch issues because it approaches
    the answer critically rather than defensively.
    """
    generation = state.get("generation", "")
    relevant_docs = [d for d in state["retrieved_documents"] if d.grade == "relevant"]
    regen_count = state.get("regeneration_count", 0)

    if not relevant_docs or not generation:
        return {
            "hallucination_detected": False,
            "confidence_score": 0.5,
            "audit_log": [make_audit_entry(
                step="hallucination_check",
                message="Skipped — no relevant docs or no generation to check",
                data={}
            )]
        }

    context = "\n\n".join([d.content for d in relevant_docs])

    response = get_llm_fast().invoke([
        SystemMessage(content=HALLUCINATION_PROMPT),
        HumanMessage(content=f"Source documents:\n{context}\n\nGenerated answer:\n{generation}")
    ])

    parsed = safe_json_parse(
        response.content,
        fallback={
            "hallucinating": False,
            "confidence": 0.5,
            "explanation": "parse error — accepting answer",
            "claims_checked": []
        }
    )

    hallucinating = parsed.get("hallucinating", False)
    confidence = float(parsed.get("confidence", 0.5))
    explanation = parsed.get("explanation", "")

    return {
        "hallucination_detected": hallucinating,
        "confidence_score": confidence,
        "regeneration_count": regen_count + (1 if hallucinating else 0),
        "audit_log": [make_audit_entry(
            step="hallucination_check",
            message="Hallucination detected — will regenerate" if hallucinating else "Answer is grounded in sources",
            data={
                "hallucinating": hallucinating,
                "confidence": confidence,
                "explanation": explanation,
                "claims_checked": parsed.get("claims_checked", []),
                "is_final_check": regen_count >= 1
            }
        )]
    }
