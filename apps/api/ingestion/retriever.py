import os
from typing import List
from sqlalchemy.orm import Session
from sqlalchemy import text
import cohere
from dataclasses import dataclass

from .embedder import embed_texts
from database.models import Document


@dataclass
class RetrievedDocument:
    id: str
    content: str
    source: str
    score: float


def retrieve_and_rerank(
    question: str,
    db: Session,
    top_k_retrieve: int = 10,
    top_k_rerank: int = 4
) -> List[RetrievedDocument]:
    """
    Two-stage retrieval:
    1. Bi-encoder: embed question, cosine search in pgvector (fast, returns top 10)
    2. Cross-encoder: Cohere rerank those 10, keep top 4 (accurate)
    """

    # Stage 1: embed the question and do cosine similarity search
    query_embedding = embed_texts([question])[0]

    # pgvector <=> is cosine distance (lower = more similar)
    # 1 - distance = cosine similarity (higher = more similar)
    sql = text("""
        SELECT
            id::text,
            content,
            source,
            1 - (embedding <=> CAST(:embedding AS vector)) AS score
        FROM documents
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
    """)

    result = db.execute(sql, {
        "embedding": str(query_embedding),
        "limit": top_k_retrieve
    })
    rows = result.fetchall()

    if not rows:
        return []

    # Stage 2: rerank with Cohere cross-encoder
    cohere_client = cohere.ClientV2(os.getenv("COHERE_API_KEY", ""))

    try:
        rerank_response = cohere_client.rerank(
            model="rerank-english-v3.0",
            query=question,
            documents=[row.content for row in rows],
            top_n=top_k_rerank
        )

        reranked = []
        for r in rerank_response.results:
            row = rows[r.index]
            reranked.append(RetrievedDocument(
                id=row.id,
                content=row.content,
                source=row.source,
                score=r.relevance_score
            ))
        return reranked

    except Exception as e:
        # Cohere failed — fall back to cosine similarity top-k
        print(f"Cohere rerank failed: {e}. Falling back to cosine similarity.")
        return [
            RetrievedDocument(id=row.id, content=row.content, source=row.source, score=row.score)
            for row in rows[:top_k_rerank]
        ]
