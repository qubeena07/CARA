import json
import asyncio
import uuid
import tempfile
import os
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text, func, cast, Float

from database.connection import get_db
from database.models import Session as DBSession, Query, AuditLog
from ingestion.ingest import ingest_file
from graph.graph import build_graph
from graph.state import GraphState

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None


def format_sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_FILE_SIZE_MB = 50


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a PDF, .txt, or .md file for ingestion into the vector database.
    Runs chunking + embedding synchronously (acceptable for research use;
    swap to a background task queue for production scale).
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read into memory, check size before writing to disk
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max is {MAX_FILE_SIZE_MB} MB.",
        )

    source_name = file.filename or f"upload{ext}"

    # Write to a temp file so ingest_file (which uses file path) can read it
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        chunk_count = ingest_file(tmp_path, source_name, db)
    finally:
        os.unlink(tmp_path)

    return {
        "source": source_name,
        "chunks_inserted": chunk_count,
        "size_mb": round(size_mb, 2),
    }


@app.post("/ask")
async def ask(request: AskRequest, db: Session = Depends(get_db)):
    """
    Main RAG endpoint. Streams SSE events as each LangGraph node completes.

    Stream protocol:
      data: {"type": "start", ...}
      data: {"type": "retrieve", ...}    ← after retrieve node
      data: {"type": "grade", ...}       ← after grade node
      data: {"type": "rewrite", ...}     ← if rewrite triggered
      data: {"type": "generate", ...}    ← after generate node
      data: {"type": "hallucination_check", ...}
      data: {"type": "final", "data": {"answer": ..., "sources": [...], ...}}
      data: {"type": "error", "message": ...}  ← on exception
    """

    async def event_generator():
        # Resolve session — create DB row if this is a new session
        session_id = request.session_id or str(uuid.uuid4())
        query_id = str(uuid.uuid4())

        if not request.session_id:
            # New session — persist the Session row so Query FK is valid
            db.add(DBSession(id=session_id))
            db.commit()

        yield format_sse({
            "type": "start",
            "message": "Processing your question...",
            "session_id": session_id,
            "query_id": query_id,
        })

        initial_state: GraphState = {
            "question": request.question,
            "session_id": session_id,
            "query_id": query_id,
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

        final_state = None

        try:
            graph = build_graph(db)

            # stream_mode="values" yields the FULL accumulated state after each node.
            # The last yielded value is the final state — no second graph.invoke() needed.
            seen_audit_count = 0
            for state in graph.stream(initial_state, stream_mode="values"):
                final_state = state  # track last — this becomes final_state after loop

                # audit_log accumulates via operator.add; new entries are at the end
                audit_log = state.get("audit_log", [])
                new_entries = audit_log[seen_audit_count:]
                seen_audit_count = len(audit_log)

                for entry in new_entries:
                    yield format_sse({
                        "type": entry["step"],
                        "message": entry["message"],
                        "data": entry["data"],
                        "timestamp": entry["timestamp"],
                    })
                    await asyncio.sleep(0)  # yield control so events flush immediately

            if final_state is None:
                yield format_sse({"type": "error", "message": "Graph produced no output."})
                return

            yield format_sse({
                "type": "final",
                "message": "Complete",
                "data": {
                    "answer": final_state.get("generation") or "Unable to answer.",
                    "confidence": final_state.get("confidence_score", 0.0),
                    "retry_count": final_state.get("retry_count", 0),
                    "hallucination_detected": final_state.get("hallucination_detected", False),
                    "sources": [
                        {"source": d.source, "grade": d.grade}
                        for d in final_state.get("retrieved_documents", [])
                    ],
                },
            })

            # Persist query + audit trail
            db.add(Query(
                id=query_id,
                session_id=session_id,
                original_question=request.question,
                rewritten_question=final_state.get("rewritten_question"),
                final_answer=final_state.get("generation"),
                confidence_score=final_state.get("confidence_score", 0.0),
                retry_count=final_state.get("retry_count", 0),
                hallucination_detected=final_state.get("hallucination_detected", False),
                regeneration_count=final_state.get("regeneration_count", 0),
            ))
            for entry in final_state.get("audit_log", []):
                db.add(AuditLog(
                    query_id=query_id,
                    step_type=entry["step"],
                    message=entry["message"],
                    data=entry["data"],
                ))
            db.commit()

        except Exception as e:
            yield format_sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


@app.get("/analytics")
async def get_analytics(db: Session = Depends(get_db)):
    """
    Aggregated analytics for the dashboard.

    All queries scoped to last 30 days except recent_queries (last 20 total).
    Uses raw SQL for window aggregates that SQLAlchemy ORM makes verbose.
    """
    since = datetime.now(timezone.utc) - timedelta(days=30)

    # ── Summary stats ──────────────────────────────────────────────────────────
    stats_row = db.execute(text("""
        SELECT
            COUNT(*)                                        AS total_queries,
            COALESCE(AVG(confidence_score), 0)              AS avg_confidence,
            COALESCE(
                SUM(CASE WHEN hallucination_detected THEN 1 ELSE 0 END)::float
                / NULLIF(COUNT(*), 0), 0
            )                                               AS hallucination_rate,
            COALESCE(AVG(retry_count), 0)                   AS avg_retries,
            COALESCE(
                SUM(CASE WHEN retry_count > 0 THEN 1 ELSE 0 END)::float
                / NULLIF(COUNT(*), 0), 0
            )                                               AS retry_rate
        FROM queries
        WHERE created_at >= :since
    """), {"since": since}).fetchone()

    summary = {
        "total_queries":      int(stats_row.total_queries),
        "avg_confidence":     round(float(stats_row.avg_confidence), 3),
        "hallucination_rate": round(float(stats_row.hallucination_rate) * 100, 1),
        "avg_retries":        round(float(stats_row.avg_retries), 2),
        "retry_rate":         round(float(stats_row.retry_rate) * 100, 1),
    }

    # ── Daily query volume (last 30 days) ──────────────────────────────────────
    daily_rows = db.execute(text("""
        SELECT
            DATE(created_at AT TIME ZONE 'UTC') AS day,
            COUNT(*)                             AS count
        FROM queries
        WHERE created_at >= :since
        GROUP BY DATE(created_at AT TIME ZONE 'UTC')
        ORDER BY day
    """), {"since": since}).fetchall()

    daily_volume = [
        {"date": str(r.day), "queries": int(r.count)}
        for r in daily_rows
    ]

    # ── Confidence score distribution (10 buckets: 0.0–0.1, …, 0.9–1.0) ──────
    bucket_rows = db.execute(text("""
        SELECT
            FLOOR(confidence_score * 10) / 10.0  AS bucket,
            COUNT(*)                              AS count
        FROM queries
        WHERE created_at >= :since
        GROUP BY bucket
        ORDER BY bucket
    """), {"since": since}).fetchall()

    # Fill all 10 buckets even if count=0
    bucket_map = {round(float(r.bucket), 1): int(r.count) for r in bucket_rows}
    confidence_dist = [
        {"range": f"{x:.1f}–{x+0.1:.1f}", "count": bucket_map.get(round(x, 1), 0)}
        for x in [i / 10 for i in range(10)]
    ]

    # ── Retry distribution (0 / 1 / 2+ rewrites) ──────────────────────────────
    retry_rows = db.execute(text("""
        SELECT
            CASE
                WHEN retry_count = 0 THEN '0 rewrites'
                WHEN retry_count = 1 THEN '1 rewrite'
                ELSE '2+ rewrites'
            END AS label,
            COUNT(*) AS count
        FROM queries
        WHERE created_at >= :since
        GROUP BY label
        ORDER BY label
    """), {"since": since}).fetchall()

    retry_dist = [{"label": r.label, "value": int(r.count)} for r in retry_rows]

    # ── Top 10 source documents ────────────────────────────────────────────────
    # audit_logs.data is JSONB; each "retrieve" step stores {"sources": [...]}
    # We unnest the sources array and count appearances.
    source_rows = db.execute(text("""
        SELECT
            src,
            COUNT(*) AS query_count
        FROM (
            SELECT jsonb_array_elements_text(data->'sources') AS src
            FROM audit_logs
            WHERE step_type = 'retrieve'
        ) unnested
        GROUP BY src
        ORDER BY query_count DESC
        LIMIT 10
    """)).fetchall()

    top_sources = [
        {"source": r.src, "query_count": int(r.query_count)}
        for r in source_rows
    ]

    # ── Recent queries (last 20) ───────────────────────────────────────────────
    recent_rows = db.execute(text("""
        SELECT
            id,
            original_question,
            confidence_score,
            retry_count,
            hallucination_detected,
            created_at
        FROM queries
        ORDER BY created_at DESC
        LIMIT 20
    """)).fetchall()

    recent_queries = [
        {
            "id":                   str(r.id),
            "question":             r.original_question,
            "confidence":           round(float(r.confidence_score or 0), 3),
            "retry_count":          int(r.retry_count or 0),
            "hallucination":        bool(r.hallucination_detected),
            "created_at":           r.created_at.isoformat() if r.created_at else None,
        }
        for r in recent_rows
    ]

    return {
        "summary":          summary,
        "daily_volume":     daily_volume,
        "confidence_dist":  confidence_dist,
        "retry_dist":       retry_dist,
        "top_sources":      top_sources,
        "recent_queries":   recent_queries,
    }


@app.get("/sessions/{session_id}/queries")
async def get_session_queries(session_id: str, db: Session = Depends(get_db)):
    """Query history for a session."""
    return db.query(Query).filter(Query.session_id == session_id).all()


@app.get("/queries/{query_id}/audit")
async def get_audit_log(query_id: str, db: Session = Depends(get_db)):
    """Full audit trail for a query."""
    return (
        db.query(AuditLog)
        .filter(AuditLog.query_id == query_id)
        .order_by(AuditLog.created_at)
        .all()
    )
