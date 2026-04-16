# Cara

A RAG (Retrieval-Augmented Generation) research assistant with a self-correcting LangGraph pipeline, SSE streaming, and an analytics dashboard.

## Architecture

```
apps/
  api/          FastAPI backend (Python)
    graph/      LangGraph pipeline nodes + routing
    ingestion/  PDF/txt/md → chunk → embed → pgvector
    database/   SQLAlchemy models + Alembic migrations
  web/          Next.js 16 frontend
    app/        Chat UI + analytics dashboard
    components/ AuditTrail, ChatMessage, UploadButton
    hooks/      useRAGStream (SSE state management)
```

### Pipeline

```
question
  → retrieve (pgvector cosine search → Cohere rerank)
  → grade documents (Gemini: relevant / irrelevant per doc)
  → [no relevant docs?] rewrite query → retrieve again (max 2x)
  → generate answer (Gemini, relevant docs only)
  → hallucination check (Gemini auditor)
  → [hallucination?] regenerate (max 2x)
  → stream final answer via SSE
```

Every step is logged to `audit_logs` for full traceability.

## Prerequisites

- Python 3.12+
- Node.js 20+
- PostgreSQL 16 with the `pgvector` extension
- [Google AI API key](https://aistudio.google.com/) (Gemini 2.5 Flash)
- [Cohere API key](https://cohere.com/) (reranker)

## Setup

### 1. Start the database

```bash
docker compose up -d postgres
```

This runs pgvector/pgvector:pg16 on port **5433** (avoids conflicts with local Postgres).

### 2. Configure the API

```bash
cd apps/api
cp .env.example .env   # if it exists, otherwise create:
```

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/cara
GOOGLE_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
```

### 3. Install API dependencies

```bash
cd apps/api
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run database migrations

```bash
cd apps/api
venv/bin/alembic upgrade head
```

### 5. Ingest documents

```bash
# Ingest the sample papers included in the repo
venv/bin/python scripts/ingest_docs.py --dir ../../sample_docs

# Or ingest your own files
venv/bin/python scripts/ingest_docs.py --file paper.pdf
venv/bin/python scripts/ingest_docs.py --dir /path/to/docs
```

Supported formats: `.pdf`, `.txt`, `.md`

First run downloads the embedding model (~420 MB, cached after).

### 6. Start the API

```bash
cd apps/api
venv/bin/uvicorn main:app --reload --port 8000
```

Verify: `curl http://localhost:8000/health`

### 7. Install and start the web app

```bash
# From repo root
npm install
npm run dev:web
```

Open [http://localhost:3000](http://localhost:3000).

---

## Usage

### Chat interface (`/`)

- Type a question and press **Enter** to send
- Left panel shows the live reasoning trace (retrieve → grade → generate → hallucinate check)
- Each assistant message shows confidence score, source files, and a "View reasoning" link
- Use **Upload doc** in the header to add new documents without restarting the server

### Analytics dashboard (`/analytics`)

- Summary stats: total queries, avg confidence, hallucination rate, avg rewrites
- Query volume chart (last 30 days)
- Confidence score distribution (calibration view)
- Rewrite distribution pie chart (retrieval quality)
- Top source documents by retrieval count
- Recent query log with per-row confidence badges

---

## Development

### Run tests

```bash
cd apps/api
venv/bin/python -m pytest tests/test_graph.py -v
```

All LLM calls and DB are mocked — no API keys or database needed.

### Project scripts

```bash
# From repo root
npm run dev:web     # start Next.js dev server
npm run dev:api     # start FastAPI with reload
```

### Configuration knobs (`apps/api/graph/config.py`)

| Variable | Default | Effect |
|---|---|---|
| `TOP_K_RETRIEVE` | 10 | Chunks fetched from pgvector before reranking |
| `TOP_K_RERANK` | 4 | Chunks kept after Cohere rerank |
| `MAX_RETRIES` | 2 | Max query rewrites if no relevant docs found |
| `MAX_REGENERATIONS` | 2 | Max regenerations if hallucination detected |
| `MIN_RELEVANT_DOCS` | 1 | Minimum relevant docs required to attempt generation |
| `EMBEDDING_MODEL` | `multi-qa-mpnet-base-dot-v1` | Must match `Vector(768)` in schema |

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/ask` | SSE streaming RAG query |
| `POST` | `/upload` | Upload and ingest a document |
| `GET` | `/analytics` | Aggregated dashboard data |
| `GET` | `/sessions/{id}/queries` | Query history for a session |
| `GET` | `/queries/{id}/audit` | Full audit trail for a query |

### SSE event types (`POST /ask`)

```
start               → session_id, query_id assigned
retrieve            → N documents fetched
grade               → relevance grades per document
rewrite             → query rewritten (only if no relevant docs)
generate            → answer produced
hallucination_check → grounding verified
final               → { answer, confidence, sources, ... }
error               → { message }
```
