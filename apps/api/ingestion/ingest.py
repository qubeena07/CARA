from typing import List
from .chunker import chunk_text
from .embedder import embed_texts


def ingest_document(content: str, source: str) -> List[dict]:
    """Chunk, embed, and store a document. Returns list of stored chunk records."""
    chunks = chunk_text(content)
    embeddings = embed_texts(chunks)
    # TODO: store in pgvector
    records = [
        {"source": source, "content": chunk, "embedding": emb}
        for chunk, emb in zip(chunks, embeddings)
    ]
    return records
