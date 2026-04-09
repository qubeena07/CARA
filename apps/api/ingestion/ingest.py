import os
import uuid
from typing import List
from sqlalchemy.orm import Session
from pypdf import PdfReader

from .chunker import chunk_text
from .embedder import embed_texts
from database.models import Document


def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def extract_text_from_file(file_path: str) -> str:
    """Extract text from PDF, txt, or md files."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def ingest_file(file_path: str, source_name: str, db: Session) -> int:
    """
    Ingest a single file into the vector database.
    Returns the number of chunks inserted.
    """
    print(f"Ingesting: {source_name}")

    # 1. Extract text
    text = extract_text_from_file(file_path)
    if not text.strip():
        print(f"  WARNING: No text extracted from {source_name}")
        return 0

    # 2. Chunk
    chunks = chunk_text(text)
    print(f"  Split into {len(chunks)} chunks")

    if not chunks:
        return 0

    # 3. Embed all chunks in one batch (much faster than one at a time)
    print(f"  Embedding {len(chunks)} chunks...")
    embeddings = embed_texts(chunks)

    # 4. Insert into database
    doc_records = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        doc = Document(
            id=uuid.uuid4(),
            content=chunk,
            source=source_name,
            chunk_index=i,
            doc_metadata={"file_path": file_path, "chunk_index": i},
            embedding=embedding
        )
        doc_records.append(doc)

    db.bulk_save_objects(doc_records)
    db.commit()

    print(f"  Inserted {len(doc_records)} chunks for {source_name}")
    return len(doc_records)
