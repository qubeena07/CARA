from typing import List


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping word-based chunks.
    chunk_size: target words per chunk
    overlap: words shared between consecutive chunks
    """
    # Clean the text
    text = " ".join(text.split())
    words = text.split()

    if len(words) == 0:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]

        # Only keep chunks with at least 30 words
        if len(chunk_words) >= 30:
            chunks.append(" ".join(chunk_words))

        # Move forward by (chunk_size - overlap) to create the overlap
        start += chunk_size - overlap

    return chunks
