from typing import List
from .embedder import embed_texts


def retrieve_documents(query: str, top_k: int = 5) -> List[dict]:
    """Retrieve top_k documents from the vector store for the given query."""
    # TODO: implement pgvector similarity search
    query_embedding = embed_texts([query])[0]
    return []
