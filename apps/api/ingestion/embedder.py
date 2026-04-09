from sentence_transformers import SentenceTransformer
from typing import List

# Load once at module level — avoids reloading on every call
# First run downloads the model (~420MB), cached after that
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("Loading embedding model (first time only)...")
        _model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts using the sentence transformer model.
    Returns a list of 768-dimensional float vectors.
    normalize_embeddings=True makes cosine similarity = dot product.
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=len(texts) > 10,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return embeddings.tolist()
