# ============================================================
# GRAPH CONFIGURATION
# All magic numbers live here. Change these to tune the system.
# Each will become a variable in the paper's ablation study.
# ============================================================

# Retrieval
TOP_K_RETRIEVE = 10        # chunks retrieved from pgvector before reranking
TOP_K_RERANK = 4           # chunks kept after Cohere reranking

# Self-correction loop budgets
MAX_RETRIES = 2            # max query rewrites before giving up on retrieval
MAX_REGENERATIONS = 2      # max answer regenerations if hallucination detected
MIN_RELEVANT_DOCS = 1      # minimum relevant docs needed to attempt generation

# Confidence thresholds
LOW_CONFIDENCE_THRESHOLD = 0.4  # below this, we note the answer is uncertain

# Embedding model — must match Vector(768) in the schema
EMBEDDING_MODEL = "multi-qa-mpnet-base-dot-v1"
EMBEDDING_DIM = 768

# LLM models
FAST_LLM_MODEL = "gemini-2.0-flash"         # for grading, rewriting, hallucination check
GENERATE_LLM_MODEL = "gemini-2.0-flash"     # for answer generation (same model, but streaming=True)
