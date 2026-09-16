# All settings in one place: paths, model names, and search parameters.

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / 'document'

# File types the loader will read.
SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md'}

# How we cut pages into passages.
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Models.
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'                       # local, 384-dim
RERANK_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'      # local cross-encoder
GROQ_MODEL_NAME = 'llama-3.3-70b-versatile'                     # Groq-hosted LLM

# Qdrant collection name (like a table name for our vectors).
COLLECTION_NAME = 'nodejs_docs'

# Retrieval: pull a wide pool cheaply, then let the reranker pick the best few.
RETRIEVE_POOL = 20   # candidates from hybrid search
FINAL_TOP_K = 5      # passages kept after reranking
RRF_K = 60           # Reciprocal Rank Fusion constant
