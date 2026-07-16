import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent # path of curr file itself (base = backend folder)
FAISS_INDEX_PATH = str(BASE_DIR / "faiss_index") # directory made inside the backend folder
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE = 500        # max characters per chunk
CHUNK_OVERLAP = 50      # how many characters overlap between consecutive chunks

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K_CHUNKS = 5        # how many chunks to fetch per user query

# ── Models ─────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# GROQ_MODEL_NAME = "llama3-8b-8192" --> deprecated model
# GROQ_MODEL_NAME = "qwen/qwen3.6-27b" --> gives reasoning
GROQ_MODEL_NAME = "openai/gpt-oss-20b"

