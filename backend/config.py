import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent # path of curr file itself (base = backend folder)
# Every visitor gets their own folder: sessions/<session-id>/{uploads, faiss_index, documents.json}
SESSIONS_DIR = BASE_DIR / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

# ── Abuse limits (the app is public, so keep it cheap to run) ──────────────────
MAX_UPLOAD_MB = 25          # reject PDFs bigger than this
SESSION_TTL_HOURS = 24      # a visitor's data is deleted this long after their last upload

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

