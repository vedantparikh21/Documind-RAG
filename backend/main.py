"""
main.py
───────
FastAPI application — the HTTP wrapper around rag_pipeline.py.

This file does NOT contain any RAG logic.
Its only job is to:
  - Accept HTTP requests from the Streamlit frontend (or any client)
  - Call the right function from rag_pipeline.py
  - Return well-structured JSON responses
  - Handle errors gracefully so the frontend always gets a useful message

Run with:
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000/docs to see the auto-generated Swagger UI.
"""

import os
import json
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_pipeline import ingest_pdf, query_documents
from config import UPLOADS_DIR, FAISS_INDEX_PATH

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — App setup
# ══════════════════════════════════════════════════════════════════════════════

# load_dotenv() reads the .env file in the current directory and puts all
# key=value pairs into os.environ. This is how we avoid hardcoding secrets.
# Must be called BEFORE any os.getenv() calls.
load_dotenv()

# Create the FastAPI app instance.
# title and version appear in the auto-generated /docs Swagger page.
app = FastAPI(
    title="DocuMind API",
    description="RAG-powered document Q&A backend",
    version="1.0.0",
)

# CORS = Cross-Origin Resource Sharing.
# Browsers block requests from one origin (e.g. localhost:8501 = Streamlit)
# to a different origin (localhost:8000 = FastAPI) by default.
# This middleware tells the browser: "it's fine, allow all origins."
# allow_origins=["*"] is fine for development. For production, you'd
# restrict it to your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],   # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],   # Content-Type, Authorization, etc.
)

# Read the Groq API key from environment (set in .env file).
# We read it once at startup rather than on every request.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — Document registry
# ══════════════════════════════════════════════════════════════════════════════

# We need to remember which PDFs have been ingested so the frontend
# can display a list of uploaded documents.
# For now we use a simple JSON file — no database needed.
# This is fine for a portfolio project. Production would use PostgreSQL.

REGISTRY_PATH = Path(__file__).resolve().parent / "documents.json"

def load_registry() -> list[dict]:
    """Load the list of ingested documents from disk."""
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return []

def save_registry(docs: list[dict]) -> None:
    """Persist the document list to disk."""
    REGISTRY_PATH.write_text(json.dumps(docs, indent=2))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — Pydantic models (request/response shapes)
# ══════════════════════════════════════════════════════════════════════════════

# Pydantic models define the shape of JSON coming IN (request body)
# and going OUT (response body). FastAPI uses these for:
#   1. Automatic validation — if the client sends wrong types, FastAPI
#      rejects the request with a clear error before your code even runs.
#   2. Auto-documentation — the /docs page shows exactly what JSON to send.
#   3. Type hints — your IDE knows what fields exist.

class QueryRequest(BaseModel):
    """Body of POST /query requests."""
    question: str    # the user's natural language question

class SourceChunk(BaseModel):
    """One retrieved document chunk, returned alongside the answer."""
    filename: str    # which PDF this came from
    page: int | str  # page number (int, or "?" if unknown)
    content: str     # first 300 chars of the chunk text

class QueryResponse(BaseModel):
    """Body of POST /query responses."""
    answer: str               # the LLM's answer
    sources: list[SourceChunk]  # the chunks used to generate it


# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — Endpoints
# ══════════════════════════════════════════════════════════════════════════════

# ── Health check ──────────────────────────────────────────────────────────────
# Every production API has a /health endpoint.
# Load balancers, Docker health checks, and monitoring tools ping this
# to know if the service is alive. Returns instantly with no business logic.

@app.get("/health")
def health_check():
    """Quick check that the API is running."""
    return {
        "status": "ok",
        "groq_key_loaded": GROQ_API_KEY is not None,
    }


# ── List documents ────────────────────────────────────────────────────────────
# The frontend calls this on startup to show the user which PDFs are
# already indexed (so they don't have to re-upload on every session).

@app.get("/documents")
def list_documents():
    """Return the list of all ingested documents."""
    return {"documents": load_registry()}


# ── Upload & ingest a PDF ─────────────────────────────────────────────────────
# UploadFile is FastAPI's type for multipart/form-data file uploads.
# The client sends the PDF as binary data in the request body.

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accept a PDF upload, save it, ingest into FAISS, update registry.

    Steps:
      1. Validate it's actually a PDF
      2. Save the raw bytes to the uploads/ folder
      3. Call ingest_pdf() from rag_pipeline.py
      4. Record the document in documents.json
      5. Return ingestion metadata
    """

    # Validate file type by checking the filename extension.
    # A more robust check would read the file's magic bytes (first 4 bytes = "%PDF"),
    # but extension check is fine for a portfolio project.
    if not file.filename.lower().endswith(".pdf"):
        # HTTPException tells FastAPI to return an HTTP error response.
        # 400 = Bad Request (client sent something wrong).
        # The detail string becomes the error message in the JSON response.
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Build the save path and write the uploaded bytes to disk.
    # await file.read() reads ALL bytes from the upload.
    # For very large files (100MB+) you'd use streaming chunks — fine to skip for now.
    save_path = UPLOADS_DIR / file.filename
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Run the ingestion pipeline from rag_pipeline.py.
    # Wrapped in try/except so upload errors return a clean JSON error
    # instead of a raw Python traceback.
    # try:
    #     result = ingest_pdf(str(save_path))
    # except Exception as e:
    #     # Clean up the saved file if ingestion failed
    #     save_path.unlink(missing_ok=True)
    #     # 500 = Internal Server Error (something broke on our side)
    #     raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    import traceback

    try:
        result = ingest_pdf(str(save_path))
    except Exception as e:
        traceback.print_exc()          # <-- add this
        print(f"ERROR: {e}")           # <-- add this

        save_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )

    # Update the document registry.
    # Check for duplicates by filename so re-uploading the same PDF
    # doesn't add it to the list twice.
    registry = load_registry()
    existing_names = [doc["filename"] for doc in registry]
    if result["filename"] not in existing_names:
        registry.append(result)
        save_registry(registry)

    # FastAPI automatically converts this dict to a JSON response with status 200.
    return {
        "message": f"Successfully ingested '{result['filename']}'",
        "filename": result["filename"],
        "pages": result["pages"],
        "chunks": result["chunks"],
    }

# @app.delete("/documents")
# def clear_all_documents():
#     """Delete the FAISS index and document registry — full reset."""
#     import shutil

#     # Delete FAISS index folder
#     faiss_path = Path(FAISS_INDEX_PATH)
#     if faiss_path.exists():
#         shutil.rmtree(faiss_path)

#     # Delete uploaded PDFs
#     if UPLOADS_DIR.exists():
#         shutil.rmtree(UPLOADS_DIR)
#         UPLOADS_DIR.mkdir(exist_ok=True)

#     # Clear the registry
#     save_registry([])

#     return {"message": "All documents cleared. You can now upload fresh PDFs."}
@app.delete("/documents")
def clear_all_documents():
    """Delete all indexed documents and uploaded PDFs."""

    import shutil

    # Clear FAISS files
    faiss_path = Path(FAISS_INDEX_PATH)
    if faiss_path.exists():
        for item in faiss_path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    # Clear uploaded PDFs
    if UPLOADS_DIR.exists():
        for item in UPLOADS_DIR.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    # Reset registry
    save_registry([])

    return {
        "message": "All documents cleared successfully."
    }

# ── Query the documents ───────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Answer a natural language question using the ingested documents.

    The response_model=QueryResponse tells FastAPI to:
      1. Validate our return value matches QueryResponse's shape
      2. Document this exact response format in /docs
    """

    # Basic input validation.
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Confirm the API key is available before making an expensive call.
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not set. Add it to your .env file.",
        )

    try:
        result = query_documents(req.question, GROQ_API_KEY)
    except ValueError as e:
        # ValueError from rag_pipeline means no documents ingested yet.
        # This is a client error (they should upload first), so 400 not 500.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION E — Entry point (for running directly with `python main.py`)
# ══════════════════════════════════════════════════════════════════════════════

# When you run `uvicorn main:app --reload`, uvicorn imports this file and
# uses the `app` object directly — this block doesn't run.
# But if you run `python main.py` directly, this starts uvicorn programmatically.
# Useful for debugging inside an IDE.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)