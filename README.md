# DocuMind — AI Research Assistant

> Upload any PDF. Ask questions in plain English. Get answers with exact page citations.

**Live Demo →** [jgtesjgwlvs62gvmrhap6p.streamlit.app](https://jgtesjgwlvs62gvmrhap6p.streamlit.app)  
**API Docs →** [your-railway-url.up.railway.app/docs](https://your-railway-url.up.railway.app/docs)

---

## What it does

DocuMind is a Retrieval-Augmented Generation (RAG) application that lets users query PDF documents using natural language. Instead of scrolling through a 60-page report, users upload it and ask questions — the system retrieves the most relevant passages and generates a grounded answer with source citations.

**Example use cases:**
- Query a company's annual report before investing
- Extract key findings from a research paper
- Ask questions about a legal contract or policy document

---

## Architecture

```
User
 │
 ▼
Streamlit Frontend (Streamlit Cloud)
 │  POST /upload  ─────────────────────────────────────┐
 │  POST /query   ─────────────────────────────────┐   │
 │  GET  /documents ───────────────────────────┐   │   │
 ▼                                             │   │   │
FastAPI Backend (Railway)                      │   │   │
 │                                             │   │   │
 ├── PDF Loader (PyPDF)  ◄────────────────────────────┘
 ├── Chunker (RecursiveCharacterTextSplitter)  │   │
 ├── Embeddings (all-MiniLM-L6-v2)  ──────────┘   │
 ├── Vector Store (FAISS)                          │
 └── RAG Chain (LangChain + Groq LLaMA)  ─────────┘
      └── Returns: answer + source chunks + page numbers
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, free) |
| Vector store | FAISS (Facebook AI Similarity Search) |
| LLM | Groq API — `openai/gpt-oss-20b` (fast inference) |
| RAG framework | LangChain — RetrievalQA chain |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker + docker-compose |
| Deployment | Railway (backend) + Streamlit Cloud (frontend) |

---

## Key Design Decisions

**Why LangChain chains over agents?**  
For fixed-flow Document QA, chains are deterministic, faster, and cheaper. Agents add LLM decision-making overhead that's unnecessary when the retrieval path is always the same.

**Why FAISS over ChromaDB?**  
FAISS is production-grade and faster at scale. ChromaDB was prototyped first; FAISS was chosen for the final build for its performance and persistence characteristics.

**Why `all-MiniLM-L6-v2`?**  
22MB model, no API key required, good quality for semantic similarity at its size. Loads once at startup and stays in memory.

**Chunk size = 500, overlap = 50**  
Tested with 256, 500, and 1000 character chunks. 500 with 50 overlap balances retrieval precision (smaller chunks) with context preservation (overlap prevents sentence boundary cuts).

---

## Running Locally

**Prerequisites:** Python 3.11+, Docker Desktop, Groq API key ([console.groq.com](https://console.groq.com))

```bash
git clone https://github.com/your-username/documind-rag.git
cd documind-rag
```

Create `.env` in the project root:
```
GROQ_API_KEY=your_key_here
```

**Option 1 — Docker (recommended):**
```bash
docker-compose up --build
```
Open [localhost:8501](http://localhost:8501)

**Option 2 — Local dev (two terminals):**
```bash
# Terminal 1 — backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

```
documind-rag/
├── backend/
│   ├── main.py              # FastAPI endpoints
│   ├── rag_pipeline.py      # Core RAG logic
│   ├── config.py            # Constants (chunk size, model names)
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py               # Streamlit UI
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Known Limitations

- **Ephemeral storage on Railway free tier:** The FAISS index lives in the container filesystem and resets on server restart. Users need to re-upload PDFs after a restart. Production fix: persist the index to S3 or a cloud volume.
- **Single-PDF retrieval per query:** Answers draw from all indexed PDFs simultaneously. Cross-document reasoning (e.g. "compare doc A and doc B") works but may be less precise.
- **Scanned PDFs:** Only text-layer PDFs are supported. Image-based scanned PDFs require an OCR preprocessing step (not implemented).

---

## What's Next

- [ ] Cloud storage (S3) for persistent FAISS index
- [ ] Re-ranking retrieved chunks with a cross-encoder
- [ ] Hybrid search (dense FAISS + sparse BM25)
- [ ] RAGAS evaluation framework for answer quality metrics
- [ ] Streaming responses (token-by-token via FastAPI StreamingResponse)

---

*Built by Vedant Parikh — [LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)*
