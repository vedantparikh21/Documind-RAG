"""
rag_pipeline.py
───────────────
Core RAG logic. This file does three things:
  1. Ingest a PDF  →  chunks → embeddings → FAISS index saved to disk
  2. Build a RAG chain  →  retriever + LLM wired together
  3. Answer a question  →  retrieve chunks → pass to LLM → return answer + sources
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from config import (
    FAISS_INDEX_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_CHUNKS,
    EMBEDDING_MODEL_NAME,
    GROQ_MODEL_NAME,
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — Embedding model (loaded once, reused everywhere)
# ══════════════════════════════════════════════════════════════════════════════

# This is intentionally at module level (outside any function).
# Loading a sentence-transformer model takes ~1-2 seconds.
# If it were inside a function, it would reload on every call — very slow.
# Module-level means it loads once when Python imports this file, then stays
# in memory for the entire life of your application.

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True},  # makes all vectors length=1
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — Prompt template
# ══════════════════════════════════════════════════════════════════════════════

# A prompt template is the instruction you give the LLM before every query.
# {context} and {input} are placeholders that LangChain fills in at runtime:
#   - {context} = the retrieved chunks from your PDF (the "library pages")
#   - {input}   = the user's actual question

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about uploaded documents.\n"
    "Use ONLY the context provided below to answer. Do not use outside knowledge.\n"
    "If the answer is not in the context, say: "
    "'I could not find this information in the uploaded documents.'\n"
    "Be concise. Always mention which document or section you found the answer in.\n\n"
    "Context:\n{context}"
)

# # Replace your SYSTEM_PROMPT with this:
# SYSTEM_PROMPT = (
#     "You are a helpful assistant that answers questions about uploaded documents.\n"
#     "Base your answer primarily on the context provided below.\n"
#     "If the context contains relevant information, use it to give a complete answer.\n"
#     "If the answer genuinely cannot be found in the context, say: "
#     "'I could not find this information in the uploaded documents.'\n"
#     "Be concise and mention the document/section your answer draws from.\n\n"
#     "Context:\n{context}"
# )

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — FAISS vector store helpers
# ══════════════════════════════════════════════════════════════════════════════

# def load_vector_store() -> FAISS | None:
#     """
#     Load the FAISS index from disk.
#     Returns the FAISS object if it exists, or None if no PDFs have been ingested yet.
#     """
#     if Path(FAISS_INDEX_PATH).exists():
#         return FAISS.load_local(
#             FAISS_INDEX_PATH,
#             embeddings,
#             allow_dangerous_deserialization=True,
#             # ^ This sounds scary but is safe here. It means:
#             # "trust the pickle file we're loading."
#             # We saved it ourselves, so we know it's safe.
#         )
#     return None
def load_vector_store() -> FAISS | None:
    """
    Load the FAISS index from disk.
    Returns None if no index has been created yet.
    """

    index_file = Path(FAISS_INDEX_PATH) / "index.faiss"

    if not index_file.exists():
        print("No FAISS index found. Creating a new one.")
        return None

    print("Loading existing FAISS index...")

    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def save_vector_store(store: FAISS) -> None:
    """Persist the FAISS index to disk so it survives app restarts."""
    store.save_local(FAISS_INDEX_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — PDF ingestion
# ══════════════════════════════════════════════════════════════════════════════

def ingest_pdf(pdf_path: str) -> dict:
    """
    Full ingestion pipeline for one PDF file.

    Steps:
      1. Load PDF  →  one Document object per page
      2. Split     →  many small chunk Documents
      3. Embed     →  convert chunks to vectors
      4. Store     →  add vectors to FAISS index, save to disk

    Args:
        pdf_path: absolute or relative path to the PDF file

    Returns:
        dict with filename, page count, and chunk count
    """

    # ── Step 1: Load PDF ──────────────────────────────────────────────────────
    # PyPDFLoader reads the PDF and returns a list of Document objects.
    # Each Document = one page of the PDF.
    # Each Document has two things:
    #   .page_content  →  the raw text of that page (string)
    #   .metadata      →  {"source": "/path/to/file.pdf", "page": 0}
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()

    # ── Step 2: Split into chunks ─────────────────────────────────────────────
    # RecursiveCharacterTextSplitter tries to split on natural boundaries first:
    # it tries "\n\n" (paragraphs) first, then "\n" (lines), then " " (words),
    # then individual characters as a last resort.
    # This keeps sentences intact much better than a naive fixed-size split.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,  # adds {"start_index": N} to metadata, useful for debugging
    )
    chunks = splitter.split_documents(raw_docs)
    # chunks is now a list of Documents — many more than raw_docs.
    # Each chunk has metadata inherited from its parent page +
    # the start_index we just added.

    # ── Step 3: Tag each chunk with the filename ──────────────────────────────
    # FAISS stores vectors + metadata. We add "filename" so later, when we
    # retrieve a chunk, we know which PDF it came from.
    filename = Path(pdf_path).name
    for chunk in chunks:
        chunk.metadata["filename"] = filename

    # ── Step 4: Embed and store ───────────────────────────────────────────────
    # FAISS.from_documents() does two things in one call:
    #   a) Calls embeddings.embed_documents() on every chunk → list of float vectors
    #   b) Builds a FAISS index from those vectors
    # If a FAISS index already exists on disk (from a previous PDF), we load it
    # and ADD to it rather than overwriting it. This is how multi-PDF works.
    existing_store = load_vector_store()

    if existing_store is not None:
        # Add new chunks to the existing index
        existing_store.add_documents(chunks)
        save_vector_store(existing_store)
    else:
        # First PDF ever — create a brand new index
        new_store = FAISS.from_documents(chunks, embeddings)
        save_vector_store(new_store)

    return {
        "filename": filename,
        "pages": len(raw_docs),
        "chunks": len(chunks),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION E — RAG chain builder
# ══════════════════════════════════════════════════════════════════════════════

def build_rag_chain(groq_api_key: str):
    """
    Assemble and return the RAG chain.
    This must be called after at least one PDF has been ingested.

    The chain works like this:
      user question
          ↓
      retriever.get_relevant_documents(question)  →  top-3 chunks
          ↓
      prompt template fills {context} and {input}
          ↓
      LLM generates answer
          ↓
      returns {"answer": "...", "context": [chunk1, chunk2, chunk3]}
    """

    # Load the vector store from disk
    store = load_vector_store()
    if store is None:
        raise ValueError(
            "No documents ingested yet. "
            "Call ingest_pdf() with at least one PDF first."
        )

    # Retriever: a wrapper around the vector store that does similarity search.
    # When you call retriever.invoke("some question"), it:
    #   1. Embeds the question into a vector
    #   2. Does cosine similarity against all stored chunk vectors
    #   3. Returns the top-k most similar chunks
    retriever = store.as_retriever(
        search_type="similarity",        # plain cosine similarity
        search_kwargs={"k": TOP_K_CHUNKS},
    )

    # LLM: ChatGroq with temperature=0
    # temperature=0 means the model always picks the most probable next token.
    # No randomness. This is what you want for factual Q&A.
    # temperature=0.7 would give more creative/varied responses — wrong for RAG.
    llm = ChatGroq(
        model=GROQ_MODEL_NAME,
        groq_api_key=groq_api_key,
        temperature=0,
        max_tokens=1024,
    )

    # create_stuff_documents_chain:
    # "Stuff" is LangChain's name for the simplest document combination strategy.
    # It takes all retrieved chunks and STUFFS them all into one prompt as {context}.
    # Simple. Works well for 3-5 chunks. Would break if you had 50 chunks
    # (context window overflow) — but TOP_K=3 keeps us safe.
    document_chain = create_stuff_documents_chain(llm, prompt)

    # create_retrieval_chain:
    # Wires the retriever and document_chain together into one callable object.
    # When you call rag_chain.invoke({"input": "..."}), it automatically:
    #   1. Runs the retriever on your input
    #   2. Passes retrieved docs + input into document_chain
    #   3. Returns the final answer
    rag_chain = create_retrieval_chain(retriever, document_chain)

    return rag_chain


# ══════════════════════════════════════════════════════════════════════════════
# SECTION F — Public query function
# ══════════════════════════════════════════════════════════════════════════════

def query_documents(question: str, groq_api_key: str) -> dict:
    """
    Answer a question using the ingested documents.

    Args:
        question:     the user's natural language question
        groq_api_key: your Groq API key

    Returns:
        {
          "answer":  "...",           # LLM-generated answer
          "sources": [                # chunks that were retrieved
            {
              "filename": "report.pdf",
              "page":     3,
              "content":  "first 300 chars of chunk..."
            },
            ...
          ]
        }
    """
    chain = build_rag_chain(groq_api_key)

    # .invoke() runs the full chain synchronously (blocking)
    # result has two keys:
    #   "answer"  →  the LLM's response string
    #   "context" →  list of Document objects that were retrieved
    result = chain.invoke({"input": question})

    # Parse source metadata from retrieved Document objects
    sources = []
    for doc in result.get("context", []):
        sources.append({
            "filename": doc.metadata.get("filename", "unknown"),
            "page":     doc.metadata.get("page", "?"),
            "content":  doc.page_content[:300],
        })

    return {
        "answer":  result["answer"],
        "sources": sources,
    }