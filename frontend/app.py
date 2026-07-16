"""
app.py
──────
Streamlit frontend for DocuMind — AI Research Assistant.

This file contains ZERO RAG logic. Its only job:
  - Show a chat UI to the user
  - Call FastAPI endpoints (upload / query / documents)
  - Display responses nicely

Run with:
    streamlit run app.py

Make sure uvicorn is already running on port 8000 before starting this.
"""

import requests
import streamlit as st

import os

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — Page config (must be the FIRST Streamlit call in the script)
# ══════════════════════════════════════════════════════════════════════════════

# st.set_page_config() controls the browser tab title, icon, and layout.
# layout="wide" uses the full browser width instead of a narrow centered column.
# This must come before any other st.* call — Streamlit enforces this strictly.
st.set_page_config(
    page_title="DocuMind — AI Research Assistant",
    page_icon="📄",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — Constants
# ══════════════════════════════════════════════════════════════════════════════

# The FastAPI backend URL. Since both run locally, this is always localhost:8000.
# When you deploy, you'd change this to your Railway/Render backend URL.
# API_BASE = "http://localhost:8000"
API_BASE = os.getenv("BACKEND_URL", "http://localhost:8000")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — Session state initialization
# ══════════════════════════════════════════════════════════════════════════════

# st.session_state is a dictionary that persists across Streamlit reruns.
# Think of it as the "memory" of your app for one browser session.
#
# We initialize keys here with a guard ("if key not in st.session_state")
# so we only set them on the FIRST run — not on every rerun.
# Without the guard, every interaction would reset the chat history.

if "messages" not in st.session_state:
    # messages is a list of dicts: [{"role": "user"/"assistant", "content": "..."}]
    # This is the standard format used by OpenAI, LangChain, and Streamlit's chat UI.
    st.session_state.messages = []

if "indexed_docs" not in st.session_state:
    # List of document metadata dicts returned from GET /documents
    st.session_state.indexed_docs = []

if "sources_history" not in st.session_state:
    # Parallel list to messages — stores source chunks for each assistant reply.
    # Index i in sources_history corresponds to the i-th assistant message.
    st.session_state.sources_history = []


# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — Helper functions (API calls)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_documents() -> list[dict]:
    """Call GET /documents and return the list of ingested docs."""
    try:
        response = requests.get(f"{API_BASE}/documents", timeout=5)
        response.raise_for_status()
        return response.json().get("documents", [])
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Is uvicorn running on port 8000?")
        return []
    except Exception as e:
        st.error(f"Failed to fetch documents: {e}")
        return []


def upload_pdf(file) -> dict | None:
    """
    Send a PDF file to POST /upload.

    `file` here is a Streamlit UploadedFile object — it behaves like a file handle.
    We wrap it in a tuple (filename, bytes, mime_type) which is how the requests
    library sends multipart/form-data (the same format as an HTML file input).
    """
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        response = requests.post(f"{API_BASE}/upload", files=files, timeout=300)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Is uvicorn running on port 8000?")
        return None
    except requests.exceptions.HTTPError as e:
        # HTTPError means the server responded but with a 4xx/5xx status.
        # We extract the detail message from FastAPI's standard error format.
        detail = e.response.json().get("detail", str(e))
        st.error(f"Upload failed: {detail}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def ask_question(question: str) -> dict | None:
    """Send a question to POST /query and return the answer + sources."""
    try:
        payload = {"question": question}
        response = requests.post(
            f"{API_BASE}/query",
            json=payload,   # sends as application/json (not form data)
            timeout=30,     # LLM calls can take a few seconds
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend.")
        return None
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e))
        st.error(f"Query failed: {detail}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION E — Sidebar
# ══════════════════════════════════════════════════════════════════════════════

# Everything inside `with st.sidebar:` renders in the left panel.
# The sidebar survives across reruns — it re-renders but keeps its state.

with st.sidebar:
    st.title("📄 DocuMind")
    st.caption("AI Research Assistant")
    st.divider()

    # ── Document uploader ─────────────────────────────────────────────────────
    st.subheader("Upload Documents")

    # st.file_uploader returns an UploadedFile object when a file is selected,
    # or None when nothing is uploaded yet.
    # accept_multiple_files=True returns a list instead of a single object.
    uploaded_files = st.file_uploader(
        label="Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to query against.",
    )

    # The "Index Documents" button triggers ingestion.
    # Streamlit buttons return True only on the rerun immediately after clicking.
    if st.button("⚡ Index Documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Please select at least one PDF first.")
        else:
            # st.progress() shows a progress bar. We update it manually.
            progress_bar = st.progress(0, text="Starting ingestion...")
            success_count = 0

            for i, file in enumerate(uploaded_files):
                progress_bar.progress(
                    (i + 1) / len(uploaded_files),
                    text=f"Indexing {file.name}...",
                )
                result = upload_pdf(file)
                if result:
                    success_count += 1
                    # Immediately refresh the indexed docs list
                    st.session_state.indexed_docs = fetch_documents()

            progress_bar.empty()  # removes the progress bar after completion

            if success_count == len(uploaded_files):
                st.success(f"Indexed {success_count} document(s)!")
            elif success_count > 0:
                st.warning(f"Indexed {success_count}/{len(uploaded_files)} documents.")
            # errors already shown inside upload_pdf()

    st.divider()

    # ── Indexed documents list ────────────────────────────────────────────────
    st.subheader("Indexed Documents")

    # Load documents on first run (session_state.indexed_docs starts empty)
    if not st.session_state.indexed_docs:
        st.session_state.indexed_docs = fetch_documents()

    if st.session_state.indexed_docs:
        for doc in st.session_state.indexed_docs:
            # st.expander creates a collapsible section
            with st.expander(f"📑 {doc['filename']}"):
                # We use columns for a neat two-column layout inside the expander
                col1, col2 = st.columns(2)
                col1.metric("Pages", doc.get("pages", "—"))
                col2.metric("Chunks", doc.get("chunks", "—"))
    else:
        st.info("No documents indexed yet.\nUpload a PDF above to get started.")

    st.divider()

    # ── Clear chat button ─────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources_history = []
        st.rerun()  # force a rerun so the cleared chat renders immediately

    st.divider()
    st.subheader("⚠️ Danger Zone")
    if st.button("🗑️ Flush All Documents", use_container_width=True, type="primary"):
        try:
            response = requests.delete(f"{API_BASE}/documents", timeout=10)
            response.raise_for_status()
            st.session_state.indexed_docs = []
            st.session_state.messages = []
            st.session_state.sources_history = []
            st.success("All documents cleared!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to clear: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION F — Main chat area
# ══════════════════════════════════════════════════════════════════════════════

st.title("Ask Your Documents")
st.caption("Upload PDFs in the sidebar, then ask questions below.")
st.divider()

# ── Render existing chat history ──────────────────────────────────────────────
# On every rerun, we redraw the full chat history from session_state.
# st.chat_message("user") renders a user bubble.
# st.chat_message("assistant") renders an assistant bubble with a bot icon.

assistant_message_index = 0  # tracks which assistant message we're on

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # For assistant messages, show the sources that were used
        if msg["role"] == "assistant":
            if assistant_message_index < len(st.session_state.sources_history):
                sources = st.session_state.sources_history[assistant_message_index]
                if sources:
                    with st.expander(f"📎 Sources ({len(sources)} chunks used)"):
                        for j, source in enumerate(sources, 1):
                            st.markdown(
                                f"**Source {j}** — `{source['filename']}`, "
                                f"Page {source['page']}"
                            )
                            # st.caption renders smaller, muted text
                            st.caption(source["content"] + "...")
                            if j < len(sources):
                                st.divider()
            assistant_message_index += 1


# ── Chat input ────────────────────────────────────────────────────────────────
# st.chat_input() renders a fixed-at-bottom text input (like ChatGPT).
# It returns the submitted string on the rerun after the user presses Enter,
# or None if nothing was submitted.

if question := st.chat_input("Ask a question about your documents..."):

    # Guard: make sure at least one document is indexed
    if not st.session_state.indexed_docs:
        st.warning("Please upload and index at least one PDF first.")
        st.stop()  # stops the rest of the script from running this rerun

    # 1. Add user message to history and render it immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 2. Call the API and render the assistant response
    with st.chat_message("assistant"):
        # st.spinner shows a "thinking" animation while waiting for the API
        with st.spinner("Searching documents and generating answer..."):
            result = ask_question(question)

        if result:
            answer = result["answer"]
            sources = result.get("sources", [])

            # Render the answer text
            # st.markdown renders **bold**, *italic*, code blocks, etc.
            st.markdown(answer)

            # Render the source chunks in a collapsible expander
            if sources:
                with st.expander(f"📎 Sources ({len(sources)} chunks used)"):
                    for j, source in enumerate(sources, 1):
                        st.markdown(
                            f"**Source {j}** — `{source['filename']}`, "
                            f"Page {source['page']}"
                        )
                        st.caption(source["content"] + "...")
                        if j < len(sources):
                            st.divider()

            # 3. Save to session state for persistence across future reruns
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.sources_history.append(sources)

        else:
            # API call failed — show error and add placeholder to keep indices aligned
            error_msg = "Sorry, I couldn't get an answer. Check that the backend is running."
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.session_state.sources_history.append([])