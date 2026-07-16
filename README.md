# DocuMind — AI Research Assistant

**Live demo:** [https://your-streamlit-url.streamlit.app ](https://jgtesjgwlvs62gvmrhap6p.streamlit.app/) 
**API docs:** [https://your-railway-url.up.railway.app/docs](https://documind-rag-production-a027.up.railway.app/docs)

RAG-powered document Q&A. Upload any PDF, ask questions in plain English, 
get answers with exact page citations.
      
## Stack
- **Embeddings:** all-MiniLM-L6-v2 (HuggingFace)
- **Vector store:** FAISS
- **LLM:** Groq API (LLaMA 3)
- **Backend:** FastAPI + Uvicorn
- **Frontend:** Streamlit
- **Containerized:** Docker + docker-compose
