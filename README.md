# CX Knowledge Assistant — Production-Safe RAG System

LLM-powered customer support assistant built with a production-oriented Retrieval-Augmented Generation (RAG) architecture.

This system ingests structured HTML knowledge base articles, builds a persistent vector store, retrieves semantically relevant content using embeddings, and generates strictly source-grounded responses with citation enforcement and fallback protection.

## Architecture Overview

## System Design Evolution

### V1 — Enterprise Modular Architecture
- FastAPI backend
- OpenAI GPT-3.5
- LangChain RAG
- SQLite → PostgreSQL upgrade path
- Full ingestion + analytics layer

Designed for enterprise CCaaS deployment.

[View Architecture v1](docs/01_mvp_enterprise_architecture.html)

---

### V2 — Simplified Cost-Optimized Stack
- Gemini 2.0 Flash (Free tier)
- HuggingFace local embeddings
- No FastAPI layer
- Fully Streamlit-based app
- Reduced infra complexity

Designed for fast deployment + zero-cost scalability.

[View Architecture v2](docs/02_cost_optimized_gemini_architecture.html)

## Ingestion Layer
- HTML cleaning and normalization (BeautifulSoup)
- Recursive chunking (500 token windows with overlap)
- SentenceTransformer embeddings (all-MiniLM-L6-v2)
- Persistent ChromaDB vector storage
## Retrieval Layer
- Cosine similarity search with distance diagnostics 
- Top-k retrieval with metadata tracking
- Debug metrics (distance, hit previews, chunk counts)
## Generation Layer
- Gemini 2.5 Pro (via LangChain)
- Strict source-only answering
- Citation enforcement
- Confidence-based fallback logic
## Analytics Layer (WIP)
- SQLite query logging
- Latency tracking
- Similarity scoring


## Stack
- Python 3.11
- LLM: Google Gemini (via LangChain)
- Sentence Transformers
- ChromaDB
- BeautifulSoup
- SQLite
- UI: Streamlit


## Setup
    pip install -r requirements.txt
    # Create .env Add GOOGLE_API_KEY
    python -m src/ingestion
    python -m src/run_one
    streamlit run app.py

## Status
- [x] Week 1: Ingestion pipeline
- [x] Week 2: RAG + Gemini
- [ ] Week 3: Analytics
- [ ] Week 4: Streamlit UI
- [ ] Week 5: Polish
- [ ] Week 6: Deploy
