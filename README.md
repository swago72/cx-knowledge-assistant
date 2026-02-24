# CX Knowledge Assistant

An AI-powered RAG chatbot that lets support agents query documentation using plain English and get cited answers.

## Stack
- LLM: Google Gemini (via LangChain)
- Embeddings: HuggingFace all-MiniLM-L6-v2
- Vector DB: ChromaDB
- Orchestration: LangChain
- UI: Streamlit

## Setup
    pip install -r requirements.txt
    # Add GOOGLE_API_KEY to .env
    python src/ingestion.py
    streamlit run app.py

## Status
- [x] Week 1: Ingestion pipeline
- [ ] Week 2: RAG + Gemini
- [ ] Week 3: Analytics
- [ ] Week 4: Streamlit UI
- [ ] Week 5: Polish
- [ ] Week 6: Deploy
