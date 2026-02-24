
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import os
import re

def _safe_chunk_id(source: str, i: int) -> str:
    base = re.sub(r"[^a-zA-Z0-9_]+", "_", source)
    return f"{base[:80]}_chunk_{i}"

def load_html_files(folder_path="data/html"):
    documents = []
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".html"):
            continue

        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_html = f.read()

        soup = BeautifulSoup(raw_html, "lxml")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript", "iframe"]):
            tag.decompose()

        clean_text = soup.get_text(separator="\\n", strip=True)
        lines = [l for l in clean_text.splitlines() if l.strip()]
        clean_text = "\\n".join(lines)

        if len(clean_text) < 100:
            print(f"  ⚠ Skipping {filename} — too short")
            continue

        short_name = filename.replace(" - Google Play Help.html", "")

        documents.append({
            "content": clean_text,
            "source": filename,
            "short_name": short_name,
            "chars": len(clean_text)
        })
        print(f"  ✓ {short_name} ({len(clean_text):,} chars)")

    print(f"\\nTotal documents loaded: {len(documents)}")
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\\n\\n", "\\n", ".", " "]
    )

    chunks = []
    for doc in documents:
        texts = splitter.split_text(doc["content"])
        for i, text in enumerate(texts):
            chunks.append({
                "text": text,
                "source": doc["source"],
                "short_name": doc["short_name"],
                "chunk_id": _safe_chunk_id(doc["source"], i),
                "chunk_num": i,
                "total": len(texts),
            })

    print(f"Total chunks created: {len(chunks)}")
    return chunks

def embed_and_store(chunks, collection_name="cx_knowledge_base"):
    print("Loading HuggingFace model (downloads once)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✓ Model ready\\n")

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    existing = collection.get(include=[])
    existing_ids = set(existing["ids"]) if existing and "ids" in existing else set()

    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
    if not new_chunks:
        print("Already embedded. Nothing to add.")
        print(f"✓ ChromaDB contains {collection.count()} total chunks")
        return collection, model

    print(f"Embedding {len(new_chunks)} chunks...")
    batch_size = 50

    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        embeds = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            embeddings=embeds,
            documents=[c["text"] for c in batch],
            metadatas=[{
                "source": c["source"],
                "short_name": c["short_name"],
                "chunk_num": c["chunk_num"]
            } for c in batch],
            ids=[c["chunk_id"] for c in batch]
        )

        done = min(i + batch_size, len(new_chunks))
        print(f"  Stored {done}/{len(new_chunks)}")

    print(f"\\n✓ ChromaDB contains {collection.count()} total chunks")
    return collection, model

def main():
    print("=" * 55)
    print("  CX Knowledge Assistant — Ingestion Pipeline")
    print("=" * 55 + "\\n")

    print("STEP 1: Loading HTML files...")
    documents = load_html_files()

    print("\\nSTEP 2: Chunking...")
    chunks = chunk_documents(documents)

    print("\\nSTEP 3: Embedding and storing in ChromaDB...")
    embed_and_store(chunks)

    print("\\n✅ Done. Knowledge base is ready for Week 2.")

if __name__ == "__main__":
    main()
