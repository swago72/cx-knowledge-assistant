from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import os
import re
import hashlib
from typing import List, Dict, Tuple, Optional


# ----------------------------
# IDs (avoid collisions)
# ----------------------------
def _safe_chunk_id(source: str, i: int) -> str:
    """
    Creates a stable, collision-resistant chunk id.
    - Uses sanitized filename + md5 hash of full source path/name.
    - Prevents collisions when filenames are similar or get truncated.
    """
    base = re.sub(r"[^a-zA-Z0-9_]+", "_", source)
    h = hashlib.md5(source.encode("utf-8")).hexdigest()[:10]
    return f"{base[:60]}_{h}_chunk_{i}"


# ----------------------------
# HTML loader
# ----------------------------
def load_html_files(folder_path: str = "data/html") -> List[Dict[str, str]]:
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"Folder not found: {folder_path}. "
            f"Create it and put .html files inside, or pass the correct folder_path."
        )

    documents = []
    files = sorted(os.listdir(folder_path))

    if not files:
        print(f"⚠ No files found in {folder_path}")
        return documents

    print(f"Scanning folder: {folder_path} ({len(files)} items)")

    for filename in files:
        if not filename.lower().endswith(".html"):
            continue

        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_html = f.read()

        soup = BeautifulSoup(raw_html, "lxml")

        # Remove noisy elements
        for tag in soup(
            ["script", "style", "nav", "header", "footer", "aside", "noscript", "iframe"]
        ):
            tag.decompose()

        clean_text = soup.get_text(separator="\n", strip=True)

        # Normalize whitespace / remove empty lines
        lines = [l.strip() for l in clean_text.splitlines() if l.strip()]
        clean_text = "\n".join(lines)

        if len(clean_text) < 200:
            print(f"  ⚠ Skipping {filename} — too short ({len(clean_text)} chars)")
            continue

        # Clean display name
        short_name = filename.replace(" - Google Play Help.html", "").replace(".html", "")

        documents.append(
            {
                "content": clean_text,
                "source": filename,        # keep original filename for traceability
                "short_name": short_name,  # display label for citations
                "chars": str(len(clean_text)),
            }
        )
        print(f"  ✓ {short_name} ({len(clean_text):,} chars)")

    print(f"\nTotal documents loaded: {len(documents)}")
    return documents


# ----------------------------
# Chunking
# ----------------------------
def chunk_documents(
    documents: List[Dict[str, str]],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict[str, object]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = []
    for doc in documents:
        texts = splitter.split_text(doc["content"])
        for i, text in enumerate(texts):
            chunks.append(
                {
                    "text": text,
                    "source": doc["source"],
                    "short_name": doc["short_name"],
                    "chunk_id": _safe_chunk_id(doc["source"], i),
                    "chunk_num": i,
                    "total": len(texts),
                }
            )

    print(f"Total chunks created: {len(chunks)}")
    return chunks


# ----------------------------
# Embedding + Chroma write
# ----------------------------
def embed_and_store(
    chunks: List[Dict[str, object]],
    collection_name: str = "cx_knowledge_base",
    chroma_path: str = "./chroma_db",
    reset_collection: bool = False,
    batch_size: int = 64,
) -> Tuple[object, SentenceTransformer]:
    """
    - reset_collection=True will wipe and rebuild the collection (useful for debugging)
    - otherwise it incrementally adds only new chunk_ids
    """
    print("Loading HuggingFace model (downloads once)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✓ Model ready\n")

    client = chromadb.PersistentClient(path=chroma_path)

    if reset_collection:
        try:
            client.delete_collection(name=collection_name)
            print(f"🧹 Deleted existing collection: {collection_name}")
        except Exception:
            pass  # collection may not exist

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Fast sanity print
    try:
        print(f"Chroma collection count (before): {collection.count()}")
    except Exception:
        pass

    # Identify existing ids (incremental ingest)
    existing = collection.get(include=[])
    existing_ids = set(existing.get("ids", [])) if existing else set()

    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
    if not new_chunks:
        print("Already embedded. Nothing to add.")
        print(f"✓ ChromaDB contains {collection.count()} total chunks")
        return collection, model

    print(f"Embedding {len(new_chunks)} new chunks (batch_size={batch_size})...")

    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        embeds = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            embeddings=embeds,
            documents=texts,
            metadatas=[
                {
                    "source": c["source"],
                    "short_name": c["short_name"],
                    "chunk_num": int(c["chunk_num"]),
                    "total_chunks_in_doc": int(c["total"]),
                }
                for c in batch
            ],
            ids=[c["chunk_id"] for c in batch],
        )

        done = min(i + batch_size, len(new_chunks))
        print(f"  Stored {done}/{len(new_chunks)}")

    print(f"\n✓ ChromaDB contains {collection.count()} total chunks")
    return collection, model


# ----------------------------
# Main
# ----------------------------
def main():
    print("=" * 55)
    print("  CX Knowledge Assistant — Ingestion Pipeline")
    print("=" * 55 + "\n")

    print("STEP 1: Loading HTML files...")
    documents = load_html_files(folder_path="data/html")

    if not documents:
        print("\n⚠ No documents loaded. Add .html files to data/html and rerun.")
        return

    print("\nSTEP 2: Chunking...")
    chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=50)

    print("\nSTEP 3: Embedding and storing in ChromaDB...")
    # Set reset_collection=True if you want a clean rebuild during debugging
    embed_and_store(
        chunks,
        collection_name="cx_knowledge_base",
        chroma_path="./chroma_db",
        reset_collection=False,
        batch_size=64,
    )

    print("\n✅ Done. Knowledge base is ready.")


if __name__ == "__main__":
    main()