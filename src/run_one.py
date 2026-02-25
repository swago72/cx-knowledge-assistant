from src.rag import get_chroma_collection, ask, ChatMemory
from sentence_transformers import SentenceTransformer

def main():
    collection = get_chroma_collection("./chroma_db", "cx_knowledge_base")

    # 1) Chroma count
    print("Chroma chunk count:", collection.count())

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    memory = ChatMemory(max_turns=5)

    question = "How do I request a refund on Google Play?"

    # Turn on debug to print retrieval diagnostics
    answer, hits = ask(
        question=question,
        collection=collection,
        embed_model=embed_model,
        memory=memory,
        k=8,
        debug=True,          # <-- important
        max_distance=0.55,   # <-- gate
    )

    print("\nANSWER:\n", answer)

    # 2) Top 3 hit lines (title + distance + preview)
    print("\nTOP 3 HITS:")
    for h in hits[:3]:
        title = h.get("short_name") or h.get("source")
        dist = h.get("distance")
        preview = (h.get("text") or "").replace("\n", " ")[:180]
        print(f"- {title} | dist: {dist} | preview: {preview}")

if __name__ == "__main__":
    main()