import os
import time
from typing import List, Dict, Any, Tuple

import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

from src.analytics import log_query, NO_CONFIDENT_ANSWER


# ----------------------------
# Memory (last N turns)
# ----------------------------
class ChatMemory:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.turns: List[Dict[str, str]] = []

    def add_user(self, text: str):
        self.turns.append({"role": "user", "text": text})
        self.turns = self.turns[-2 * self.max_turns :]

    def add_assistant(self, text: str):
        self.turns.append({"role": "assistant", "text": text})
        self.turns = self.turns[-2 * self.max_turns :]

    def format_for_prompt(self) -> str:
        if not self.turns:
            return ""
        lines = []
        for t in self.turns:
            prefix = "User" if t["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {t['text']}")
        return "\n".join(lines)


# ----------------------------
# Chroma collection loader
# ----------------------------
def get_chroma_collection(
    chroma_path: str = "./chroma_db",
    collection_name: str = "cx_knowledge_base",
):
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ----------------------------
# Retriever
# ----------------------------
def retrieve(
    query: str,
    collection,
    embed_model,
    k: int = 5,
) -> List[Dict[str, Any]]:
    q_emb = embed_model.encode([query]).tolist()[0]

    result = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs = result["documents"][0] if result.get("documents") else []
    metas = result["metadatas"][0] if result.get("metadatas") else []
    dists = result["distances"][0] if result.get("distances") else []

    hits = []
    for text, meta, dist in zip(docs, metas, dists):
        hits.append(
            {
                "text": text,
                "source": meta.get("source", ""),
                "short_name": meta.get("short_name", meta.get("source", "")),
                "chunk_num": meta.get("chunk_num", None),
                "distance": float(dist) if dist is not None else None,  # cosine distance (lower=better)
            }
        )
    return hits


# ----------------------------
# Prompt builder
# ----------------------------
def build_prompt(question: str, hits: List[Dict[str, Any]], memory_text: str = "") -> str:
    context_blocks = []
    for i, h in enumerate(hits, start=1):
        title = h["short_name"] or h["source"] or f"Doc {i}"
        context_blocks.append(f"[SOURCE {i}] Title: {title}\nText:\n{h['text']}")
    context = "\n\n".join(context_blocks)

    rules = """
You are a customer support knowledge assistant.
Answer ONLY using the provided sources. If the answer is not in the sources, say:
"I couldn't find a confident answer in the knowledge base."

Citations rule (strict):
- Every factual statement must end with a citation like: (Source: <Title>)
- Use the Title exactly as shown in the SOURCE blocks.
- If multiple sources support a statement, cite the most relevant one.

Output format:
Answer:
<your answer>

Sources Used:
- <Title 1>
- <Title 2>
"""

    memory_section = ""
    if memory_text.strip():
        memory_section = f"Conversation so far:\n{memory_text}\n\n"

    return f"""{rules}

{memory_section}User Question: {question}

SOURCES:
{context}
"""


# ----------------------------
# Generator (Gemini)
# ----------------------------
def generate_answer(
    question: str,
    hits: List[Dict[str, Any]],
    memory_text: str = "",
    model_name: str = "models/gemini-2.5-pro",
    temperature: float = 0.2,
) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY in environment.")

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
    )

    prompt = build_prompt(question, hits, memory_text=memory_text)
    resp = llm.invoke(prompt)
    return resp.content


# ----------------------------
# Orchestrator
# ----------------------------
def ask(
    question: str,
    collection,
    embed_model,
    memory: ChatMemory = None,
    k: int = 5,
    model_name: str = "models/gemini-2.5-pro",
    temperature: float = 0.2,
    # NEW: confidence gate + debug controls
    max_distance: float = 0.55,  # cosine distance threshold (lower=better)
    min_hits: int = 1,
    debug: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns: (answer, hits)

    - Uses a deterministic confidence gate:
      If no hits or top_distance > max_distance => returns NO_CONFIDENT_ANSWER without calling the LLM.
    - Adds debug printing to see what retrieval is doing.
    - Still logs to SQLite via log_query.
    """
    if memory is None:
        memory = ChatMemory(max_turns=5)

    start = time.perf_counter()
    fallback_used = False

    memory.add_user(question)

    # Quick sanity: is your collection empty?
    try:
        collection_count = int(collection.count())
    except Exception:
        collection_count = None

    hits = retrieve(question, collection, embed_model, k=k)

    # Distances from hits (cosine distance: lower is better)
    dists = [h["distance"] for h in hits if h.get("distance") is not None]
    top_distance = float(min(dists)) if dists else None
    avg_distance = float(sum(dists) / len(dists)) if dists else None

    # Sources from hits (dedupe, preserve order)
    sources_used = []
    for h in hits:
        title = h.get("short_name") or h.get("source")
        if title:
            sources_used.append(str(title))
    sources_used = list(dict.fromkeys(sources_used))

    # DEBUG: print retrieval diagnostics
    if debug:
        print("\n[DEBUG] Collection count:", collection_count)
        print("[DEBUG] Question:", question)
        print("[DEBUG] num_hits:", len(hits), "k:", k)
        print("[DEBUG] top_distance:", top_distance, "| avg_distance:", avg_distance)
        for idx, h in enumerate(hits[:5], start=1):
            preview = (h.get("text") or "").replace("\n", " ")[:180]
            print(f"[DEBUG] Hit {idx}: {h.get('short_name')} | dist={h.get('distance')}")
            print(f"        preview: {preview}")

    # NEW: deterministic confidence gate BEFORE calling the LLM
    gate_reason = None
    if collection_count == 0:
        fallback_used = True
        answer = NO_CONFIDENT_ANSWER
        gate_reason = "empty_collection"
    elif len(hits) < min_hits:
        fallback_used = True
        answer = NO_CONFIDENT_ANSWER
        gate_reason = "too_few_hits"
    elif top_distance is None or top_distance > max_distance:
        fallback_used = True
        answer = NO_CONFIDENT_ANSWER
        gate_reason = "low_retrieval_confidence"
    else:
        # Generate answer (LLM)
        try:
            answer = generate_answer(
                question,
                hits,
                memory_text=memory.format_for_prompt(),
                model_name=model_name,
                temperature=temperature,
            )
        except Exception:
            fallback_used = True
            answer = NO_CONFIDENT_ANSWER
            gate_reason = "llm_exception"

    latency_ms = int((time.perf_counter() - start) * 1000)

    # Log to SQLite
    log_query(
        user_question=question,
        answer=answer,
        sources_used=sources_used,
        top_distance=top_distance,
        avg_distance=avg_distance,
        model_name=model_name,
        latency_ms=latency_ms,
        fallback_used=fallback_used,
        extra={
            "k": k,
            "temperature": temperature,
            "num_hits": len(hits),
            "max_distance": max_distance,
            "min_hits": min_hits,
            "collection_count": collection_count,
            "gate_reason": gate_reason,
        },
    )

    memory.add_assistant(answer)
    return answer, hits