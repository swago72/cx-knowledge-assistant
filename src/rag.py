import time
from typing import List, Dict, Any

import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import get_settings
from src.analytics import log_query, NO_CONFIDENT_ANSWER
from src.dialogue.intent_router import detect_intent
from src.dialogue.dialogue_policy import decide_state
from src.policy_messages import build_policy_message


# ============================================================
# Memory (last N turns)
# ============================================================

class ChatMemory:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.turns: List[Dict[str, str]] = []

    def add_user(self, text: str):
        self.turns.append({"role": "user", "text": text})
        self.turns = self.turns[-2 * self.max_turns:]

    def add_assistant(self, text: str):
        self.turns.append({"role": "assistant", "text": text})
        self.turns = self.turns[-2 * self.max_turns:]

    def format_for_prompt(self) -> str:
        if not self.turns:
            return ""
        lines = []
        for t in self.turns:
            prefix = "User" if t["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {t['text']}")
        return "\n".join(lines)


# ============================================================
# Chroma Loader
# ============================================================

def get_chroma_collection(
    chroma_path: str = "./chroma_db",
    collection_name: str = "cx_knowledge_base",
):
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# ============================================================
# Retriever
# ============================================================

def retrieve(query: str, collection, embed_model, k: int = 5):
    q_emb = embed_model.encode([query]).tolist()[0]

    result = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]

    hits = []
    for text, meta, dist in zip(docs, metas, dists):
        hits.append(
            {
                "text": text,
                "source": meta.get("source", ""),
                "short_name": meta.get("short_name", meta.get("source", "")),
                "chunk_num": meta.get("chunk_num"),
                "distance": float(dist) if dist is not None else None,
            }
        )
    return hits


# ============================================================
# Prompt Builder
# ============================================================

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


# ============================================================
# Generator
# ============================================================

def generate_answer(
    question: str,
    hits: List[Dict[str, Any]],
    memory_text: str = "",
    model_name: str = "models/gemini-2.5-pro",
    temperature: float = 0.2,
) -> str:
    settings = get_settings()

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=settings.google_api_key,
    )

    prompt = build_prompt(question, hits, memory_text)
    resp = llm.invoke(prompt)
    return resp.content


# ============================================================
# Orchestrator
# ============================================================

def ask(
    question: str,
    collection,
    embed_model,
    memory: ChatMemory = None,
    k: int = 5,
    model_name: str = "models/gemini-2.5-pro",
    temperature: float = 0.2,
    max_distance: float = 0.55,
    min_hits: int = 1,
    debug: bool = False,
):
    if memory is None:
        memory = ChatMemory(max_turns=5)

    start = time.perf_counter()
    memory.add_user(question)

    # ----------------------------
    # Retrieval
    # ----------------------------
    hits = retrieve(question, collection, embed_model, k=k)

    dists = [h["distance"] for h in hits if h.get("distance") is not None]
    top_distance = float(min(dists)) if dists else None
    avg_distance = float(sum(dists) / len(dists)) if dists else None

    retrieval_confidence = 1 - top_distance if top_distance is not None else 0.0

    # ----------------------------
    # Intent Detection
    # ----------------------------
    intent, intent_confidence = detect_intent(question)

    # ----------------------------
    # Gate + Dialogue Decision
    # ----------------------------
    fallback_used = False
    gate_reason = None

    if (not hits) or (len(hits) < min_hits):
        state, reason = "HANDOFF", "too_few_hits"
        fallback_used = True
        gate_reason = "too_few_hits"
    elif top_distance is None or top_distance > max_distance:
        state, reason = "HANDOFF", "low_retrieval_confidence"
        fallback_used = True
        gate_reason = "low_retrieval_confidence"
    else:
        # decide_state uses BOTH intent_confidence and retrieval_confidence now
        state, reason = decide_state(
            intent_confidence=float(intent_confidence),
            retrieval_confidence=float(retrieval_confidence),
            fallback_used=False,
        )

    # ----------------------------
    # Execute State (visible behavior)
    # ----------------------------
    policy_message = build_policy_message(state, intent, reason)

    if state == "ANSWER":
        try:
            answer = generate_answer(
                question,
                hits,
                memory.format_for_prompt(),
                model_name=model_name,
                temperature=temperature,
            )
        except Exception:
            answer = NO_CONFIDENT_ANSWER
            fallback_used = True
            state = "HANDOFF"
            reason = "llm_exception"
            policy_message = build_policy_message(state, intent, reason)
    else:
        # CLARIFY / HANDOFF: show the policy message instead of a generic string
        answer = policy_message or (NO_CONFIDENT_ANSWER if state == "HANDOFF" else "Could you clarify your request?")

    latency_ms = int((time.perf_counter() - start) * 1000)

    # ----------------------------
    # Logging
    # ----------------------------
    sources_used = []
    for h in hits:
        t = h.get("short_name") or h.get("source")
        if t:
            sources_used.append(str(t))
    sources_used = list(dict.fromkeys(sources_used))  # dedupe, preserve order

    if debug:
        print("\n[DEBUG] intent:", intent, "intent_conf:", intent_confidence)
        print("[DEBUG] state:", state, "reason:", reason, "gate_reason:", gate_reason)
        print("[DEBUG] top_distance:", top_distance, "retrieval_conf:", retrieval_confidence)

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
            "min_hits": min_hits,
            "max_distance": max_distance,
            "num_hits": len(hits),
            "gate_reason": gate_reason,
            "intent": intent,
            "intent_confidence": float(intent_confidence),
            "retrieval_confidence": float(retrieval_confidence),
            "state": state,
            "decision_reason": reason,
            "policy_message": policy_message,
        },
    )

    memory.add_assistant(answer)

    return {
        "answer": answer,
        "hits": hits,
        "state": state,
        "intent": intent,
        "reason": reason,
    }