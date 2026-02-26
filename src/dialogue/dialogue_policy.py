def decide_state(
    intent_confidence: float,
    retrieval_confidence: float,
    fallback_used: bool
) -> tuple[str, str]:
    
    if fallback_used:
        return "HANDOFF", "low_retrieval_confidence"

    if intent_confidence < 0.5:
        return "CLARIFY", "low_intent_confidence"

    return "ANSWER", "sufficient_confidence"