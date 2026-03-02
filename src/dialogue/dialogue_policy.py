def decide_state(
    intent_confidence: float,
    retrieval_confidence: float,
    fallback_used: bool,
) -> tuple[str, str]:

    # Hard failure from earlier gate
    if fallback_used:
        return "HANDOFF", "retrieval_gate_failure"

    # Very weak retrieval → escalate
    if retrieval_confidence < 0.4:
        return "HANDOFF", "very_low_retrieval_confidence"

    # Medium retrieval but unclear intent → clarify
    if intent_confidence < 0.5:
        return "CLARIFY", "low_intent_confidence"

    # Medium retrieval zone → cautious clarify
    if 0.4 <= retrieval_confidence < 0.6:
        return "CLARIFY", "borderline_retrieval_confidence"

    # Strong signals
    return "ANSWER", "sufficient_confidence"