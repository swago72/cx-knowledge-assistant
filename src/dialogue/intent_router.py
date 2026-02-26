def detect_intent(question: str) -> tuple[str, float]:
    q = question.lower()

    if "refund" in q:
        return "refund_request", 0.9
    if "subscription" in q:
        return "subscription_issue", 0.85
    if "payment" in q:
        return "payment_issue", 0.8
    if "install" in q or "download" in q:
        return "install_issue", 0.8

    return "unknown", 0.4