from __future__ import annotations
from typing import Dict, Tuple

CLARIFY_TEMPLATES: Dict[str, str] = {
    "refund_request": "Quick question so I can answer accurately: was this purchase made in the last 48 hours, and was it an app, in-app purchase, or a subscription?",
    "payment_issue": "To help troubleshoot: are you seeing an error message (like OR-*) and are you on Android, web, or another device?",
    "cancel_subscription": "To confirm: are you trying to cancel a subscription through Google Play, and do you see it under Payments & subscriptions?",
    "unknown": "Quick clarification: what exactly are you trying to do (refund, cancel subscription, fix payment, or app install issue)?",
}

HANDOFF_TEMPLATES: Dict[str, str] = {
    "low_retrieval_confidence": "I can’t confidently answer from the current knowledge base. If you paste the exact error message / screenshot text or share the specific policy page you’re referencing, I can help. Otherwise, this likely needs escalation to an agent.",
    "llm_exception": "I hit a system error while generating a grounded answer. Please try again, or escalate to an agent if this is urgent.",
    "empty_collection": "The knowledge base is currently empty/unavailable. Please re-run ingestion or escalate to an agent.",
    "too_few_hits": "I didn’t find enough relevant policy content in the knowledge base to answer confidently. Please add the relevant support article or escalate to an agent.",
}

def build_policy_message(state: str, intent: str | None, reason: str | None) -> str:
    intent_key = intent or "unknown"
    reason_key = reason or "low_retrieval_confidence"

    if state == "CLARIFY":
        return CLARIFY_TEMPLATES.get(intent_key, CLARIFY_TEMPLATES["unknown"])

    if state == "HANDOFF":
        return HANDOFF_TEMPLATES.get(reason_key, HANDOFF_TEMPLATES["low_retrieval_confidence"])

    return ""  # ANSWER handled by model output