import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.rag import ensure_knowledge_base, ask, ChatMemory
from src.analytics import (
    fetch_logs,
    containment_metrics,
    daily_volume,
    knowledge_gaps,
)

st.set_page_config(page_title="CX Knowledge Assistant — Optimization Dashboard", layout="wide")


# -----------------------------
# Cached loaders (performance)
# -----------------------------
@st.cache_resource
def _load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def _load_collection():
    return ensure_knowledge_base()




collection = _load_collection()
st.caption(f"Knowledge base chunks loaded: {collection.count()}")



# -----------------------------
# Header
# -----------------------------
st.title("CX Knowledge Assistant — Chat + Optimization Dashboard")
st.caption("Shareable demo: chat experience + decision telemetry + containment analytics")


# -----------------------------
# Sidebar controls (dashboard filters)
# -----------------------------
st.sidebar.header("Filters")

since_days = st.sidebar.selectbox(
    "Time window",
    options=[7, 14, 30, 90, 365],
    index=2,
)

limit = st.sidebar.number_input(
    "Max rows to load",
    min_value=100,
    max_value=200000,
    value=5000,
    step=100
)

show_raw = st.sidebar.checkbox("Show raw logs", value=False)


# -----------------------------
# Tabs: Chat + Dashboard
# -----------------------------
tab_chat, tab_dashboard = st.tabs(["💬 Chat", "📊 Dashboard"])


# ============================================================
# TAB 1: CHAT
# ============================================================
with tab_chat:
    st.subheader("Ask the CX Knowledge Assistant")

    # Per-user session memory (each browser user gets their own)
    if "chat_memory" not in st.session_state:
        st.session_state.chat_memory = ChatMemory(max_turns=5)

    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}

    # Render chat history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_q = st.chat_input("Ask a support question…")

    if user_q:
        # Display user's message
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.write(user_q)

        embed_model = _load_embed_model()
        collection = _load_collection()

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask(
                    user_q,
                    collection,
                    embed_model,
                    memory=st.session_state.chat_memory,
                    debug=False,
                )

            st.write(result["answer"])
            st.caption(
                f"State: **{result.get('state')}** | Intent: **{result.get('intent')}** | Reason: **{result.get('reason')}**"
            )

        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

    col_reset, col_spacer = st.columns([1, 5])
    with col_reset:
        if st.button("Reset chat"):
            st.session_state.messages = []
            st.session_state.chat_memory = ChatMemory(max_turns=5)
            st.rerun()


# ============================================================
# TAB 2: DASHBOARD
# ============================================================
with tab_dashboard:
    # -----------------------------
    # Load data
    # -----------------------------
    df = fetch_logs(limit=int(limit), since_days=int(since_days))

    if df.empty:
        st.warning("No logs found yet. Ask a few questions in the Chat tab first.")
        st.stop()

    # Normalize extra fields
    def _get_extra_val(x, key):
        if isinstance(x, dict):
            return x.get(key)
        return None

    if "extra" in df.columns:
        df["state"] = df["extra"].apply(lambda x: _get_extra_val(x, "state"))
        df["intent"] = df["extra"].apply(lambda x: _get_extra_val(x, "intent"))
        df["decision_reason"] = df["extra"].apply(lambda x: _get_extra_val(x, "decision_reason"))
        df["intent_confidence"] = df["extra"].apply(lambda x: _get_extra_val(x, "intent_confidence"))
        df["retrieval_confidence"] = df["extra"].apply(lambda x: _get_extra_val(x, "retrieval_confidence"))
        df["policy_message"] = df["extra"].apply(lambda x: _get_extra_val(x, "policy_message"))
        df["gate_reason"] = df["extra"].apply(lambda x: _get_extra_val(x, "gate_reason"))

    # -----------------------------
    # KPI Row
    # -----------------------------
    kpis = containment_metrics()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Logged (state-known)", kpis.get("total", 0))
    c2.metric("Containment Rate (ANSWER)", kpis.get("containment_rate", 0))
    c3.metric("Clarify Rate (CLARIFY)", kpis.get("clarify_rate", 0))
    c4.metric("Handoff Rate (HANDOFF)", kpis.get("handoff_rate", 0))

    st.divider()

    # =============================
    # Workforce Impact Simulation
    # =============================
    st.subheader("Workforce Impact Simulation")

    avg_handle_time = st.slider(
        "Avg Handle Time (minutes)",
        min_value=1.0,
        max_value=15.0,
        value=6.0,
        step=0.5
    )

    cost_per_hour = st.slider(
        "Cost per Agent Hour ($)",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=1.0
    )

    containment = kpis.get("containment_rate", 0)
    total = kpis.get("total", 0)

    deflected = total * containment
    minutes_saved = deflected * avg_handle_time
    hours_saved = minutes_saved / 60
    cost_saved = hours_saved * cost_per_hour

    w1, w2, w3 = st.columns(3)
    w1.metric("Deflected Queries", round(deflected, 2))
    w2.metric("Agent Hours Saved", round(hours_saved, 2))
    w3.metric("Estimated Cost Saved ($)", round(cost_saved, 2))

    st.divider()

    # -----------------------------
    # Daily Volume Trend
    # -----------------------------
    st.subheader("Daily Query Volume")

    dv = daily_volume(since_days=int(since_days))
    if dv.empty:
        st.info("No daily volume data yet.")
    else:
        dv = dv.rename(columns={"queries": "count"})
        dv["day"] = pd.to_datetime(dv["day"])
        st.line_chart(dv.set_index("day")["count"])

    st.divider()

    # -----------------------------
    # Containment by Intent
    # -----------------------------
    st.subheader("Containment by Intent")

    intent_df = df.dropna(subset=["intent", "state"]).copy()

    if intent_df.empty:
        st.info("No intent/state telemetry found yet. Ask questions in the Chat tab to generate logs.")
    else:
        counts = (
            intent_df.groupby(["intent", "state"])
            .size()
            .reset_index(name="count")
            .sort_values(["intent", "state"])
        )

        totals = intent_df.groupby("intent").size().reset_index(name="total")
        rates = counts.merge(totals, on="intent", how="left")
        rates["rate"] = (rates["count"] / rates["total"]).round(3)

        left, right = st.columns([1, 1])

        with left:
            st.markdown("**Counts**")
            st.dataframe(counts, use_container_width=True, height=300)

        with right:
            st.markdown("**Rates**")
            st.dataframe(rates[["intent", "state", "count", "total", "rate"]], use_container_width=True, height=300)

    st.divider()

    # -----------------------------
    # Escalation Drivers (HANDOFF reasons)
    # -----------------------------
    st.subheader("Escalation Drivers (HANDOFF)")

    handoff = df[df.get("state") == "HANDOFF"].copy()

    if handoff.empty:
        st.info("No HANDOFF rows yet in this time window.")
    else:
        if "decision_reason" not in handoff.columns:
            st.warning("No decision_reason column found in logs.")
        else:
            reasons = (
                handoff["decision_reason"]
                .dropna()
                .value_counts()
                .reset_index()
            )
            reasons.columns = ["decision_reason", "count"]

            if reasons.empty:
                st.info("No escalation reasons found.")
            else:
                st.bar_chart(reasons.set_index("decision_reason")["count"])

    st.divider()

    # -----------------------------
    # Knowledge Gaps
    # -----------------------------
    st.subheader("Knowledge Gaps")

    gap_threshold = st.slider(
        "Distance threshold (higher = worse)",
        min_value=0.0,
        max_value=1.5,
        value=0.75,
        step=0.01
    )
    gaps = knowledge_gaps(distance_threshold=float(gap_threshold), since_days=int(since_days), limit=200)

    if gaps.empty:
        st.success("No knowledge gaps detected in this window.")
    else:
        st.dataframe(gaps, use_container_width=True, height=350)

    st.divider()

    # -----------------------------
    # Raw logs (optional)
    # -----------------------------
    if show_raw:
        st.subheader("Raw Logs")
        cols = [
            "timestamp_utc",
            "user_question",
            "fallback_used",
            "top_distance",
            "avg_distance",
            "model_name",
            "state",
            "intent",
            "decision_reason",
            "gate_reason",
            "policy_message",
            "intent_confidence",
            "retrieval_confidence",
        ]
        cols = [c for c in cols if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, height=450)