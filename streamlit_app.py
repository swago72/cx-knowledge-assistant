import pandas as pd
import streamlit as st

from src.analytics import (
    fetch_logs,
    containment_metrics,
    daily_volume,
    knowledge_gaps,
)

st.set_page_config(page_title="CX Knowledge Assistant — Optimization Dashboard", layout="wide")

st.title("CX Knowledge Assistant — Optimization Dashboard")
st.caption("Decision telemetry + containment analytics (ANSWER / CLARIFY / HANDOFF)")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Filters")

since_days = st.sidebar.selectbox(
    "Time window",
    options=[7, 14, 30, 90, 365],
    index=2,
)

limit = st.sidebar.number_input("Max rows to load", min_value=100, max_value=200000, value=5000, step=100)

show_raw = st.sidebar.checkbox("Show raw logs", value=False)

# -----------------------------
# Load data
# -----------------------------
df = fetch_logs(limit=int(limit), since_days=int(since_days))

if df.empty:
    st.warning("No logs found yet. Run a few queries first (e.g., python -m src.run_one).")
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
    st.info("No intent/state telemetry found yet. Run queries through the RAG pipeline that logs intent+state.")
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

# -----------------------------
# Knowledge Gaps
# -----------------------------
st.subheader("Knowledge Gaps")

gap_threshold = st.slider("Distance threshold (higher = worse)", min_value=0.0, max_value=1.5, value=0.75, step=0.01)
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
        "intent_confidence",
        "retrieval_confidence",
    ]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, height=450)