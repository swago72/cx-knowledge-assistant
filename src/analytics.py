# src/analytics.py
from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd


# -----------------------------
# Config
# -----------------------------
from src.config import get_settings
DEFAULT_DB_PATH = get_settings().analytics_db_path

# Define a canonical "no answer" string so gap detection is consistent.
NO_CONFIDENT_ANSWER = "couldn't find confident answer"


# -----------------------------
# Helpers
# -----------------------------
def utc_now_iso() -> str:
    """UTC timestamp in ISO-8601 with 'Z' suffix."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_parent_dir(path: Union[str, Path]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _json_loads(s: Optional[str]) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return s


# -----------------------------
# Source parsing
# -----------------------------
_SOURCE_PAREN_RE = re.compile(r"\(Source:\s*([^)]+)\)", re.IGNORECASE)
_SOURCES_SECTION_RE = re.compile(
    r"(?:^|\n)\s*Sources Used\s*:?\s*\n(?P<body>.+?)(?:\n\s*\n|$)",
    re.IGNORECASE | re.DOTALL,
)


def get_sources_used(answer_text: str) -> List[str]:
    """
    Parse sources from:
      - Inline patterns like: (Source: doc_name.html)
      - A section like:
            Sources Used:
            - doc1.html
            - doc2.html
    Returns a de-duped list preserving first-seen order.
    """
    if not answer_text:
        return []

    found: List[str] = []

    # 1) Inline "(Source: ...)"
    for m in _SOURCE_PAREN_RE.finditer(answer_text):
        raw = m.group(1).strip()
        # Allow multiple sources separated by commas
        for part in [p.strip() for p in raw.split(",") if p.strip()]:
            found.append(part)

    # 2) "Sources Used:" section
    m2 = _SOURCES_SECTION_RE.search(answer_text)
    if m2:
        body = m2.group("body").strip()
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            # Remove bullets
            line = re.sub(r"^[-*•]\s*", "", line).strip()
            if line:
                found.append(line)

    # De-dupe, preserve order
    deduped: List[str] = []
    seen = set()
    for s in found:
        if s not in seen:
            deduped.append(s)
            seen.add(s)
    return deduped


# -----------------------------
# Schema
# -----------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS query_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc   TEXT NOT NULL,

    user_question   TEXT NOT NULL,
    answer          TEXT NOT NULL,

    sources_used_json   TEXT,   -- JSON list of strings (doc ids / titles)
    top_distance        REAL,   -- best (lowest) distance from Chroma hits
    avg_distance        REAL,   -- average distance over returned hits

    model_name      TEXT,
    latency_ms      INTEGER,
    fallback_used   INTEGER NOT NULL DEFAULT 0,  -- 0/1

    extra_json      TEXT        -- optional: store anything else safely
);

CREATE INDEX IF NOT EXISTS idx_query_logs_timestamp ON query_logs(timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_query_logs_model ON query_logs(model_name);
"""


def _connect(db_path: Union[str, Path]) -> sqlite3.Connection:
    ensure_parent_dir(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Union[str, Path] = DEFAULT_DB_PATH) -> None:
    conn = _connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


# -----------------------------
# Logging
# -----------------------------
@dataclass
class QueryLog:
    timestamp_utc: str
    user_question: str
    answer: str
    sources_used: List[str]
    top_distance: Optional[float] = None
    avg_distance: Optional[float] = None
    model_name: Optional[str] = None
    latency_ms: Optional[int] = None
    fallback_used: bool = False
    extra: Optional[Dict[str, Any]] = None


def log_query(
    *,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    timestamp_utc: Optional[str] = None,
    user_question: str,
    answer: str,
    sources_used: Optional[Union[List[str], str]] = None,
    top_distance: Optional[float] = None,
    avg_distance: Optional[float] = None,
    model_name: Optional[str] = None,
    latency_ms: Optional[int] = None,
    fallback_used: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Insert a query log row. Returns inserted row id.

    sources_used:
      - If None, we will parse from answer via get_sources_used(answer)
      - If a string, we store it as a single-item list
      - If list, stored as JSON
    """
    init_db(db_path)

    ts = timestamp_utc or utc_now_iso()

    if sources_used is None:
        sources_list = get_sources_used(answer)
    elif isinstance(sources_used, str):
        sources_list = [sources_used]
    else:
        sources_list = list(sources_used)

    conn = _connect(db_path)
    try:
        cur = conn.execute(
            """
            INSERT INTO query_logs (
                timestamp_utc, user_question, answer,
                sources_used_json, top_distance, avg_distance,
                model_name, latency_ms, fallback_used, extra_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                user_question,
                answer,
                _json_dumps(sources_list) if sources_list is not None else None,
                top_distance,
                avg_distance,
                model_name,
                latency_ms,
                1 if fallback_used else 0,
                _json_dumps(extra) if extra else None,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


# -----------------------------
# Querying utilities
# -----------------------------
def _df_from_sql(conn: sqlite3.Connection, sql: str, params: Tuple[Any, ...] = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def fetch_logs(
    *,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    limit: int = 5000,
    since_days: Optional[int] = None,
) -> pd.DataFrame:
    init_db(db_path)
    conn = _connect(db_path)
    try:
        if since_days is not None:
            # SQLite text ISO ordering works if timestamps are ISO-8601.
            cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)).replace(microsecond=0)
            cutoff_iso = cutoff.isoformat().replace("+00:00", "Z")
            df = _df_from_sql(
                conn,
                """
                SELECT *
                FROM query_logs
                WHERE timestamp_utc >= ?
                ORDER BY timestamp_utc DESC
                LIMIT ?
                """,
                (cutoff_iso, limit),
            )
        else:
            df = _df_from_sql(
                conn,
                """
                SELECT *
                FROM query_logs
                ORDER BY timestamp_utc DESC
                LIMIT ?
                """,
                (limit,),
            )
    finally:
        conn.close()

    # Parse JSON fields
    if "sources_used_json" in df.columns:
        df["sources_used"] = df["sources_used_json"].apply(_json_loads)
    if "extra_json" in df.columns:
        df["extra"] = df["extra_json"].apply(_json_loads)

    return df


# -----------------------------
# Analysis functions (Week 3)
# -----------------------------
def most_common_questions(
    *,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    limit: int = 20,
    since_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Returns: question, count
    """
    df = fetch_logs(db_path=db_path, limit=200000, since_days=since_days)
    if df.empty:
        return pd.DataFrame(columns=["user_question", "count"])

    out = (
        df.groupby("user_question", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(limit)
    )
    return out


def citations_per_doc(
    *,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    limit: int = 30,
    since_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Returns: doc, citations
    Counts occurrences of docs in sources_used.
    """
    df = fetch_logs(db_path=db_path, limit=200000, since_days=since_days)
    if df.empty:
        return pd.DataFrame(columns=["doc", "citations"])

    rows: List[str] = []
    for sources in df.get("sources_used", []):
        if isinstance(sources, list):
            rows.extend([str(s) for s in sources])
        elif sources:
            rows.append(str(sources))

    if not rows:
        return pd.DataFrame(columns=["doc", "citations"])

    out = (
        pd.Series(rows, name="doc")
        .value_counts()
        .reset_index()
        .rename(columns={"index": "doc", "doc": "citations"})
        .head(limit)
    )
    return out


def daily_volume(
    *,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    since_days: int = 14,
) -> pd.DataFrame:
    """
    Returns: day (YYYY-MM-DD), queries
    """
    df = fetch_logs(db_path=db_path, limit=200000, since_days=since_days)
    if df.empty:
        return pd.DataFrame(columns=["day", "queries"])

    # Convert ISO to datetime
    ts = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    out = (
        pd.DataFrame({"day": ts.dt.date.astype(str)})
        .value_counts()
        .reset_index()
        .rename(columns={0: "queries"})
        .sort_values("day", ascending=True)
    )
    return out


def knowledge_gaps(
    *,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    distance_threshold: float = 0.75,
    limit: int = 50,
    since_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Flags queries where:
      - answer is a canonical no-confident string
      OR
      - top_distance is NULL or above a threshold
    """
    df = fetch_logs(db_path=db_path, limit=200000, since_days=since_days)
    if df.empty:
        return pd.DataFrame(
            columns=["timestamp_utc", "user_question", "top_distance", "avg_distance", "answer", "model_name"]
        )

    ans = df["answer"].fillna("").str.lower()
    no_answer = ans.str.contains(NO_CONFIDENT_ANSWER)

    # If no distances logged, treat as a gap (forces you to wire distances in Week 3)
    td = pd.to_numeric(df["top_distance"], errors="coerce")
    dist_gap = td.isna() | (td > float(distance_threshold))

    gaps = df[no_answer | dist_gap].copy()

    cols = ["timestamp_utc", "user_question", "top_distance", "avg_distance", "answer", "model_name", "fallback_used"]
    cols = [c for c in cols if c in gaps.columns]
    gaps = gaps[cols].head(limit)
    return gaps


# -----------------------------
# Milestone report helper
# -----------------------------
def milestone_report(
    *,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    top_docs_n: int = 10,
    top_questions_n: int = 10,
    since_days: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Returns a dict of DataFrames you can print or render in Streamlit:
      - top_docs
      - top_questions
      - daily_volume
      - knowledge_gaps
    """
    return {
        "top_docs": citations_per_doc(db_path=db_path, limit=top_docs_n, since_days=since_days),
        "top_questions": most_common_questions(db_path=db_path, limit=top_questions_n, since_days=since_days),
        "daily_volume": daily_volume(db_path=db_path, since_days=since_days or 14),
        "knowledge_gaps": knowledge_gaps(db_path=db_path, since_days=since_days),
    }


if __name__ == "__main__":
    # Quick sanity check / local run:
    #   python -m src.analytics
    init_db(DEFAULT_DB_PATH)
    print(f"DB ready at: {DEFAULT_DB_PATH}")

    rep = milestone_report(db_path=DEFAULT_DB_PATH, since_days=30)
    for name, df in rep.items():
        print("\n" + "=" * 80)
        print(name)
        print(df.head(20).to_string(index=False))





def containment_metrics(db_path=DEFAULT_DB_PATH):
    df = fetch_logs(db_path=db_path, limit=200000)
    if df.empty:
        return {}

    states = df["extra"].apply(lambda x: x.get("state") if isinstance(x, dict) else None)

    total = len(states)
    answer = (states == "ANSWER").sum()
    clarify = (states == "CLARIFY").sum()
    handoff = (states == "HANDOFF").sum()

    return {
        "total": total,
        "containment_rate": round(answer / total, 3) if total else 0,
        "clarify_rate": round(clarify / total, 3) if total else 0,
        "handoff_rate": round(handoff / total, 3) if total else 0,
    }
