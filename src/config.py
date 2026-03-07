# src/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env once, globally
load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Required
    google_api_key: str = ""

    # Optional (defaults match your current code)
    chroma_path: Path = Path("chroma_db")
    collection_name: str = "cx_knowledge_base"
    html_dir: Path = Path("data") / "html"
    analytics_db_path: Path = Path("data") / "analytics.db"
    base_dir: Path = Path(__file__).resolve().parent.parent

    # RAG defaults
    model_name: str = "models/gemini-2.5-pro"
    temperature: float = 0.2
    k: int = 5
    max_distance: float = 0.55
    min_hits: int = 1


def get_settings() -> Settings:
    """
    Single source of truth for config with safe defaults for Streamlit Cloud.
    GOOGLE_API_KEY may be empty; callers that require LLM access should validate it.
    """
    key = os.getenv("GOOGLE_API_KEY", "").strip()

    base_dir = Path(__file__).resolve().parent.parent

    chroma_path = Path(os.getenv("CHROMA_PATH", "chroma_db"))
    if not chroma_path.is_absolute():
        chroma_path = base_dir / chroma_path

    collection = os.getenv("CHROMA_COLLECTION", "cx_knowledge_base")

    html_dir = Path(os.getenv("HTML_DIR", "data/html"))
    if not html_dir.is_absolute():
        html_dir = base_dir / html_dir

    db_path = Path(os.getenv("ANALYTICS_DB_PATH", "data/analytics.db"))
    if not db_path.is_absolute():
        db_path = base_dir / db_path

    # Optional overrides for tuning
    model_name = os.getenv("MODEL_NAME", "models/gemini-2.5-pro")
    temperature = float(os.getenv("TEMPERATURE", "0.2"))
    k = int(os.getenv("TOP_K", "5"))
    max_distance = float(os.getenv("MAX_DISTANCE", "0.55"))
    min_hits = int(os.getenv("MIN_HITS", "1"))

    return Settings(
        google_api_key=key,
        chroma_path=chroma_path,
        collection_name=collection,
        html_dir=html_dir,
        analytics_db_path=Path(db_path),
        base_dir=base_dir,
        model_name=model_name,
        temperature=temperature,
        k=k,
        max_distance=max_distance,
        min_hits=min_hits,
    )
