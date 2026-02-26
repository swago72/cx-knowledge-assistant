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
    google_api_key: str

    # Optional (defaults match your current code)
    chroma_path: str = "./chroma_db"
    collection_name: str = "cx_knowledge_base"
    html_dir: str = "data/html"
    analytics_db_path: Path = Path("data") / "analytics.db"

    # RAG defaults
    model_name: str = "models/gemini-2.5-pro"
    temperature: float = 0.2
    k: int = 5
    max_distance: float = 0.55
    min_hits: int = 1


def get_settings() -> Settings:
    """
    Single source of truth for config. Fails fast if required values are missing.
    Allows overrides via .env while keeping safe defaults.
    """
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY.\n"
            "Fix:\n"
            "1) Copy .env.example -> .env\n"
            "2) Put your key in GOOGLE_API_KEY=...\n"
        )

    chroma_path = os.getenv("CHROMA_PATH", "./chroma_db")
    collection = os.getenv("CHROMA_COLLECTION", "cx_knowledge_base")
    html_dir = os.getenv("HTML_DIR", "data/html")
    db_path = os.getenv("ANALYTICS_DB_PATH", "data/analytics.db")

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
        model_name=model_name,
        temperature=temperature,
        k=k,
        max_distance=max_distance,
        min_hits=min_hits,
    )