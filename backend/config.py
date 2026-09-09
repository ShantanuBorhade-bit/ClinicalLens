# backend/config.py

import os
from pathlib import Path

from dotenv import load_dotenv

# All file-system paths are anchored to the project root (the folder that
# contains backend/, frontend/, uploads/, chroma_db/) so the app behaves the
# same no matter which directory you launch uvicorn or streamlit from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load GEMINI_API_KEY etc. from <project root>/.env if present.
# Existing environment variables always win over .env values.
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_path(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Vector store
COLLECTION_NAME = "clinical_documents"

# Retrieval
TOP_K = int(os.getenv("TOP_K", "4"))

# Gemini
# Get a key at https://aistudio.google.com/apikey and put it in your .env file.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Free-tier-friendly default: the flash-LITE alias is the lightest (and
# fastest/cheapest) model, and per-model daily quotas are separate buckets,
# so using lite preserves the heavier models' quotas.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")

# If the primary model's free-tier daily quota is exhausted (429 per-day),
# the pipeline automatically tries these models, in order, before giving up.
# Separate comma-separated list; keep "lite" models first.
GEMINI_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-2.5-flash-lite,gemini-flash-latest",
    ).split(",")
    if m.strip()
]
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "2048"))

# Timeout (seconds) for Gemini API calls
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "60"))

# Retry with exponential backoff for TRANSIENT Gemini errors
# (504 Deadline Exceeded, 503 Unavailable, 429 rate limits, timeouts).
# GEMINI_MAX_RETRIES = extra attempts after the first one fails.
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
GEMINI_RETRY_BASE_DELAY = float(os.getenv("GEMINI_RETRY_BASE_DELAY", "2.0"))

# Filesystem
UPLOAD_FOLDER = _resolve_path(os.getenv("UPLOAD_FOLDER", "uploads"))
CHROMA_DB_PATH = _resolve_path(os.getenv("CHROMA_DB_PATH", "chroma_db"))

# API
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

# URL the Streamlit frontend uses to reach the FastAPI backend
BACKEND_URL = os.getenv("BACKEND_URL", f"http://{API_HOST}:{API_PORT}")

# Upload constraints
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
ALLOWED_EXTENSIONS = {".pdf"}
