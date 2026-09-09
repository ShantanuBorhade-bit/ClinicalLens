# backend/rag.py

"""
RAG pipeline: embed a user query, retrieve the top-k most relevant chunks from
the whole ChromaDB collection (across ALL uploaded documents), build a grounded
prompt, call Gemini, and return an answer with source citations.

Failure handling:
- Transient errors (504 Deadline Exceeded, 503 Unavailable, network timeouts)
  are retried automatically with exponential backoff.
- Free-tier DAILY quota exhaustion (429 per-day) is NOT retried on the same
  model — instead the pipeline falls back to the next model in
  GEMINI_FALLBACK_MODELS, because per-model daily quotas are separate buckets.
- Permanent errors (invalid key, retired model, safety blocks) fail fast with
  a clear, actionable message.
"""

import time

import google.generativeai as genai

from backend.config import (
    GEMINI_API_KEY,
    GEMINI_FALLBACK_MODELS,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MAX_RETRIES,
    GEMINI_MODEL,
    GEMINI_RETRY_BASE_DELAY,
    GEMINI_TEMPERATURE,
    GEMINI_TIMEOUT_SECONDS,
    TOP_K,
)
from backend.database import ChromaDBManager
from backend.embeddings import EmbeddingGenerator
from backend.prompts import SYSTEM_PROMPT, build_user_prompt

NOT_FOUND_MESSAGE = "Not found in the documents."

# Substrings that identify TRANSIENT Google API errors worth retrying.
_TRANSIENT_ERROR_MARKERS = (
    "504",
    "deadline exceeded",
    "503",
    "unavailable",
    "timed out",
    "timeout",
    "connection reset",
    "connection aborted",
    "internal error",
    "500",
)

# Substrings that identify PERMANENT errors — retrying can never help.
_PERMANENT_ERROR_MARKERS = (
    "api key not valid",
    "api_key_invalid",
    "permission denied",
    "403",
    "401",
    "404",
    "not found for api version",
    "is no longer available",
    "safety",
    "blocked",
)

# 429 subtypes. A per-minute rate limit clears in seconds (worth a short
# retry); a per-DAY quota will not clear for hours (fall back to another
# model instead — per-model daily quotas are separate buckets).
_DAILY_QUOTA_MARKERS = (
    "perday",
    "per_day",
    "per day",
    "daily",
    "generatecontentfreetierrequests",
    "generate requests per day",
)


def _is_daily_quota_error(exc: Exception) -> bool:
    """True when a 429 is a per-DAY quota exhaustion (fallback-worthy)."""
    message = str(exc).lower()
    return "429" in message and any(m in message for m in _DAILY_QUOTA_MARKERS)


def _classify_gemini_error(exc: Exception) -> tuple[bool, str | None]:
    """
    Decide whether a Gemini exception is transient (retryable) and build a
    friendly message for permanent failures.

    Returns (is_transient, friendly_message_or_None).
    """
    message = str(exc).lower()

    for marker in _PERMANENT_ERROR_MARKERS:
        if marker in message:
            friendly = {
                "api key not valid": (
                    "Gemini rejected the API key (API_KEY_INVALID). Create a "
                    "valid key at https://aistudio.google.com/apikey, put it "
                    "in GEMINI_API_KEY in your .env, and restart the backend."
                ),
                "permission denied": (
                    "Gemini denied access (403). The Generative Language API may "
                    "be disabled for the key's project, or the key has API "
                    "restrictions. Create a new key in a fresh project in AI "
                    "Studio."
                ),
                "404": (
                    f"Gemini model issue (404): the configured model was likely "
                    f"retired. Set GEMINI_MODEL=gemini-flash-latest in .env and "
                    "restart. Original error: {exc}"
                ),
            }.get(marker)
            return False, friendly or f"Gemini API call failed: {exc}"

    # 429 quota: per-minute limits clear in seconds (brief retry is fine);
    # per-DAY quotas never clear inside a request, so fail fast — the caller
    # falls back to the next model in the chain.
    if "429" in message or "resource_exhausted" in message or "quota" in message:
        if _is_daily_quota_error(exc):
            return False, None  # handled by model fallback, not retrying
        return True, None

    for marker in _TRANSIENT_ERROR_MARKERS:
        if marker in message:
            return True, None

    # Unknown error: retry once anyway — flaky network errors often look
    # like nothing recognizable.
    return True, None


def _is_transient(exc: Exception) -> bool:
    return _classify_gemini_error(exc)[0]


def _make_gemini_model(model_name: str):
    """Build a Gemini GenerativeModel with the grounded system prompt."""
    return genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_PROMPT,
    )


class DailyQuotaExhausted(Exception):
    """Raised when a model's per-day free-tier quota is used up."""

    def __init__(self, model_name: str, original: Exception):
        self.model_name = model_name
        self.original = original
        super().__init__(f"Daily quota exhausted for {model_name}: {original}")


def _call_gemini_with_retry(model, prompt: str, model_name: str = "model"):
    """
    Call model.generate_content with exponential backoff on transient errors.

    - Permanent errors (invalid key, retired model, safety blocks) raise
      immediately as RuntimeError with an actionable message.
    - Per-DAY quota exhaustion raises DailyQuotaExhausted immediately (no
      pointless retries — it won't clear for hours).
    - Other transient errors are retried up to GEMINI_MAX_RETRIES extra
      times, then raised as RuntimeError.
    """
    last_exc: Exception | None = None

    for attempt in range(1 + GEMINI_MAX_RETRIES):
        try:
            return model.generate_content(
                prompt,
                generation_config={
                    "temperature": GEMINI_TEMPERATURE,
                    "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
                },
                request_options={"timeout": GEMINI_TIMEOUT_SECONDS},
            )
        except Exception as exc:
            last_exc = exc

            if _is_daily_quota_error(exc):
                raise DailyQuotaExhausted(model_name, exc) from exc

            transient, friendly = _classify_gemini_error(exc)

            if not transient:
                raise RuntimeError(friendly or f"Gemini API call failed: {exc}") from exc

            if attempt < GEMINI_MAX_RETRIES:  # no sleep after the last try
                delay = GEMINI_RETRY_BASE_DELAY * (2 ** attempt)
                print(
                    f"[rag] Gemini transient error on {model_name} "
                    f"(attempt {attempt + 1}/{1 + GEMINI_MAX_RETRIES}): {exc} "
                    f"— retrying in {delay:.1f}s"
                )
                time.sleep(delay)

    raise RuntimeError(
        f"Gemini is unavailable right now (tried {1 + GEMINI_MAX_RETRIES} "
        f"times). Last error: {last_exc}"
    ) from last_exc


class RAGResult:
    """
    Answer plus provenance.

    Attributes:
        answer: Gemini's grounded answer text.
        sources: deduplicated list of {source, page, document_id} dicts, in
            retrieval-rank order.
        chunks: the raw retrieved chunks (text, source, page, distance, ...).
    """

    def __init__(self, answer: str, chunks: list[dict]):
        self.answer = answer
        self.chunks = chunks

        sources: list[dict] = []
        seen: set[tuple] = set()
        for chunk in chunks:
            key = (chunk["source"], chunk["page"], chunk["document_id"])
            if key not in seen:
                seen.add(key)
                sources.append(
                    {
                        "source": chunk["source"],
                        "page": chunk["page"],
                        "document_id": chunk["document_id"],
                    }
                )
        self.sources = sources

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
        }


class RAGPipeline:
    """
    Wires EmbeddingGenerator + ChromaDBManager + Gemini into one query flow.

    Dependencies are injectable so tests can pass fakes:

        RAGPipeline(embedder=my_fake_embedder, db=my_fake_db)
    """

    def __init__(
        self,
        embedder: EmbeddingGenerator | None = None,
        db: ChromaDBManager | None = None,
    ):
        self.embedder = embedder if embedder is not None else EmbeddingGenerator()
        self.db = db if db is not None else ChromaDBManager()

        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env, add your "
                "key from https://aistudio.google.com/apikey, and restart."
            )
        genai.configure(api_key=GEMINI_API_KEY)

        # Primary model first, then fallbacks (each has a separate per-model
        # daily quota bucket, so falling back genuinely buys headroom).
        self.model_chain: list[str] = [GEMINI_MODEL, *GEMINI_FALLBACK_MODELS]
        # De-duplicate while preserving order.
        seen: set[str] = set()
        self.model_chain = [m for m in self.model_chain
                            if not (m in seen or seen.add(m))]
        self.models = {name: _make_gemini_model(name)
                       for name in self.model_chain}
        self.model = self.models[self.model_chain[0]]

    # ------------------------------------------------------------------ #
    # Retrieval                                                          #
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Embed the query and pull the top-k chunks across the whole collection.

        Returns a list of chunk dicts:
            {text, source, page, document_id, chunk_id, distance}
        """
        # Collection is empty -> nothing to retrieve. ChromaDB raises on
        # queries against an empty collection, so short-circuit instead.
        if self.db.collection.count() == 0:
            return []

        query_embedding = self.embedder.embed_text(query)
        # ChromaDB's REST/serialization layer chokes on raw numpy scalars
        # ("Could not embed query / np.float32 not JSON serializable"), so
        # convert to plain Python floats.
        query_embedding = [float(v) for v in query_embedding]

        results = self.db.search(query_embedding, top_k=top_k)

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        chunks: list[dict] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            chunks.append(
                {
                    "text": text,
                    "source": metadata.get("filename", "unknown"),
                    "page": metadata.get("page", 0),
                    "document_id": metadata.get("document_id", ""),
                    "chunk_id": metadata.get("chunk_id"),
                    "distance": distance,
                }
            )
        return chunks

    # ------------------------------------------------------------------ #
    # Generation                                                         #
    # ------------------------------------------------------------------ #

    def _generate(self, question: str, chunks: list[dict]) -> str:
        """
        Call Gemini with the grounded prompt, retrying transient failures and
        falling back through the model chain when a model's daily quota is
        exhausted. Used to remember which model answered last.
        """
        prompt = build_user_prompt(question, chunks)

        last_error: Exception | None = None
        for name in self.model_chain:
            try:
                response = _call_gemini_with_retry(self.models[name], prompt,
                                                   model_name=name)
                self.model = self.models[name]  # remember the responder
                break
            except DailyQuotaExhausted as exc:
                print(f"[rag] {exc} — falling back to next model")
                last_error = exc
        else:
            raise RuntimeError(
                "All Gemini models have hit their daily free-tier quota. "
                "Free tier allows ~20 requests/day per model. Wait for the "
                "quota to reset (midnight Pacific Time), enable billing on "
                "your Google AI Studio project for higher limits, or ask "
                "fewer questions per day. Last error: " f"{last_error}"
            ) from last_error

        # Blocked or empty responses raise by default; keep the failure
        # explicit and readable instead.
        try:
            answer = response.text.strip()
        except ValueError as exc:
            blocked = getattr(response, "prompt_feedback", None)
            raise RuntimeError(
                f"Gemini returned no answer (possibly safety-blocked). "
                f"Feedback: {blocked}"
            ) from exc

        return answer or NOT_FOUND_MESSAGE

    # ------------------------------------------------------------------ #
    # Public entry point                                                 #
    # ------------------------------------------------------------------ #

    def ask(self, question: str, top_k: int = TOP_K) -> RAGResult:
        """
        Full RAG flow for one question.

        Raises:
            ValueError: if the question is empty.
            RuntimeError: if Gemini fails or the API key is missing.
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("Question must not be empty.")

        # API layer may pass top_k=None to mean "use the configured default".
        top_k = top_k or TOP_K

        chunks = self.retrieve(question, top_k=top_k)

        if not chunks:
            # Clear, explicit state for "no documents uploaded yet".
            return RAGResult(
                answer=NOT_FOUND_MESSAGE + " No documents have been uploaded yet.",
                chunks=[],
            )

        answer = self._generate(question, chunks)
        return RAGResult(answer=answer, chunks=chunks)


# ---------------------------------------------------------------------- #
# Lazy singleton so FastAPI/Streamlit share one model + one DB handle    #
# ---------------------------------------------------------------------- #

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Create the pipeline on first use, reuse it afterwards."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
