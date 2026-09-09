"""
Offline tests for the Gemini retry/error-classification logic in backend/rag.py.

Run from the project root:
    venv/Scripts/python.exe tests/test_gemini_retry.py

No network access needed: the Gemini client is simulated with fakes.
"""
import os
import sys
import time

# Provide a dummy key before importing backend.config.
os.environ.setdefault("GEMINI_API_KEY", "test-key-for-offline-tests")

sys.path.insert(0, ".")

from backend.rag import (
    DailyQuotaExhausted,
    NOT_FOUND_MESSAGE,
    RAGPipeline,
    RAGResult,
    _call_gemini_with_retry,
    _is_daily_quota_error,
    _is_transient,
)

fails = []


def check(name, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        fails.append(name)


# --------------------------------------------------------------------- #
# 1. Error classification                                               #
# --------------------------------------------------------------------- #
class GoogleishError(Exception):
    pass


transient_cases = [
    "504 Deadline Exceeded",
    "503 Service Unavailable",
    "429 Resource has been exhausted (e.g. check quota).",
    "HTTPSConnectionPool(host='generativelanguage.googleapis.com'): Read timed out.",
    "Connection aborted.', RemoteDisconnected('Remote end closed connection'))",
    "500 Internal Server Error",
]
permanent_cases = [
    "400 API key not valid. Please pass a valid API key. [API_KEY_INVALID]",
    "403 Permission denied: Generative Language API has not been used...",
    "404 models/gemini-2.0-flash is no longer available.",
]

for msg in transient_cases:
    check(f"transient: {msg[:45]}", _is_transient(GoogleishError(msg)))
for msg in permanent_cases:
    check(f"permanent: {msg[:45]}", not _is_transient(GoogleishError(msg)))

# Unknown errors default to transient (retry once more can only help).
check("unknown error treated as transient", _is_transient(GoogleishError("weird failure")))

# 429 subtypes: per-minute -> brief retry; per-DAY -> fail fast for fallback.
per_minute = GoogleishError(
    "429 Resource exhausted, requests per minute limit reached. retry in 3s"
)
per_day = GoogleishError(
    '429 You exceeded your current quota. quota_id: '
    '"GenerateRequestsPerDayPerProjectPerModel-FreeTier" '
    'quota_metric: "generativelanguage.googleapis.com/'
    'generate_content_free_tier_requests", limit: 20, model: gemini-3.8-flash '
    'Please retry in 47s'
)
check("per-minute 429 -> transient (retry)", _is_transient(per_minute))
check("per-day 429 -> NOT retried", not _is_transient(per_day))
check("per-day 429 detected as daily quota", _is_daily_quota_error(per_day))
check("per-minute 429 not daily quota", not _is_daily_quota_error(per_minute))

# --------------------------------------------------------------------- #
# 2. Retry behavior with a fake model                                   #
# --------------------------------------------------------------------- #


class FlakyModel:
    """Raises `fail_times` transient errors, then succeeds."""

    def __init__(self, fail_times, exc=None):
        self.fail_times = fail_times
        self.exc = exc or GoogleishError("504 Deadline Exceeded")
        self.calls = 0

    def generate_content(self, prompt, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc

        class R:
            text = "grounded answer [Source 1]"

        return R()


# Speed up the test: shrink backoff.
os.environ["GEMINI_RETRY_BASE_DELAY"] = "0.05"
os.environ["GEMINI_MAX_RETRIES"] = "2"
import importlib
import backend.rag as ragmod
import backend.config as configmod

importlib.reload(configmod)
importlib.reload(ragmod)

# Rebind after reload: reload() creates NEW class objects, so names imported
# at the top would no longer match the reloaded module's classes
# (except-clauses would silently never match).
DailyQuotaExhausted = ragmod.DailyQuotaExhausted
NOT_FOUND_MESSAGE = ragmod.NOT_FOUND_MESSAGE
RAGPipeline = ragmod.RAGPipeline
RAGResult = ragmod.RAGResult
_call_gemini_with_retry = ragmod._call_gemini_with_retry
_is_daily_quota_error = ragmod._is_daily_quota_error
_is_transient = ragmod._is_transient

start = time.time()
result = _call_gemini_with_retry(FlakyModel(fail_times=2), "prompt")
check("recovers after 2 transient failures", result.text == "grounded answer [Source 1]")
elapsed = time.time() - start
check("backoff actually slept between attempts", 0.05 <= elapsed < 2.0, f"{elapsed:.2f}s")

try:
    _call_gemini_with_retry(FlakyModel(fail_times=99), "prompt")
    check("retries exhausted -> RuntimeError", False)
except RuntimeError as exc:
    check("retries exhausted -> RuntimeError", "tried 3 times" in str(exc), str(exc)[:70])

# Per-day quota must raise DailyQuotaExhausted immediately (exactly 1 call).
quota_model = FlakyModel(fail_times=99, exc=per_day)
try:
    _call_gemini_with_retry(quota_model, "prompt", "test-model")
    check("per-day 429 raises DailyQuotaExhausted immediately", False)
except DailyQuotaExhausted as exc:
    check("per-day 429 raises DailyQuotaExhausted immediately",
          quota_model.calls == 1 and exc.model_name == "test-model")

# Permanent errors must fail FAST (1 call, no retries, actionable message).
fast = FlakyModel(fail_times=99, exc=GoogleishError("400 API key not valid. [API_KEY_INVALID]"))
try:
    _call_gemini_with_retry(fast, "prompt")
    check("invalid key fails fast", False)
except RuntimeError as exc:
    check("invalid key fails fast", fast.calls == 1 and "aistudio.google.com/apikey" in str(exc))

retired = FlakyModel(fail_times=99, exc=GoogleishError("404 models/x is no longer available."))
try:
    _call_gemini_with_retry(retired, "prompt")
    check("retired model fails fast", False)
except RuntimeError as exc:
    check("retired model fails fast", retired.calls == 1 and "GEMINI_MODEL" in str(exc))

# --------------------------------------------------------------------- #
# 3. Full pipeline flow with fakes                                      #
# --------------------------------------------------------------------- #


class FakeEmbedder:
    def embed_text(self, text):
        return [0.1, 0.2]


class FakeCollection:
    def __init__(self, n):
        self._n = n

    def count(self):
        return self._n


class FakeDB:
    def __init__(self, n=2):
        self.collection = FakeCollection(n)

    def search(self, embedding, top_k=4):
        return {
            "documents": [["some retrieved passage"]],
            "metadatas": [[{"document_id": "d1", "filename": "study.pdf",
                            "page": 5, "chunk_id": 0}]],
            "distances": [[0.2]],
        }


pipe = RAGPipeline(embedder=FakeEmbedder(), db=FakeDB())
# Inject fakes into the model dict (never assign pipe.model directly —
# _generate now walks self.models through the chain).
for _name in pipe.model_chain:
    pipe.models[_name] = FlakyModel(fail_times=1)  # first call 504s, then OK
res = pipe.ask("What is the dose?")
check("pipeline survives one transient error", "[Source 1]" in res.answer)
check("sources retrieved", res.sources[0]["source"] == "study.pdf"
      and res.sources[0]["page"] == 5)

empty = RAGPipeline(embedder=FakeEmbedder(), db=FakeDB(n=0))
for _name in empty.model_chain:
    empty.models[_name] = FlakyModel(fail_times=99)  # must never be reached
res2 = empty.ask("anything")
check("empty DB -> not-found without Gemini call", res2.answer.startswith(NOT_FOUND_MESSAGE))

# --------------------------------------------------------------------- #
# 4. Daily-quota model fallback                                          #
# --------------------------------------------------------------------- #


class QuotaAwareModel(FlakyModel):
    """Raises per-day 429 forever (quota never resets mid-request)."""

    def __init__(self):
        super().__init__(fail_times=99,
                         exc=GoogleishError(
                             '429 quota_id: "GenerateRequestsPerDayPerProjectPerModel'
                             '-FreeTier", limit: 20, retry in 47s'))


class AnswerModel(FlakyModel):
    def __init__(self, tag):
        super().__init__(fail_times=0)
        self.tag = tag

    def generate_content(self, prompt, **kwargs):
        self.calls += 1

        class R:
            text = f"answer from {self.tag} [Source 1]"

        return R()


pipe_fb = RAGPipeline(embedder=FakeEmbedder(), db=FakeDB())
# Primary (first chain entry) is quota-dead; the rest answer normally.
dead_names = pipe_fb.model_chain[:1]  # only the primary
for name in dead_names:
    pipe_fb.models[name] = QuotaAwareModel()
for name in pipe_fb.model_chain[1:]:
    pipe_fb.models[name] = AnswerModel(name)

res_fb = pipe_fb.ask("What is the dose?")
check("falls back when primary quota is dead",
      "answer from" in res_fb.answer and res_fb.answer.startswith("answer from"),
      res_fb.answer[:50])
check("pipeline switched to a fallback model",
      pipe_fb.model in [pipe_fb.models[n] for n in pipe_fb.model_chain[1:]])

# ALL models dead -> clear quota message.
pipe_dead = RAGPipeline(embedder=FakeEmbedder(), db=FakeDB())
for name in pipe_dead.model_chain:
    pipe_dead.models[name] = QuotaAwareModel()
try:
    pipe_dead.ask("q")
    check("all models dead -> clear quota error", False)
except RuntimeError as exc:
    check("all models dead -> clear quota error",
          "daily free-tier quota" in str(exc) and "20 requests/day" in str(exc),
          str(exc)[:80])

print()
print("ALL RETRY TESTS PASSED" if not fails else f"FAILURES: {fails}")
sys.exit(1 if fails else 0)
