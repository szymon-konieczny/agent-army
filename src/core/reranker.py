"""Code-oriented reranker for Code Horde RAG pipelines.

Provides a model-agnostic reranking service that scores code fragment
candidates against a natural-language query.  Three layers:

  1. **LLM reranker** — sends candidates to any configured LLM with a
     specialised system prompt; returns per-candidate energy scores
     (lower = more relevant).
  2. **BM25 / TF-IDF fallback** — fast heuristic scorer used when the
     LLM is unavailable or times out.
  3. **Two-tier cache** — session (in-memory) + global (Redis/InMemory
     store) so repeated query+candidate sets skip both LLM and
     heuristic work.

Batch policy: candidates are split into configurable chunks and
processed in parallel to stay within LLM context limits and reduce
latency.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import structlog

logger = structlog.get_logger(__name__)

# ── Constants / defaults ─────────────────────────────────────────────

K_RETRIEVER: int = 80          # max candidates from the retriever
K_RERANK: int = 32             # top-K returned after reranking
BATCH_SIZE: int = 16           # candidates per LLM call
MAX_PARALLEL: int = 4          # concurrent LLM batch calls
LLM_TIMEOUT: float = 3.0       # seconds per batch call
SESSION_CACHE_TTL: int = 1800   # 30 min  (seconds)
GLOBAL_CACHE_TTL: int = 7200    # 2 h     (seconds)

# ── Reranker system prompt ───────────────────────────────────────────

RERANKER_SYSTEM_PROMPT = """\
You are a model-agnostic reranking service used inside a code-oriented \
RAG / agentic system.

Your job:
* Given a natural-language query (possibly mixed with code) and a list \
of candidate code fragments, assign a scalar relevance score to each \
candidate.
* Lower score = better (more relevant) candidate.  This is interpreted \
as an "energy": the system will select candidates with the lowest energy.
* You are agnostic to the programming language, framework, and to the \
LLM used later.  You only see text and metadata.

Input format — a JSON object:
{
  "query": "<string>",
  "candidates": [
    {
      "id": "<string>",
      "content": "<string — code fragment or related text>",
      "metadata": {
        "language": "<string>",
        "framework": "<string — optional>",
        "path": "<string — file path>",
        "repo": "<string — repository>"
      }
    }
  ]
}

Output format — a JSON array:
[
  {"id": "<same as candidate.id>", "score": <float, lower is better>}
]

Scoring guidelines:
1. Language / framework agnostic — treat all uniformly.
2. Focus on semantic relevance to the query: definitions, \
implementations, configurations, types, tests that help resolve the \
query score lowest (best).
3. Prefer candidates with actual implementation logic, function/class \
definitions, key configuration, comments/docstrings referring to the \
query concept.
4. Penalise (higher score): loose keyword overlap only, boilerplate, \
trivial re-exports, clearly test-only / deprecated code.
5. Ensure clearly better candidates get noticeably lower scores.
6. Handle any kind of query: natural-language questions, refactoring \
requests, bug descriptions, stack traces.
7. No side effects — do not modify input or generate new code.

Return ONLY the JSON array, no commentary."""


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class RerankCandidate:
    """A single candidate for reranking."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "content": self.content, "metadata": self.metadata}


@dataclass
class RerankResult:
    """Scored candidate after reranking."""

    id: str
    score: float
    source: str = "reranked"  # reranked | cache | fallback_*

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "score": self.score, "source": self.source}


# ── BM25 / TF-IDF fallback scorer ───────────────────────────────────

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "it", "in", "of", "to", "and", "or", "for",
    "on", "with", "as", "at", "by", "this", "that", "from", "be", "are",
    "was", "were", "been", "not", "but", "if", "do", "does", "did",
    "has", "have", "had", "will", "would", "could", "should", "can",
    "may", "no", "yes", "so", "up", "out", "its", "my", "we", "i",
    "me", "he", "she", "they", "them", "you", "your", "our",
    "import", "from", "return", "def", "class", "const", "let", "var",
    "function", "async", "await", "export", "default", "new", "self",
})

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def _tokenise(text: str) -> list[str]:
    """Split text into lowercase tokens, filtering stop words."""
    return [
        t.lower()
        for t in _TOKEN_RE.findall(text)
        if t.lower() not in _STOP_WORDS and len(t) > 1
    ]


def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    avg_dl: float,
    df: dict[str, int],
    n_docs: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Okapi BM25 score for a single document (lower energy = better).

    Returns negative BM25 so that higher BM25 → lower energy score.
    """
    dl = len(doc_tokens)
    if dl == 0 or avg_dl == 0:
        return 1.0

    tf_map = Counter(doc_tokens)
    score = 0.0
    for qt in query_tokens:
        tf = tf_map.get(qt, 0)
        if tf == 0:
            continue
        doc_freq = df.get(qt, 0)
        idf = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
        score += idf * tf_norm

    # Normalise into [0, 1] range (approx.) and invert so lower = better
    if score == 0:
        return 1.0
    # Use sigmoid-like mapping: energy = 1 / (1 + score)
    return 1.0 / (1.0 + score)


def fallback_bm25_rank(
    query: str,
    candidates: list[RerankCandidate],
) -> list[RerankResult]:
    """Rank candidates using BM25 when LLM reranker is unavailable."""
    query_tokens = _tokenise(query)
    if not query_tokens:
        # Can't score without query tokens — return in original order
        return [
            RerankResult(id=c.id, score=i / max(len(candidates), 1), source="fallback_no_query")
            for i, c in enumerate(candidates)
        ]

    doc_tokens_list = [
        _tokenise(c.content + " " + c.metadata.get("path", ""))
        for c in candidates
    ]

    # Document frequency for IDF
    df: dict[str, int] = Counter()
    for dt in doc_tokens_list:
        seen = set(dt)
        for t in seen:
            df[t] += 1

    n_docs = len(candidates)
    avg_dl = sum(len(dt) for dt in doc_tokens_list) / max(n_docs, 1)

    results = []
    for cand, dt in zip(candidates, doc_tokens_list):
        score = _bm25_score(query_tokens, dt, avg_dl, df, n_docs)
        results.append(RerankResult(id=cand.id, score=score, source="fallback_bm25"))

    results.sort(key=lambda r: r.score)
    return results


# ── Cache helpers ────────────────────────────────────────────────────

def _make_cache_key(query: str, candidates: list[RerankCandidate]) -> str:
    """Deterministic cache key from query + candidate IDs."""
    q_norm = query.strip().lower()
    q_hash = hashlib.sha256(q_norm.encode("utf-8")).hexdigest()[:16]
    ids_sorted = ",".join(sorted(c.id for c in candidates))
    c_hash = hashlib.sha256(ids_sorted.encode("utf-8")).hexdigest()[:16]
    return f"rr:{q_hash}:{c_hash}"


class _SessionCache:
    """Simple in-memory LRU-ish cache for the current process."""

    def __init__(self, max_entries: int = 256) -> None:
        self._store: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._max = max_entries

    def get(self, key: str, ttl: int = SESSION_CACHE_TTL) -> Optional[list[dict[str, Any]]]:
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, data = entry
        if time.time() - ts > ttl:
            del self._store[key]
            return None
        return data

    def set(self, key: str, data: list[dict[str, Any]]) -> None:
        if len(self._store) >= self._max:
            # Evict oldest entry
            oldest_key = min(self._store, key=lambda k: self._store[k][0])
            del self._store[oldest_key]
        self._store[key] = (time.time(), data)

    def clear(self) -> None:
        self._store.clear()


# Module-level session cache (one per process)
_session_cache = _SessionCache()


# ── Main reranker class ──────────────────────────────────────────────

class CodeReranker:
    """Code-oriented reranker with LLM scoring, BM25 fallback, and caching.

    Usage::

        reranker = CodeReranker()
        results = await reranker.rerank(
            query="Where is the email validation logic?",
            candidates=[
                RerankCandidate(id="src/user/validation.ts:10-30", content="..."),
                ...
            ],
        )
        # results is a sorted list of RerankResult (lowest score = best)

    Attributes:
        k_retriever: Max candidates accepted from the retriever.
        k_rerank: How many top results to return after scoring.
        batch_size: Candidates per LLM call.
        max_parallel: Concurrent LLM batch calls.
        llm_timeout: Seconds before falling back to BM25.
    """

    def __init__(
        self,
        *,
        k_retriever: int = K_RETRIEVER,
        k_rerank: int = K_RERANK,
        batch_size: int = BATCH_SIZE,
        max_parallel: int = MAX_PARALLEL,
        llm_timeout: float = LLM_TIMEOUT,
        session_cache_ttl: int = SESSION_CACHE_TTL,
        global_cache_ttl: int = GLOBAL_CACHE_TTL,
        global_cache: Optional[Any] = None,  # RedisStore or InMemoryStore
    ) -> None:
        self.k_retriever = k_retriever
        self.k_rerank = k_rerank
        self.batch_size = batch_size
        self.max_parallel = max_parallel
        self.llm_timeout = llm_timeout
        self.session_cache_ttl = session_cache_ttl
        self.global_cache_ttl = global_cache_ttl
        self._global_cache = global_cache

    # ── Public API ───────────────────────────────────────────────────

    async def rerank(
        self,
        query: str,
        candidates: list[Union[RerankCandidate, dict[str, Any]]],
    ) -> list[RerankResult]:
        """Score and rank candidates against query.

        Args:
            query: Natural-language query, possibly mixed with code.
            candidates: Code fragments to rank.  Can be RerankCandidate
                instances or plain dicts with ``id``, ``content``,
                ``metadata`` keys.

        Returns:
            Sorted list of RerankResult (lowest score = most relevant),
            truncated to ``k_rerank`` entries.
        """
        start = time.time()

        # Normalise input
        cands = [
            c if isinstance(c, RerankCandidate) else RerankCandidate(**c)
            for c in candidates
        ]

        # Truncate to K_retriever
        cands = cands[: self.k_retriever]
        if not cands:
            return []

        # ── Check caches ────────────────────────────────────────────
        cache_key = _make_cache_key(query, cands)

        cached = _session_cache.get(cache_key, ttl=self.session_cache_ttl)
        if cached is not None:
            await logger.ainfo(
                "reranker_cache_hit",
                source="session",
                candidates=len(cands),
                latency_ms=round((time.time() - start) * 1000, 1),
            )
            return self._scores_to_results(cached, cands, source="cache_session")

        if self._global_cache is not None:
            try:
                g_cached = await self._global_cache.get(cache_key)
                if g_cached is not None:
                    _session_cache.set(cache_key, g_cached)
                    await logger.ainfo(
                        "reranker_cache_hit",
                        source="global",
                        candidates=len(cands),
                        latency_ms=round((time.time() - start) * 1000, 1),
                    )
                    return self._scores_to_results(g_cached, cands, source="cache_global")
            except Exception:
                pass  # global cache failure is non-fatal

        # ── Try LLM reranking ───────────────────────────────────────
        llm_available = self._check_llm_available()
        if llm_available:
            try:
                scores = await self._llm_rerank(query, cands)
                source = "reranked"
            except asyncio.TimeoutError:
                await logger.awarning("reranker_llm_timeout")
                scores = None
                source = "fallback_timeout"
            except Exception as exc:
                await logger.awarning("reranker_llm_error", error=str(exc)[:200])
                scores = None
                source = "fallback_error"
        else:
            scores = None
            source = "fallback_no_llm"

        # ── Fallback to BM25 ───────────────────────────────────────
        if scores is None:
            results = fallback_bm25_rank(query, cands)
            results = results[: self.k_rerank]
            await logger.ainfo(
                "reranker_fallback",
                source=source,
                candidates=len(cands),
                returned=len(results),
                latency_ms=round((time.time() - start) * 1000, 1),
            )
            # Still cache the fallback scores
            score_data = [{"id": r.id, "score": r.score} for r in results]
            _session_cache.set(cache_key, score_data)
            return results

        # ── Cache and return LLM results ────────────────────────────
        _session_cache.set(cache_key, scores)
        if self._global_cache is not None:
            try:
                await self._global_cache.set(
                    cache_key,
                    scores,
                    ttl=self.global_cache_ttl,
                )
            except Exception:
                pass

        results = self._scores_to_results(scores, cands, source="reranked")
        await logger.ainfo(
            "reranker_llm_success",
            candidates=len(cands),
            returned=len(results),
            latency_ms=round((time.time() - start) * 1000, 1),
        )
        return results

    # ── LLM reranking with batching ──────────────────────────────────

    async def _llm_rerank(
        self, query: str, candidates: list[RerankCandidate]
    ) -> list[dict[str, Any]]:
        """Call LLM reranker in batches, merge scores."""
        batches = [
            candidates[i : i + self.batch_size]
            for i in range(0, len(candidates), self.batch_size)
        ]

        sem = asyncio.Semaphore(self.max_parallel)

        async def _score_batch(batch: list[RerankCandidate]) -> list[dict[str, Any]]:
            async with sem:
                return await asyncio.wait_for(
                    self._call_llm_reranker(query, batch),
                    timeout=self.llm_timeout,
                )

        tasks = [asyncio.create_task(_score_batch(b)) for b in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results — if a batch failed, fall back to BM25 for that batch
        all_scores: list[dict[str, Any]] = []
        for batch, result in zip(batches, batch_results):
            if isinstance(result, Exception):
                await logger.awarning(
                    "reranker_batch_failed",
                    batch_size=len(batch),
                    error=str(result)[:120],
                )
                fb = fallback_bm25_rank("", batch)
                all_scores.extend({"id": r.id, "score": r.score} for r in fb)
            else:
                all_scores.extend(result)

        # Sort globally across all batches
        all_scores.sort(key=lambda x: x.get("score", 1.0))
        return all_scores[: self.k_rerank]

    async def _call_llm_reranker(
        self, query: str, batch: list[RerankCandidate]
    ) -> list[dict[str, Any]]:
        """Make a single LLM call to score a batch of candidates."""
        import os

        from src.models.schemas import LLMRequest, ModelTier

        payload = json.dumps(
            {
                "query": query,
                "candidates": [c.to_dict() for c in batch],
            },
            ensure_ascii=False,
        )

        request = LLMRequest(
            prompt=payload,
            system_prompt=RERANKER_SYSTEM_PROMPT,
            model_preference=ModelTier.FAST,
            max_tokens=1024,
            temperature=0.0,
        )

        # Try each provider in order (same pattern as agent_base)
        for provider_name, env_var in [
            ("claude", "AGENTARMY_CLAUDE_API_KEY"),
            ("openai", "AGENTARMY_OPENAI_API_KEY"),
            ("gemini", "AGENTARMY_GEMINI_API_KEY"),
            ("kimi", "AGENTARMY_KIMI_API_KEY"),
        ]:
            api_key = os.environ.get(env_var, "")
            if not api_key or api_key.startswith(("your_", "YOUR_")):
                continue

            try:
                client = self._create_client(provider_name, api_key)
                response = await client.complete(request)
                return self._parse_llm_response(response.content, batch)
            except Exception:
                continue

        # Ollama as last resort
        try:
            client = self._create_client("ollama", None)
            response = await client.complete(request)
            return self._parse_llm_response(response.content, batch)
        except Exception as exc:
            raise RuntimeError(f"All LLM providers failed for reranking: {exc}")

    @staticmethod
    def _create_client(provider_name: str, api_key: Optional[str]) -> Any:
        """Create an LLM client for the given provider."""
        if provider_name == "claude":
            from src.models.claude_client import ClaudeClient
            return ClaudeClient(api_key=api_key)
        elif provider_name == "openai":
            from src.models.openai_client import OpenAIClient
            return OpenAIClient(api_key=api_key)
        elif provider_name == "gemini":
            from src.models.gemini_client import GeminiClient
            return GeminiClient(api_key=api_key)
        elif provider_name == "kimi":
            from src.models.kimi_client import KimiClient
            return KimiClient(api_key=api_key)
        elif provider_name == "ollama":
            from src.models.ollama_client import OllamaClient
            return OllamaClient()
        raise ValueError(f"Unknown provider: {provider_name}")

    @staticmethod
    def _parse_llm_response(
        content: str,
        batch: list[RerankCandidate],
    ) -> list[dict[str, Any]]:
        """Parse LLM JSON response into score dicts.

        Robust to markdown fences, extra whitespace, partial JSON, etc.
        Falls back to assigning equal scores if parsing completely fails.
        """
        # Strip markdown code fences if present
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Try direct JSON parse
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [
                    {"id": item["id"], "score": float(item["score"])}
                    for item in data
                    if "id" in item and "score" in item
                ]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

        # Try extracting JSON array from within the text
        bracket_start = text.find("[")
        bracket_end = text.rfind("]")
        if bracket_start != -1 and bracket_end > bracket_start:
            try:
                data = json.loads(text[bracket_start : bracket_end + 1])
                if isinstance(data, list):
                    return [
                        {"id": item["id"], "score": float(item["score"])}
                        for item in data
                        if "id" in item and "score" in item
                    ]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass

        # Last resort: extract id/score pairs with regex
        results = []
        pattern = re.compile(
            r'"id"\s*:\s*"([^"]+)"[^}]*"score"\s*:\s*([\d.]+)',
            re.DOTALL,
        )
        for match in pattern.finditer(text):
            results.append({"id": match.group(1), "score": float(match.group(2))})

        if results:
            return results

        # Complete failure — assign equal scores (effectively no reranking)
        await_logger_msg = f"Failed to parse LLM reranker response ({len(content)} chars)"
        logger.warning("reranker_parse_failed", detail=await_logger_msg)
        return [{"id": c.id, "score": 0.5} for c in batch]

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _check_llm_available() -> bool:
        """Quick check if any LLM provider key is configured."""
        import os

        for env_var in [
            "AGENTARMY_CLAUDE_API_KEY",
            "AGENTARMY_OPENAI_API_KEY",
            "AGENTARMY_GEMINI_API_KEY",
            "AGENTARMY_KIMI_API_KEY",
        ]:
            key = os.environ.get(env_var, "")
            if key and not key.startswith(("your_", "YOUR_")):
                return True
        # Ollama doesn't need a key — check if we should include it
        # For performance reasons, only use Ollama fallback if no cloud providers
        return False  # Will still try Ollama in _call_llm_reranker as last resort

    def _scores_to_results(
        self,
        scores: list[dict[str, Any]],
        candidates: list[RerankCandidate],
        source: str = "reranked",
    ) -> list[RerankResult]:
        """Convert score dicts to sorted RerankResult list."""
        score_map = {s["id"]: s["score"] for s in scores}
        results = []
        for c in candidates:
            sc = score_map.get(c.id, 1.0)
            results.append(RerankResult(id=c.id, score=sc, source=source))
        results.sort(key=lambda r: r.score)
        return results[: self.k_rerank]

    def clear_session_cache(self) -> None:
        """Clear the in-process session cache."""
        _session_cache.clear()

    def get_config(self) -> dict[str, Any]:
        """Return current reranker configuration as a dict."""
        return {
            "k_retriever": self.k_retriever,
            "k_rerank": self.k_rerank,
            "batch_size": self.batch_size,
            "max_parallel": self.max_parallel,
            "llm_timeout": self.llm_timeout,
            "session_cache_ttl": self.session_cache_ttl,
            "global_cache_ttl": self.global_cache_ttl,
            "has_global_cache": self._global_cache is not None,
        }
