"""
Prompt bucketing, decode-budget estimation, and SLA-aware priority scoring.

Two hashes are maintained per request:
  1. Exact hash – SHA-256 of the full prompt string for detecting exact
     duplicates and hot repeated requests.
  2. Template hash – SHA-256 of a normalised version of the prompt
     (whitespace-collapsed, lowered, digits removed) to group semantically
     similar customer-support prompts into stable buckets.

Per-bucket statistics (median, p75, p95 output length, hit-rate) are
updated online after every completed request and feed into the priority
scoring function used by the vLLM priority scheduler.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field


_DIGIT_RE = re.compile(r"\d+")
_MULTI_WS_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """
    Collapse whitespace, convert to lowercase, and strip digits to create a stable template string.
    """
    text = text.lower()
    text = _DIGIT_RE.sub("", text)
    text = _MULTI_WS_RE.sub(" ", text).strip()
    return text


def _sha256(text: str) -> str:
    """
    Generate a SHA-256 hex digest for the given string.
    """
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


@dataclass
class BucketStats:
    """
    Running statistics for one template bucket.
    """

    output_lengths: list[int] = field(default_factory=list)
    hit_count: int = 0
    total_count: int = 0

    _dirty: bool = True
    _median: float = 128.0
    _p75: float = 192.0
    _p95: float = 256.0

    def record(self, output_length: int) -> None:
        """
        Record the output length of a completed request and mark statistics for recomputation.
        """
        self.output_lengths.append(output_length)
        self.hit_count += 1
        self._dirty = True

    def _recompute(self) -> None:
        """
        Lazily recalculate the median, p75, and p95 quantiles if the underlying data has changed.
        """
        if not self._dirty or not self.output_lengths:
            return
        s = sorted(self.output_lengths)
        n = len(s)
        self._median = float(s[n // 2])
        self._p75 = float(s[int(n * 0.75)])
        self._p95 = float(s[int(n * 0.95)])
        self._dirty = False

    @property
    def median(self) -> float:
        """
        Return the median output length.
        """
        self._recompute()
        return self._median

    @property
    def p75(self) -> float:
        """
        Return the 75th percentile output length.
        """
        self._recompute()
        return self._p75

    @property
    def p95(self) -> float:
        """
        Return the 95th percentile output length.
        """
        self._recompute()
        return self._p95

    @property
    def hit_rate(self) -> float:
        """
        Return the occurrence rate of this bucket relative to all processed requests.
        """
        return self.hit_count / max(self.total_count, 1)


class PromptAnalytics:
    """
    Thread-safe tracker for prompt analytics estimating request decode budgets.
    """

    def __init__(self) -> None:
        """
        Initialize the prompt analytics tracker.
        """
        self._lock = threading.Lock()
        self._buckets: dict[str, BucketStats] = defaultdict(BucketStats)
        self._exact_counts: dict[str, int] = defaultdict(int)
        self._total_requests: int = 0

    def get_hashes(self, prompt: str) -> tuple[str, str]:
        """
        Return a tuple containing the exact hash and the normalized template hash for a prompt.
        """
        return _sha256(prompt), _sha256(_normalise(prompt))

    def estimate_decode_budget(self, template_hash: str) -> float:
        """
        Estimate the expected output length (p75) for a specific template hash bucket.
        """
        with self._lock:
            bucket = self._buckets.get(template_hash)
        if bucket is None or not bucket.output_lengths:
            return 128.0
        return bucket.p75

    def estimate_cache_hit_likelihood(self, exact_hash: str) -> float:
        """
        Return a normalized score between 0.0 and 1.0 indicating cache hit probability.
        """
        with self._lock:
            count = self._exact_counts.get(exact_hash, 0)
        return min(count / 5.0, 1.0)

    def record_completion(
        self,
        exact_hash: str,
        template_hash: str,
        output_length: int,
    ) -> None:
        """
        Update runtime statistics with the actual output length generated for a completed request.
        """
        with self._lock:
            self._total_requests += 1
            self._exact_counts[exact_hash] += 1
            bucket = self._buckets[template_hash]
            bucket.record(output_length)
            bucket.total_count = self._total_requests

    def compute_priority(
        self,
        prompt: str,
        prompt_token_count: int,
    ) -> int:
        """
        Compute an integer priority score between 0 and 10 based on predicted computation costs.
        Lower values denote higher priority for the vLLM scheduler.
        """
        exact_hash, template_hash = self.get_hashes(prompt)

        if prompt_token_count <= 64:
            prompt_cost = 0
        elif prompt_token_count <= 256:
            prompt_cost = 1
        elif prompt_token_count <= 512:
            prompt_cost = 2
        elif prompt_token_count <= 1024:
            prompt_cost = 3
        else:
            prompt_cost = 4

        decode_budget = self.estimate_decode_budget(template_hash)
        if decode_budget <= 64:
            decode_cost = 0
        elif decode_budget <= 128:
            decode_cost = 1
        elif decode_budget <= 192:
            decode_cost = 2
        else:
            decode_cost = 3

        cache_likelihood = self.estimate_cache_hit_likelihood(exact_hash)
        cache_bonus = int(cache_likelihood * 3)

        raw = prompt_cost + decode_cost - cache_bonus
        return max(0, min(raw, 10))

    def stats_summary(self) -> dict:
        """
        Return a summary dictionary of current prompt tracking metrics.
        """
        with self._lock:
            return {
                "total_requests": self._total_requests,
                "unique_templates": len(self._buckets),
                "unique_exact_prompts": len(self._exact_counts),
            }
