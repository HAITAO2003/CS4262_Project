"""
Prompt bucketing and decode-budget estimation.

Two hashes are maintained per request:
  1. Exact hash – SHA-256 of the full prompt string for detecting exact
     duplicates and hot repeated requests.
  2. Template hash – SHA-256 of a normalised version of the prompt
     (whitespace-collapsed, lowered, digits removed) to group semantically
     similar customer-support prompts into stable buckets.

Per-bucket statistics (median, p75, p95 output length) are updated online
after every completed request and feed into dynamic max_tokens capping.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field


_DIGIT_RE = re.compile(r"\d+")
_MULTI_WS_RE = re.compile(r"\s+")
MAX_BUCKET_HISTORY = 500


def _normalise(text: str) -> str:
    text = text.lower()
    text = _DIGIT_RE.sub("", text)
    text = _MULTI_WS_RE.sub(" ", text).strip()
    return text


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


@dataclass
class BucketStats:
    output_lengths: list[int] = field(default_factory=list)
    hit_count: int = 0
    total_count: int = 0

    _dirty: bool = True
    _median: float = 128.0
    _p75: float = 192.0
    _p95: float = 256.0

    def record(self, output_length: int) -> None:
        self.output_lengths.append(output_length)
        # Keep only the last MAX_BUCKET_HISTORY entries to bound sort cost
        if len(self.output_lengths) > MAX_BUCKET_HISTORY:
            self.output_lengths = self.output_lengths[-MAX_BUCKET_HISTORY:]
        self.hit_count += 1
        self._dirty = True

    def _recompute(self) -> None:
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
        self._recompute()
        return self._median

    @property
    def p75(self) -> float:
        self._recompute()
        return self._p75

    @property
    def p95(self) -> float:
        self._recompute()
        return self._p95


class PromptAnalytics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: dict[str, BucketStats] = defaultdict(BucketStats)
        self._exact_counts: dict[str, int] = defaultdict(int)
        self._total_requests: int = 0

    def get_hashes(self, prompt: str) -> tuple[str, str]:
        return _sha256(prompt), _sha256(_normalise(prompt))

    def estimate_decode_budget(self, template_hash: str) -> float:
        with self._lock:
            bucket = self._buckets.get(template_hash)
        if bucket is None or not bucket.output_lengths:
            return 128.0
        return bucket.p75

    def record_completion(
        self,
        exact_hash: str,
        template_hash: str,
        output_length: int,
    ) -> None:
        with self._lock:
            self._total_requests += 1
            self._exact_counts[exact_hash] += 1
            bucket = self._buckets[template_hash]
            bucket.record(output_length)
            bucket.total_count = self._total_requests

    def stats_summary(self) -> dict:
        with self._lock:
            return {
                "total_requests": self._total_requests,
                "unique_templates": len(self._buckets),
                "unique_exact_prompts": len(self._exact_counts),
            }
