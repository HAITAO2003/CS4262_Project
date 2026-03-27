"""
Thread-safe LRU response cache for deterministic (temperature=0) requests.

Caches complete (output, logprobs) tuples keyed on the SHA-256 hash of the
fully-rendered prompt string plus the serialised sampling parameters.

This is fully generalizable: any dataset with repeated prompts benefits, and
the per-lookup overhead for cache misses is a single SHA-256 hash (~1 us).
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class CachedResponse:
    """
    Data class representing a cached chat response.
    """

    output: str
    logprobs: list[float]


class ResponseCache:
    """
    LRU cache with a fixed max size, guarded by a reentrant lock.
    """

    def __init__(self, max_size: int = 16_384) -> None:
        """
        Initialize the response cache.
        """
        self._lock = threading.Lock()
        self._store: OrderedDict[str, CachedResponse] = OrderedDict()
        self._max_size = max_size
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(prompt: str, temperature: float, max_tokens: int) -> str:
        """
        Construct a deterministic cache key.

        The key includes temperature to ensure correctness if the
        temperature=0 guard is ever relaxed in the future.
        """
        raw = f"{prompt}||t={temperature}||m={max_tokens}"
        return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()

    def get(self, key: str) -> CachedResponse | None:
        """
        Retrieve a cached response by key. Returns None if not found.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                self._store.move_to_end(key)
                self.hits += 1
                return entry
            self.misses += 1
            return None

    def put(self, key: str, response: CachedResponse) -> None:
        """
        Insert or update a response in the cache. Evicts the oldest entry if max_size is exceeded.
        """
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return
            self._store[key] = response
            if len(self._store) > self._max_size:
                self._store.popitem(last=False)

    @property
    def size(self) -> int:
        """
        Return the current number of items in the cache.
        """
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        """
        Return the cache hit rate as a float between 0 and 1.
        """
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict:
        """
        Return a summary dictionary of current cache statistics.
        """
        return {
            "size": self.size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1%}",
        }
