"""
Thread-safe LRU response cache for deterministic (temperature=0) requests.

Caches complete (output, logprobs) tuples keyed on the rendered prompt string
and max token limit. In-flight requests reserve their key with a pending future
so concurrent identical requests share the same engine call.
"""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from dataclasses import dataclass

CacheKey = tuple[int, str]


@dataclass(frozen=True)
class CachedResponse:
    """
    Data class representing a cached chat response.
    """
    output: str
    logprobs: list[float]


CacheEntry = CachedResponse | asyncio.Future[CachedResponse]


class ResponseCache:
    """
    LRU cache with a fixed max size, guarded by a reentrant lock.
    """

    def __init__(self, max_size: int = 16_384) -> None:
        """
        Initialize the response cache.
        """
        self._lock = threading.Lock()
        self._store: OrderedDict[CacheKey, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(prompt: str, max_tokens: int) -> CacheKey:
        """
        Construct a deterministic cache key.
        """
        return (max_tokens, prompt)

    def get_or_reserve(
        self, key: CacheKey
    ) -> tuple[CachedResponse | None, asyncio.Future[CachedResponse] | None, bool]:
        """
        Retrieve a cached response by key or reserve it with a pending future.

        Returns (cached_response, pending_future, is_owner).
        """
        with self._lock:
            entry = self._store.get(key)
            if isinstance(entry, CachedResponse):
                self._store.move_to_end(key)
                self.hits += 1
                return entry, None, False
            if entry is not None:
                self._store.move_to_end(key)
                self.hits += 1
                return None, entry, False

            self.misses += 1
            pending = asyncio.get_running_loop().create_future()
            self._store[key] = pending
            self._evict_if_needed()
            return None, pending, True

    def put(self, key: CacheKey, response: CachedResponse) -> None:
        """
        Fill a pending cache entry with the completed response.
        """
        pending: asyncio.Future[CachedResponse] | None = None
        with self._lock:
            entry = self._store.get(key)
            if isinstance(entry, asyncio.Future):
                pending = entry
                self._store[key] = response
                self._store.move_to_end(key)
            else:
                self._store[key] = response
                self._store.move_to_end(key)
            self._evict_if_needed()

        if pending is not None and not pending.done():
            pending.set_result(response)

    def fail(self, key: CacheKey, exc: BaseException) -> None:
        """
        Remove a pending cache entry and wake any waiters with the failure.
        """
        pending: asyncio.Future[CachedResponse] | None = None
        with self._lock:
            entry = self._store.get(key)
            if isinstance(entry, asyncio.Future):
                pending = entry
                del self._store[key]

        if pending is not None and not pending.done():
            pending.set_exception(exc)

    def _evict_if_needed(self) -> None:
        """
        Evict the least recently used completed entry while preserving in-flight requests.
        """
        while len(self._store) > self._max_size:
            oldest_ready_key = None
            for candidate_key, candidate_entry in self._store.items():
                if not isinstance(candidate_entry, asyncio.Future):
                    oldest_ready_key = candidate_key
                    break
            if oldest_ready_key is None:
                break
            del self._store[oldest_ready_key]

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
