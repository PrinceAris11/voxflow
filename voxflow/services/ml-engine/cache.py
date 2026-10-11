"""cache.py — In-memory LRU cache with TTL for VoxFlow transcripts.

Eliminates redundant LLM API calls and regex computations for frequent
or duplicate voice utterances.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from threading import Lock
from typing import Any


class TranscriptCache:
    """Thread-safe LRU cache with time-to-live expiration."""

    def __init__(self, maxsize: int = 1024, default_ttl_seconds: float = 3600.0):
        self._maxsize = maxsize
        self._default_ttl = default_ttl_seconds
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        now = time.time()
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            val, expiry = self._cache[key]
            if expiry < now:
                # Expired
                del self._cache[key]
                self._misses += 1
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return val

    def set(self, key: str, value: Any, ttl_seconds: float | None = None) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        expiry = time.time() + ttl
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, expiry)
            if len(self._cache) > self._maxsize:
                # Evict oldest entry
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": round(self._hits / (self._hits + self._misses), 4) if (self._hits + self._misses) > 0 else 0.0
            }


# Singleton cache instance
transcript_cache = TranscriptCache(maxsize=1024, default_ttl_seconds=3600)
