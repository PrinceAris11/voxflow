"""test_cache.py — Unit tests for the in-memory LRU transcript cache."""

import time
from cache import TranscriptCache


def test_cache_hit_and_miss():
    cache = TranscriptCache(maxsize=10, default_ttl_seconds=60)
    assert cache.get("missing_key") is None

    cache.set("key1", "val1")
    assert cache.get("key1") == "val1"

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size"] == 1


def test_cache_lru_eviction():
    cache = TranscriptCache(maxsize=2, default_ttl_seconds=60)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)  # Evicts "a"

    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_cache_expiration():
    cache = TranscriptCache(maxsize=10, default_ttl_seconds=0.05)
    cache.set("short_lived", "content", ttl_seconds=0.05)
    assert cache.get("short_lived") == "content"

    time.sleep(0.08)
    assert cache.get("short_lived") is None
