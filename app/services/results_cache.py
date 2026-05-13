from __future__ import annotations

import time
import uuid
from threading import Lock
from typing import Any

_MAX_ENTRIES = 256
_TTL_SECONDS = 60 * 60

_lock = Lock()
_store: dict[str, tuple[float, dict[str, Any]]] = {}


def _evict_unlocked(now: float) -> None:
    for key, (ts, _) in list(_store.items()):
        if now - ts > _TTL_SECONDS:
            _store.pop(key, None)
    while len(_store) > _MAX_ENTRIES:
        oldest = min(_store.items(), key=lambda kv: kv[1][0])[0]
        _store.pop(oldest, None)


def put(payload: dict[str, Any]) -> str:
    """Store a results payload; returns opaque result id (UUID string)."""
    rid = str(uuid.uuid4())
    now = time.time()
    with _lock:
        _evict_unlocked(now)
        _store[rid] = (now, payload)
    return rid


def get(rid: str) -> dict[str, Any] | None:
    now = time.time()
    with _lock:
        entry = _store.get(rid)
        if entry is None:
            return None
        ts, payload = entry
        if now - ts > _TTL_SECONDS:
            _store.pop(rid, None)
            return None
        return payload
