"""Content-addressed response cache.

Non-negotiable infrastructure: the grid will be re-run many times while
defenses, scoring and analysis change, and paying twice for an identical
request is pure waste. Keyed on everything that can change a response, so a
cache hit is genuinely the same call.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(".cache")


def request_key(
    *,
    model: str,
    payload: Any,
    temperature: float | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Stable hash of a request.

    `sort_keys=True` is load-bearing: dict ordering varies between runs and an
    unsorted dump would produce a fresh key for an identical request, silently
    disabling the cache and tripling the bill.
    """
    blob = json.dumps(
        {
            "model": model,
            "payload": payload,
            "temperature": temperature,
            "seed": seed,
            "extra": extra or {},
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, root: Path | str = DEFAULT_ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        # Shard by first two hex chars; a flat directory of 50k files is slow
        # on Windows and unpleasant to inspect.
        d = self.root / key[:2]
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        p = self._path(key)
        if not p.exists():
            self.misses += 1
            return None
        try:
            value = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A truncated file from an interrupted write is a miss, not a crash.
            self.misses += 1
            return None
        self.hits += 1
        return value

    def put(self, key: str, value: dict[str, Any]) -> None:
        p = self._path(key)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)  # atomic, so an interrupted run leaves no half-file

    @property
    def hit_rate(self) -> float | None:
        total = self.hits + self.misses
        return self.hits / total if total else None

    def __str__(self) -> str:
        hr = self.hit_rate
        return f"cache {self.hits}h/{self.misses}m" + (f" ({hr:.0%})" if hr is not None else "")
