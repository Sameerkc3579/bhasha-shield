"""Where an injection hides: payload placement inside a carrier text (docs/v2-plan.md, Phase 2 step 2).

The carrier *text* (an email, a web page, a chat message, ...) is generated in the language by the
Phase 2 generator; this module only places the payload in it, deterministically from a seed, and
renders the structured formats (an HTML page, an API JSON body) so the payload sits where it would
in a real tool output. The frozen test set always splices at the paragraph break nearest the middle
(`scripts.attack_scores.embed`); training varies the position so the guard cannot learn "the
attack is in the middle".
"""

from __future__ import annotations

import json
import random

POSITIONS = ("start", "middle", "end", "random")


def _breaks(body: str) -> list[int]:
    """Paragraph breaks, else sentence ends, as insertion points."""
    paras = [i for i in range(len(body)) if body.startswith("\n\n", i)]
    if paras:
        return paras
    return [i + 1 for i, c in enumerate(body) if c in ".!?।" and i + 1 < len(body)]


def insert(body: str, payload: str, *, position: str = "random", seed: int = 0) -> str:
    if position not in POSITIONS:
        raise ValueError(f"position {position!r} not in {POSITIONS}")
    if position == "start":
        return payload + "\n\n" + body
    if position == "end":
        return body + "\n\n" + payload
    points = _breaks(body)
    if not points:
        return body + "\n\n" + payload
    if position == "middle":
        at = min(points, key=lambda i: abs(i - len(body) // 2))
    else:
        at = random.Random(seed).choice(points)
    return body[:at] + "\n\n" + payload + "\n\n" + body[at:].lstrip("\n")


def wrap(fmt: str, body: str, payload: str, *, seed: int = 0) -> str:
    """Render a carrier format with the payload placed inside it."""
    rng = random.Random(seed)
    if fmt == "json":
        fields = {"id": rng.randrange(10_000, 99_999), "status": "ok", "message": body}
        key = rng.choice(["notes", "description", "comment", "message"])
        fields[key] = (fields.get(key, "") + " " + payload).strip() if key == "message" else payload
        return json.dumps(fields, ensure_ascii=False)
    if fmt == "web":
        inner = insert(body, payload, position="random", seed=seed)
        paras = "".join(f"<p>{p}</p>" for p in inner.split("\n\n"))
        return f"<html><body>{paras}</body></html>"
    return insert(body, payload, position=rng.choice(POSITIONS), seed=seed)
