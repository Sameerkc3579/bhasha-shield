"""Shared machinery for the Class A text classifiers.

Every Class A defense is the same shape: score each tool output with a
sequence classifier, block above a threshold. What differs is the model and
which label means "injection", so the three wrappers are thin.

Design points that matter for the numbers:

- **The score is always recorded**, blocked or not. Thresholds are swept
  offline from the traces (Phase 6 calibration), so the grid runs each
  classifier once, not once per threshold.
- **Long outputs are chunked** and the maximum chunk score is used. These
  models see 512 tokens; a payload placed after a long benign prefix (an
  inbox, a file listing) would otherwise be truncated away -- a free bypass
  that would flatter the attack, not the defense.
- **Lazy loading.** transformers and the weights load on first use, so the
  module imports (and the tests run) on a machine with no torch. Run the real
  models on the T4, never through a paid API.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Sequence

from corpus.schema import ToolOutput
from defenses.base import DefenseClass, RunContext, Verdict

Scorer = Callable[[Sequence[str]], list[float]]

CHUNK_CHARS = 1500   # ~400-500 tokens for English; fewer chars per token in Indic
                     # scripts, so this stays under 512 tokens in every condition
CHUNK_OVERLAP = 200  # a payload split across a boundary still lands whole in one chunk


def chunks(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text or ""
    if len(text) <= size:
        return [text]
    step = size - overlap
    return [text[i : i + size] for i in range(0, len(text) - overlap, step)]


def hf_scorer(model_id: str, positive_label: str, *, revision: str | None = None,
              device: int | str | None = None, batch_size: int = 16) -> Scorer:
    """P(positive_label) per text, from a transformers text-classification pipeline."""
    state: dict = {}
    # The grid's worker threads share one pipeline; one model call at a time.
    lock = threading.Lock()

    def score(texts: Sequence[str]) -> list[float]:
        with lock:
            return _score(texts)

    def _score(texts: Sequence[str]) -> list[float]:
        if "pipe" not in state:
            from transformers import pipeline  # lazy: heavy, and absent on CPU dev boxes

            state["pipe"] = pipeline(
                "text-classification", model=model_id, revision=revision,
                top_k=None, truncation=True, max_length=512, device=device,
            )
        out = state["pipe"](list(texts), batch_size=batch_size)
        probs: list[float] = []
        for labels in out:
            by_label = {d["label"]: d["score"] for d in labels}
            if positive_label not in by_label:
                raise KeyError(
                    f"{model_id} has no label {positive_label!r}; it returned {sorted(by_label)}"
                )
            probs.append(float(by_label[positive_label]))
        return probs

    return score


class HFClassifier:
    """Block a tool output when P(injection) >= threshold."""

    name = "hf-classifier"
    klass: DefenseClass = "A"
    model_id = ""
    positive_label = "LABEL_1"

    def __init__(self, *, threshold: float = 0.5, scorer: Scorer | None = None,
                 model_id: str | None = None, revision: str | None = None,
                 device: int | str | None = None):
        self.threshold = threshold
        self.model_id = model_id or self.model_id
        self.revision = revision
        self.scorer = scorer or hf_scorer(self.model_id, self.positive_label,
                                          revision=revision, device=device)

    def inspect(self, out: ToolOutput, ctx: RunContext) -> Verdict:
        t0 = time.perf_counter()
        parts = chunks(out.content)
        s = max(self.scorer(parts)) if parts else 0.0
        return Verdict(
            block=s >= self.threshold,
            score=s,
            reason=f"P(injection)={s:.3f} over {len(parts)} chunk(s), threshold {self.threshold}",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
