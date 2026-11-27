"""TF-IDF leakage audit: training text against the frozen test set (docs/v2-plan.md, Phase 2 step 8).

For every training item, the highest cosine similarity to any test item **in any language** (a
Romanized Hindi training text can copy a code-mixed test item; different scripts share almost no
n-grams, so crossing languages costs nothing), on TF-IDF over character 3-5-grams (word boundaries kept; robust to Devanagari, Romanized Hindi and
code-mixing alike, where word tokenisers disagree). Anything at or above `threshold` is dropped and
listed. Gate G2: after the audit, zero training items are above the threshold.

The IDF weights come from the **test set only**. Then an item's similarity depends on its own text
and the test set, never on which other training items exist: re-auditing the kept items drops
nothing, and a regenerated training set is audited on the same scale.

The threshold is a choice, not a fact: the report keeps a histogram of the maximum similarities so
the cut can be defended (or moved) at G2 with the distribution in view.
"""

from __future__ import annotations

from collections import Counter

from corpus.guardset.schema import GuardItem

THRESHOLD = 0.6
BLOCK = 1000


def _max_similarity(train: list[str], test: list[str]) -> list[tuple[float, int]]:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

    counts = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True).fit(train + test)
    ta, tb = counts.transform(train), counts.transform(test)
    idf = TfidfTransformer(sublinear_tf=True).fit(tb)    # IDF from the test set only
    a, b = idf.transform(ta), idf.transform(tb)          # rows are L2-normalised: dot = cosine
    out = []
    for start in range(0, a.shape[0], BLOCK):
        sims = (a[start:start + BLOCK] @ b.T).toarray()
        for row in sims:
            j = int(row.argmax())
            out.append((float(row[j]), j))
    return out


def audit(items: list[GuardItem], test: dict[str, list[str]] | list[str], *,
          threshold: float = THRESHOLD) -> tuple[list[GuardItem], dict]:
    """(kept, report). `test`: test texts, or lang -> test texts (all languages are pooled)."""
    tests = [t for ts in test.values() for t in ts] if isinstance(test, dict) else list(test)
    kept, dropped, max_sim, hist = [], [], {}, Counter()
    sims = _max_similarity([m.text for m in items], tests) if tests and items else [(0.0, -1)] * len(items)
    for it, (s, j) in zip(items, sims):
        max_sim[it.item_id] = round(s, 4)
        hist[f"{min(int(s * 10), 9) / 10:.1f}"] += 1
        if s >= threshold:
            dropped.append({"item_id": it.item_id, "lang": it.lang, "similarity": round(s, 4),
                            "test_text": tests[j][:120]})
        else:
            kept.append(it)
    report = {"threshold": threshold, "checked": len(items), "dropped": len(dropped),
              "dropped_ids": [d["item_id"] for d in dropped], "dropped_rows": dropped,
              "max_similarity": max_sim, "histogram": dict(sorted(hist.items()))}
    return kept, report
