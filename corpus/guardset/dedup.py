"""Near-duplicate removal (docs/v2-plan.md, Phase 2 step 8).

Exact Jaccard similarity of character 5-gram sets -- the quantity MinHash estimates -- computed
with sparse matrices. At this scale (a few thousand items per language) the exact value is cheap,
so there is no estimation noise to explain. Items are compared within one (language, label):
a benign security notice that quotes an attack is meant to sit next to that attack.

The kept item of a cluster is the first by item id, so the result does not depend on input order.
"""

from __future__ import annotations

from corpus.guardset.schema import GuardItem

NGRAM = 5
THRESHOLD = 0.8
BLOCK = 2000


def _jaccard_pairs(texts: list[str], threshold: float) -> list[tuple[int, int, float]]:
    from sklearn.feature_extraction.text import CountVectorizer

    if len(texts) < 2:
        return []
    x = CountVectorizer(analyzer="char", ngram_range=(NGRAM, NGRAM), binary=True,
                        lowercase=False).fit_transform(texts).astype("float32").tocsr()
    sizes = x.sum(axis=1).A1
    pairs = []
    for start in range(0, x.shape[0], BLOCK):
        inter = (x[start:start + BLOCK] @ x.T).tocoo()
        for i, j, v in zip(inter.row, inter.col, inter.data):
            i = int(i) + start
            j = int(j)
            if j <= i:
                continue
            jac = v / (sizes[i] + sizes[j] - v)
            if jac >= threshold:
                pairs.append((i, j, float(jac)))
    return pairs


def dedup(items: list[GuardItem], *, threshold: float = THRESHOLD) -> tuple[list[GuardItem], list[dict]]:
    """(kept, removed); each removed row names the kept item it duplicates."""
    groups: dict[tuple[str, str], list[GuardItem]] = {}
    for it in sorted(items, key=lambda x: x.item_id):
        groups.setdefault((it.lang, it.label), []).append(it)
    kept, removed = [], []
    for members in groups.values():
        parent = list(range(len(members)))

        def root(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        best: dict[int, float] = {}
        for i, j, jac in _jaccard_pairs([m.text for m in members], threshold):
            ri, rj = root(i), root(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)    # the smaller index (first id) is the root
            best[j] = max(best.get(j, 0.0), jac)
        for i, m in enumerate(members):
            r = root(i)
            if r == i:
                kept.append(m)
            else:
                removed.append({"item_id": m.item_id, "kept": members[r].item_id, "jaccard": round(best.get(i, 0.0), 4)})
    return kept, removed
