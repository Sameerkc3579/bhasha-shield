"""Augmented copies of training items with v1's bounded adaptive rewrites (corpus/adaptive.py):
homoglyphs, zero-width characters, transliteration and script mixing (docs/v2-plan.md, Phase 2 step 7).

A copy keeps its source's label and split keys (goal, family, group), so it always lands on the
same side of train/validation as its source, and records the transform in `augmentations`.
Checker tokens (IBANs, UPI ids, emails, URLs, numbers) stay byte-identical, as in v1.
"""

from __future__ import annotations

from corpus.adaptive import apply as adapt
from corpus.adaptive import applicable
from corpus.guardset.schema import GuardItem


def apply(item: GuardItem, transform: str, *, seed: int = 0) -> GuardItem:
    if not applicable(transform, item.lang):
        raise ValueError(f"{transform} does not apply to {item.lang}")
    fields = item.model_dump(exclude={"item_id"})
    fields["text"] = adapt(transform, item.text, seed=seed)
    fields["augmentations"] = [*item.augmentations, transform]
    fields["provenance"] = {**item.provenance, "augmented_from": item.item_id, "augment_seed": str(seed)}
    return GuardItem.new(**fields)
