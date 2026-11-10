"""Train/validation split by attacker goal and template family (docs/v2-plan.md, Phase 2 step 8).

Each split key (goal, family, benign group) goes to validation with probability `val_frac`, by a
salted hash, so the assignment is stable across runs and machines. An injection is in validation
only if **both** its goal and its family are, in train only if both are; the cross items (a train
goal in a validation family, or the reverse) would leak one key across the split, so they are
**dropped and counted**. With val_frac = 0.2 that is about 4% validation, 64% train, 32% dropped
of the injections: generation is cheap, a leaky validation score is not.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from corpus.guardset.schema import GuardItem

SALT = "guard-2026"


def is_val(key: str, val_frac: float, salt: str = SALT) -> bool:
    h = int(hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < val_frac


@dataclass
class Split:
    train: list[GuardItem] = field(default_factory=list)
    validation: list[GuardItem] = field(default_factory=list)
    dropped: list[GuardItem] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {"train": len(self.train), "validation": len(self.validation), "dropped_cross": len(self.dropped)}


def group_split(items: list[GuardItem], *, val_frac: float = 0.2, salt: str = SALT) -> Split:
    s = Split()
    for it in sorted(items, key=lambda x: x.item_id):
        sides = {is_val(k, val_frac, salt) for k in it.split_keys()}
        if sides == {True}:
            s.validation.append(it)
        elif sides == {False}:
            s.train.append(it)
        else:
            s.dropped.append(it)
    return s
