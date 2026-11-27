"""Train/validation split by template family and benign group (docs/v2-plan.md, Phase 2 step 8).

For each kind of split key (`family`, `group`, and `goal` if asked for), exactly
round(val_frac x number of keys) keys go to validation: the ones with the lowest salted hash, so the
choice is stable across runs and machines. (A threshold per key was tried first: with only 16 template
families it put 1 of them in validation at 0.2, about 1% of the attacks.)

Default `by=("family", "group")`, decided on 2026-11-27: an injection is in validation when its
template family is, and goals may appear on both sides. Validation then measures unseen templates and
keeps every item; the frozen test set, not validation, carries the goal separation (no training goal is
a test goal). With `by=("goal", "family", "group")` an injection needs both keys on one side, and the
cross items are dropped and counted (about 40% of the attacks at 0.25).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from corpus.guardset.schema import GuardItem

SALT = "guard-2026"


def _h(key: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()


def held_out(items: list[GuardItem], val_frac: float, salt: str = SALT,
             by: tuple[str, ...] = ("family", "group")) -> dict[str, set[str]]:
    """kind -> the keys of that kind that go to validation (an exact share, lowest hash first)."""
    keys: dict[str, set[str]] = {}
    for it in items:
        for k in it.split_keys():
            kind = k.split(":", 1)[0]
            if kind in by:
                keys.setdefault(kind, set()).add(k)
    return {kind: set(sorted(ks, key=lambda k: _h(k, salt))[:round(val_frac * len(ks))])
            for kind, ks in keys.items()}


@dataclass
class Split:
    train: list[GuardItem] = field(default_factory=list)
    validation: list[GuardItem] = field(default_factory=list)
    dropped: list[GuardItem] = field(default_factory=list)
    held_out: dict[str, list[str]] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        return {"train": len(self.train), "validation": len(self.validation), "dropped_cross": len(self.dropped)}


def group_split(items: list[GuardItem], *, val_frac: float = 0.25, salt: str = SALT,
                by: tuple[str, ...] = ("family", "group")) -> Split:
    val = held_out(items, val_frac, salt, by)
    s = Split(held_out={kind: sorted(ks) for kind, ks in sorted(val.items())})
    for it in sorted(items, key=lambda x: x.item_id):
        sides = {k in val[k.split(":", 1)[0]] for k in it.split_keys() if k.split(":", 1)[0] in by}
        if sides == {True}:
            s.validation.append(it)
        elif sides <= {False}:
            s.train.append(it)
        else:
            s.dropped.append(it)
    return s
