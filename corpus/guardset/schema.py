"""GuardItem: one labelled training text for the guard, and Exclusions: what it may never share
with the frozen test set (docs/v2-plan.md, Phase 2 step 1).

Split keys. An injection is grouped by its attacker **goal** and its template **family**; a benign
item by its **group** (the seed or source document it came from, so paraphrases of one text stay
on one side). `corpus/guardset/split.py` keeps every key on one side of train/validation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

CARRIERS = ("email", "web", "document", "chat", "sms", "calendar", "review", "json")


def norm_family(name: str) -> str:
    """'Important-Instructions' and 'important_instructions' are one family."""
    return re.sub(r"[\s_\-]+", "_", name.strip().lower())


class GuardItem(BaseModel):
    item_id: str
    text: str
    label: Literal["injection", "benign"]
    lang: str
    family: str | None = None          # template family (injection) or benign style
    goal: str | None = None            # attacker goal key; never a test-set goal
    group: str | None = None           # benign: seed / source document
    carrier: str
    adversarial_benign: bool = False
    source: str                        # "template", "generated:<model id>", "indictrans2:<model id>", ...
    augmentations: list[str] = []
    provenance: dict[str, str] = {}    # model ids, seeds, prompt hashes

    @model_validator(mode="after")
    def _rules(self) -> GuardItem:
        if self.label == "injection" and not (self.goal and self.family):
            raise ValueError(f"{self.item_id}: an injection needs a goal and a template family")
        if self.label == "benign" and self.goal:
            raise ValueError(f"{self.item_id}: a benign item has no attacker goal")
        if self.carrier not in CARRIERS:
            raise ValueError(f"{self.item_id}: carrier {self.carrier!r} not in {CARRIERS}")
        return self

    @classmethod
    def new(cls, **kw) -> GuardItem:
        """Content-addressed id: the same text, label and keys always get the same id."""
        key = json.dumps([kw.get(k) for k in ("lang", "label", "family", "goal", "group", "carrier", "text")]
                         + [kw.get("augmentations", [])], ensure_ascii=False)
        kw["item_id"] = f"gs_{kw['lang']}_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]}"
        return cls(**kw)

    def split_keys(self) -> tuple[str, ...]:
        if self.label == "injection":
            return (f"goal:{self.goal}", f"family:{norm_family(self.family or '')}")
        return (f"group:{self.group or self.item_id}",)


@dataclass
class Exclusions:
    """Goals and template families of the frozen test set (its manifest): training may use none."""

    goals: set[str] = field(default_factory=set)
    families: set[str] = field(default_factory=set)

    @classmethod
    def from_manifest(cls, path: Path) -> Exclusions:
        m = json.loads(Path(path).read_text(encoding="utf-8"))
        goals = {g for half in m["goals"].values() for g in half}
        return cls(goals=goals, families=set(m["families"]))

    def violations(self, items: list[GuardItem]) -> list[str]:
        """A clash is a generator bug, so it is refused (reported), never silently dropped."""
        fams = {norm_family(f) for f in self.families}
        out = []
        for it in items:
            if it.goal and it.goal in self.goals:
                out.append(f"{it.item_id}: goal {it.goal} is a test-set goal")
            if it.label == "injection" and it.family and norm_family(it.family) in fams:
                out.append(f"{it.item_id}: family {it.family} is a test-set template family")
        return out
