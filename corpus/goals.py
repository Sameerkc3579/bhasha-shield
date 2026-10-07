"""Localized attacker goals for the Phase 2 capability probe.

The probe only needs the attacker's goal in each language -- not the user task,
not the payload -- and the 150 cases share 35 distinct goals. So the probe can
run in every language as soon as these 35 sentences exist, well before Phase 3
has localized the full corpus.

One row per (domain, injection_task) in `corpus/goals/{lang}.jsonl`. Every
string an AgentDojo checker matches on (emails, IBANs, URLs, quoted subjects and
bodies, ids) stays verbatim English; only the instruction around it is
translated. Otherwise a goal would fail because of the translation, which would
manufacture the very effect C(L) is measuring.

Review loop:

    python -m corpus.goals export hi          # -> corpus/goals/review_hi.csv (opens in Excel)
    # fix the `goal` column, put your name in `verified_by`
    python -m corpus.goals import hi          # writes the edits back, stamps verified_at
"""

from __future__ import annotations

import csv
import datetime
import json
import sys
from pathlib import Path

from pydantic import BaseModel

from corpus.provenance import record
from corpus.schema import InjectionCase

GOALS_DIR = Path("corpus/goals")


class LocalizedGoal(BaseModel):
    domain: str
    injection_task: str
    lang: str
    goal: str
    goal_en: str
    method: str
    verified_by: str | None = None
    verified_at: str | None = None
    notes: str = ""


def goals_path(lang: str) -> Path:
    return GOALS_DIR / f"{lang}.jsonl"


def load_goals(lang: str) -> dict[tuple[str, str], LocalizedGoal]:
    rows = [
        LocalizedGoal.model_validate_json(l)
        for l in goals_path(lang).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    return {(r.domain, r.injection_task): r for r in rows}


def localize_cases(
    cases: list[InjectionCase], lang: str, *, allow_unverified: bool = False
) -> list[InjectionCase]:
    """English cases with the attacker goal swapped for its `lang` version.

    Refuses unverified drafts unless asked: a C(L) measured on machine text is a
    pilot number, and it must not be mistaken for the real one.
    """
    goals = load_goals(lang)
    out: list[InjectionCase] = []
    for c in cases:
        key = (c.domain, (c.source_case_id or "").split("/")[-1])
        g = goals.get(key)
        if g is None:
            raise KeyError(f"no {lang} goal for {key}")
        if g.verified_by is None and not allow_unverified:
            raise PermissionError(
                f"{lang} goal {key} is an unverified draft. Review it "
                f"(python -m corpus.goals export {lang}) or pass --allow-unverified for a pilot run."
            )
        out.append(c.model_copy(update={"lang": lang, "attacker_goal_nl": g.goal}))
    return out


def all_verified(lang: str) -> bool:
    return all(g.verified_by for g in load_goals(lang).values())


# --------------------------------------------------------------------------
# Review round-trip through a spreadsheet
# --------------------------------------------------------------------------

FIELDS = ["domain", "injection_task", "goal_en", "goal", "verified_by", "notes"]


def export_csv(lang: str) -> Path:
    path = GOALS_DIR / f"review_{lang}.csv"
    # utf-8-sig so Excel shows Devanagari and Bengali instead of mojibake
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for g in load_goals(lang).values():
            w.writerow({k: getattr(g, k) or "" for k in FIELDS})
    return path


def import_csv(lang: str) -> tuple[int, int]:
    goals = load_goals(lang)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    changed = verified = 0
    with (GOALS_DIR / f"review_{lang}.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["domain"], row["injection_task"])
            g = goals[key]
            new_goal = row["goal"].replace("\r\n", "\n")
            by = (row.get("verified_by") or "").strip() or None
            update: dict = {}
            if new_goal != g.goal:
                update.update(goal=new_goal, method="machine+human" if g.method == "machine" else g.method)
                changed += 1
            if by and by != g.verified_by:
                update.update(verified_by=by, verified_at=now)
                verified += 1
            if update:
                goals[key] = g.model_copy(update=update)
                new = goals[key]
                record(f"goal/{key[0]}/{key[1]}", lang, new.method,
                       source_item_id=f"goal/{key[0]}/{key[1]}@en",
                       verified_by=new.verified_by, verified_at=new.verified_at,
                       notes="edited in review" if "goal" in update else "verified unchanged in review")
    goals_path(lang).write_text(
        "".join(g.model_dump_json() + "\n" for g in goals.values()), encoding="utf-8"
    )
    return changed, verified


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] not in ("export", "import"):
        print("usage: python -m corpus.goals export|import <lang>")
        return 2
    cmd, lang = argv
    if cmd == "export":
        print(f"wrote {export_csv(lang)}")
    else:
        changed, verified = import_csv(lang)
        print(f"{lang}: {changed} goals edited, {verified} newly verified; all verified: {all_verified(lang)}")
        if lang == "hi" and changed:
            print("hi changed -> hi_latn must be re-transliterated from the new hi text.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
