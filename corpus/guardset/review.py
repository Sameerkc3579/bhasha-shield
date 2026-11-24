"""The fixed-seed 5% sample check of the guard's training set (docs/v2-plan.md, Phase 2 step 9, Gate G2).

    python -m corpus.guardset.review export hi        # -> corpus/guardset/review/packs/sample_hi.csv (Excel)
    python -m corpus.guardset.review import hi --reviewer R1
    python -m corpus.guardset.review status

The sample is per language, from the whole build (`corpus/guardset/out/`, train + validation,
augmented copies included): the items with the lowest salted hash, 5% of them and at least 20.
The same build always gives the same sample. The reviewer marks every row `ok` or `wrong`:
**wrong** = the label is wrong (an "injection" that asks for nothing, a "benign" text that does), or
the text is not usable as that language (wrong language or script, garbled, cut off, or chatter
such as "Here is the translation"). Small style issues are `ok`; say so in `notes`.

Verdicts (`review/verdicts/{lang}.jsonl`: item id, text hash, verdict, reviewer code, date, the
build's report hash) are committed; the sheets carry dataset text and are gitignored. A verdict
counts only for the build it was given on: a rebuild makes it stale. A language passes at an
error rate of 10% or less (Gate G2); the Wilson interval is reported next to it.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

from corpus.guardset.schema import GuardItem

OUT = Path("corpus/guardset/out")
REVIEW = Path("corpus/guardset/review")
PACKS = REVIEW / "packs"
VERDICTS = REVIEW / "verdicts"
SHARE, FLOOR, SALT = 0.05, 20, "g2-sample-2026"
MAX_ERROR = 0.10
FIELDS = ("item_id", "label", "adversarial", "family", "carrier", "english", "text", "verdict", "notes")


def _sha(s: str | bytes) -> str:
    return hashlib.sha256(s.encode("utf-8") if isinstance(s, str) else s).hexdigest()


def _u(item_id: str) -> str:
    return _sha(f"{SALT}:{item_id}")


def report_sha() -> str:
    return _sha((OUT / "report.json").read_bytes())


def load_build() -> dict[str, list[GuardItem]]:
    by: dict[str, list[GuardItem]] = {}
    for name in ("train", "validation"):
        p = OUT / f"{name}.jsonl"
        if not p.exists():
            raise SystemExit(f"{p} missing: build first (python -m corpus.guardset.build)")
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                it = GuardItem.model_validate_json(line)
                by.setdefault(it.lang, []).append(it)
    return dict(sorted(by.items()))


def sample(items: list[GuardItem]) -> list[GuardItem]:
    n = min(len(items), max(math.ceil(SHARE * len(items)), FLOOR))
    return sorted(sorted(items, key=lambda i: _u(i.item_id))[:n], key=lambda i: i.item_id)


def _english(it: GuardItem) -> str:
    """What the reviewer compares against: the English seed of a payload, the topic of a benign text."""
    from corpus.guardset import generate, seed

    p = it.provenance
    if it.label == "injection" and p.get("seed_id"):
        if not hasattr(_english, "seeds"):
            _english.seeds = {s.seed_id: s.text for s in seed.payloads(seed.load_goals(), seed.load_templates())}
        return _english.seeds.get(p["seed_id"], "")
    if p.get("key", "").count("|") == 2:
        t = int(p["key"].split("|")[1])
        topics = generate.ADV_SEEDS if it.adversarial_benign else generate.TOPICS
        return f"(topic) {topics[t]}" if t < len(topics) else ""
    return ""


def pack(lang: str) -> Path:
    return PACKS / f"sample_{lang}.csv"


def export(lang: str, *, force: bool = False) -> Path:
    rows = sample(load_build().get(lang, []))
    if not rows:
        raise SystemExit(f"no {lang} items in {OUT}")
    path = pack(lang)
    if path.exists() and not force:
        with path.open(encoding="utf-8-sig", newline="") as f:
            if any(r.get("verdict", "").strip() for r in csv.DictReader(f)):
                raise SystemExit(f"{path} already has verdicts: import it, or pass --force to start over")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:   # utf-8-sig: Excel shows Devanagari
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for it in rows:
            w.writerow({"item_id": it.item_id, "label": it.label, "adversarial": "yes" if it.adversarial_benign else "",
                        "family": it.family or "", "carrier": it.carrier, "english": _english(it), "text": it.text,
                        "verdict": "", "notes": ""})
    return path


def import_(lang: str, *, reviewer: str) -> Path:
    from corpus.testset import reviewer_codes

    codes = reviewer_codes()
    if reviewer not in codes or codes[reviewer] == "AgentDojo":
        raise SystemExit(f"unknown reviewer {reviewer!r}: sign with a code from corpus/reviewers.json")
    code = codes[reviewer]
    want = {i.item_id: i for i in sample(load_build().get(lang, []))}
    with pack(lang).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    got = {r["item_id"]: r for r in rows}
    problems = [f"{i}: not in the sheet" for i in sorted(set(want) - set(got))]
    problems += [f"{i}: not in the sample of this build" for i in sorted(set(got) - set(want))]
    for iid, r in got.items():
        if iid not in want:
            continue
        if r["text"].replace("\r\n", "\n") != want[iid].text.replace("\r\n", "\n"):
            problems.append(f"{iid}: the text was edited (mark it wrong instead and say why in notes)")
        if r["verdict"].strip().lower() not in ("ok", "wrong"):
            problems.append(f"{iid}: verdict must be ok or wrong, not {r['verdict']!r}")
    if problems:
        raise SystemExit("not imported:\n  " + "\n  ".join(problems[:40]))
    today, rsha = dt.date.today().isoformat(), report_sha()
    out = VERDICTS / f"{lang}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps({"item_id": iid, "sha": _sha(want[iid].text), "verdict": got[iid]["verdict"].strip().lower(),
                                       "reviewer": code, "date": today, "report_sha": rsha,
                                       "notes": got[iid]["notes"].strip()}, ensure_ascii=False) + "\n"
                           for iid in sorted(want)), encoding="utf-8", newline="\n")
    return out


def _wilson(k: int, n: int) -> list[float | None]:
    if not n:
        return [None, None]
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def status() -> dict:
    build, rsha, out = load_build(), report_sha(), {}
    for lang, items in build.items():
        want = {i.item_id: _sha(i.text) for i in sample(items)}
        p = VERDICTS / f"{lang}.jsonl"
        v = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []
        fresh = [r for r in v if r["report_sha"] == rsha and want.get(r["item_id"]) == r["sha"]]
        wrong = sum(r["verdict"] == "wrong" for r in fresh)
        n = len(fresh)
        rate = wrong / n if n else None
        out[lang] = {"sample": len(want), "reviewed": n, "stale": len(v) - n, "wrong": wrong, "error_rate": rate,
                     "error_ci": _wilson(wrong, n), "reviewers": sorted({r["reviewer"] for r in fresh}),
                     "passes": n == len(want) and rate is not None and rate <= MAX_ERROR}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["export", "import", "status"])
    ap.add_argument("langs", nargs="*")
    ap.add_argument("--reviewer")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.action == "export":
        for lang in a.langs or load_build():
            print(f"{lang}: {export(lang, force=a.force)}")
    elif a.action == "import":
        if not a.reviewer or not a.langs:
            ap.error("import needs the language(s) and --reviewer <code>")
        for lang in a.langs:
            print(f"{lang}: {import_(lang, reviewer=a.reviewer)}")
    st = status()
    print(f"{'lang':<9}{'sample':>7}{'reviewed':>9}{'wrong':>6}{'error':>7}  {'95% CI':<13}{'stale':>6}  G2")
    for lang, s in st.items():
        e = "-" if s["error_rate"] is None else f"{s['error_rate']:.0%}"
        ci = "" if s["error_ci"][0] is None else f"{s['error_ci'][0]:.0%}-{s['error_ci'][1]:.0%}"
        print(f"{lang:<9}{s['sample']:>7}{s['reviewed']:>9}{s['wrong']:>6}{e:>7}  {ci:<13}{s['stale']:>6}  "
              f"{'pass' if s['passes'] else '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
