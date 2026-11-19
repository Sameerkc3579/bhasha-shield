"""English seeds for the guard's training data (docs/v2-plan.md, Phase 2 steps 3-4).

    python -m corpus.guardset.seed check        # goals + families against the frozen test set
    python -m corpus.guardset.seed export       # the payload grid -> corpus/guardset/drafts/seed_en.jsonl

`seed/goals_en.txt` holds the new attacker goals (`=== guard/<domain>_NNN | domain: d | type: t`,
then one imperative sentence) and `seed/families_en.txt` the new template families
(`=== <id> | family: f`, then a template with exactly one `{goal}`). Both are hand-reviewable text.
`check` refuses any goal or family that the frozen test set uses (its manifest), by base family name,
so `native/hi/bank_alert` in the test set also blocks a training family called `bank_alert`.

The payload grid pairs every goal with `per_goal` templates from distinct families, chosen by hash,
so it is the same on every machine. These English payloads are the input of the Kaggle localization
notebook (IndicTrans2 draft, then an LLM rewrite); carriers and augmentation come after, in `build`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from corpus.guardset.schema import Exclusions, GuardItem, base_family

SEED = Path(__file__).parent / "seed"
GOALS = SEED / "goals_en.txt"
FAMILIES = SEED / "families_en.txt"
MANIFEST = Path("corpus/testset/v2/manifest.json")
EXPORT = Path("corpus/guardset/drafts/seed_en.jsonl")

DOMAINS = ("banking", "travel", "email", "workspace", "shopping", "gov", "messaging")
TYPES = ("transaction", "exfiltration", "tampering", "destructive", "disclosure")
PER_GOAL = 6

HEADER = re.compile(r"^=== (\S+)(.*)$")
SLOT = re.compile(r"\{[^{}\s]*\}")


@dataclass(frozen=True)
class Goal:
    id: str
    domain: str
    type: str
    text: str


@dataclass(frozen=True)
class Template:
    id: str
    family: str
    text: str


@dataclass(frozen=True)
class Payload:
    seed_id: str
    goal: str
    family: str
    template: str
    domain: str
    type: str
    text: str


def _blocks(text: str) -> tuple[list[tuple[str, dict[str, str], str, int]], list[str]]:
    out, problems, cur, buf = [], [], None, []

    def close():
        if cur is not None:
            out.append((cur[0], cur[1], "\n".join(buf).strip(), cur[2]))

    for n, line in enumerate(text.splitlines(), 1):
        m = HEADER.match(line)
        if m:
            close()
            meta = {k.strip(): v.strip() for k, v in (p.split(":", 1) for p in m.group(2).split("|") if ":" in p)}
            cur, buf = (m.group(1), meta, n), []
        elif cur is None:
            if line.strip():
                problems.append(f"line {n}: text before the first '=== id' header")
        else:
            buf.append(line)
    close()
    seen: dict[str, int] = {}
    for bid, _, body, n in out:
        if bid in seen:
            problems.append(f"{bid} (line {n}): id appears twice (first at line {seen[bid]})")
        seen.setdefault(bid, n)
        if not body:
            problems.append(f"{bid} (line {n}): no text")
    return out, problems


def parse_goals(text: str) -> tuple[list[Goal], list[str]]:
    blocks, problems = _blocks(text)
    return [Goal(b, m.get("domain", ""), m.get("type", ""), body) for b, m, body, _ in blocks], problems


def parse_templates(text: str) -> tuple[list[Template], list[str]]:
    blocks, problems = _blocks(text)
    return [Template(b, m.get("family", ""), body) for b, m, body, _ in blocks], problems


def load_goals(path: Path = GOALS) -> list[Goal]:
    return parse_goals(path.read_text(encoding="utf-8"))[0]


def load_templates(path: Path = FAMILIES) -> list[Template]:
    return parse_templates(path.read_text(encoding="utf-8"))[0]


def check(goals: list[Goal], templates: list[Template], exclusions: Exclusions) -> list[str]:
    problems = []
    test_families = {base_family(f) for f in exclusions.families}
    texts: dict[str, str] = {}
    for g in goals:
        if not g.id.startswith("guard/"):
            problems.append(f"{g.id}: goal ids start with guard/ (test goals are AgentDojo's)")
        if g.id in exclusions.goals:
            problems.append(f"{g.id}: is a test-set goal")
        if g.domain not in DOMAINS:
            problems.append(f"{g.id}: domain {g.domain!r} not in {DOMAINS}")
        if g.type not in TYPES:
            problems.append(f"{g.id}: type {g.type!r} not in {TYPES}")
        if SLOT.search(g.text):
            problems.append(f"{g.id}: a goal has no placeholders")
        if g.text in texts:
            problems.append(f"{g.id}: same text as {texts[g.text]}")
        texts.setdefault(g.text, g.id)
    for t in templates:
        slots = SLOT.findall(t.text)
        if slots.count("{goal}") != 1 or len(slots) != 1:
            problems.append(f"{t.id}: needs exactly one {{goal}} and no other placeholder (found {slots})")
        if not t.family:
            problems.append(f"{t.id}: header needs '| family: <name>'")
        elif base_family(t.family) in test_families:
            problems.append(f"{t.id}: family {t.family} is a test-set template family")
        if t.text in texts:
            problems.append(f"{t.id}: same text as {texts[t.text]}")
        texts.setdefault(t.text, t.id)
    return problems


def fill(template: str, goal: str) -> str:
    """Put a goal sentence into a template: lowercase its first word mid-sentence, and drop its
    full stop when the template's sentence goes on (or a quote closes) right after it."""
    i = template.index("{goal}")
    before, after = template[:i], template[i + len("{goal}"):]
    start = before.rstrip(" \t")
    sentence_start = not start or start[-1] in ".:\n>-=[(\"'"
    g = goal
    first = g.split(" ", 1)[0]
    if not sentence_start and first[:1].isupper() and not first[1:2].isupper():
        g = g[0].lower() + g[1:]
    if g.endswith("."):
        nxt = after[:1]
        if nxt in ('"', "'", ")", "*") or (nxt == " " and after[1:2].islower()):
            g = g[:-1]
    return before + g + after


def _h(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12], 16)


def payloads(goals: list[Goal], templates: list[Template], *, per_goal: int = PER_GOAL) -> list[Payload]:
    """Each goal x `per_goal` distinct families (by hash), one template per family (by hash)."""
    by_family: dict[str, list[Template]] = {}
    for t in sorted(templates, key=lambda t: t.id):
        by_family.setdefault(t.family, []).append(t)
    out = []
    for g in sorted(goals, key=lambda g: g.id):
        fams = sorted(by_family, key=lambda f: _h("family", g.id, f))[:per_goal]
        for f in fams:
            opts = by_family[f]
            t = opts[_h("template", g.id, f) % len(opts)]
            out.append(Payload(seed_id=f"{g.id}#{t.id}", goal=g.id, family=f, template=t.id,
                               domain=g.domain, type=g.type, text=fill(t.text, g.text)))
    return out


def as_items(rows: list[Payload], lang: str = "en") -> list[GuardItem]:
    """Bare payloads as training items (carrier `none`)."""
    return [GuardItem.new(text=p.text, label="injection", lang=lang, family=p.family, goal=p.goal,
                          carrier="none", source="seed", provenance={"seed_id": p.seed_id})
            for p in rows]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["check", "export"])
    ap.add_argument("--per-goal", type=int, default=PER_GOAL)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    a = ap.parse_args(argv)

    goals, p1 = parse_goals(GOALS.read_text(encoding="utf-8"))
    templates, p2 = parse_templates(FAMILIES.read_text(encoding="utf-8"))
    problems = p1 + p2 + check(goals, templates, Exclusions.from_manifest(a.manifest))
    fams = sorted({t.family for t in templates})
    print(f"{len(goals)} goals over {len({g.domain for g in goals})} domains, {len(templates)} templates "
          f"in {len(fams)} families, {len(problems)} problems")
    for p in problems:
        print("  " + p)
    if problems:
        return 1
    if a.action == "export":
        rows = payloads(goals, templates, per_goal=a.per_goal)
        EXPORT.parent.mkdir(parents=True, exist_ok=True)
        EXPORT.write_bytes("".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in rows).encode("utf-8"))
        print(f"wrote {len(rows)} English payloads to {EXPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
