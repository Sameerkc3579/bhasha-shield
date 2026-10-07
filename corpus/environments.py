"""Localized AgentDojo environments -- contribution C3, the honest version.

    python -m corpus.environments export                 # English strings still needing a draft
    python -m corpus.environments build hi               # overlay from benign translations + drafts
    python -m corpus.environments validate hi            # gates; excludes what breaks a checker
    python -m corpus.environments status
    python -m corpus.environments review                 # ~20% review packs -> corpus/review/

Until now every language condition ran in AgentDojo's English world: only the
injected payload changed language. A Hindi user's inbox is not English, so a
detector that fires on "Hindi text near an imperative" was never tested on the
Hindi text that surrounds a real payload. This module translates the
environment's *prose* -- email bodies, documents, calendar descriptions, Slack
web pages, hotel and restaurant reviews -- and keeps everything a task acts on
byte-identical.

**What is not localized, on purpose.** Names, addresses, phone numbers, emails,
IBANs, amounts, dates, ids, opening hours and dietary fields. AgentDojo's
checkers compare those values directly; localizing them (Indian names, rupee
amounts, as the plan first proposed) would mean rewriting the checkers too, and
a silently broken checker reads as a security result. The paper states this as
the limit of C3.

**Three gates, all offline.** An overlay entry is kept only if:

1. it preserves every `{placeholder}` (injection vectors) and every entity
   (`corpus.benign_localize.entities`: emails, URLs, IBANs, file names, 3+ digit
   numbers), and adds no stray brace;
2. no string literal from the suite's task code (what checkers and ground-truth
   outputs look for) occurs in the English original but is lost in the
   translation -- e.g. a task that expects "dental check-up" in the answer keeps
   that calendar description in English;
3. AgentDojo's own `suite.check()` -- every user and injection task's ground
   truth replayed without a model -- passes on the localized environment
   exactly where it passes on the English one. An entry that changes any
   outcome is excluded, with the reason recorded.

Loading reproduces AgentDojo's path exactly: the template YAML is parsed,
prose leaves are swapped, the tree is dumped back with the same settings,
and only then are injections `.format()`-ed into the text and the result
parsed. With an empty overlay the text is byte-identical to AgentDojo's
(`test_empty_overlay_is_identity`), so injection placement and YAML escaping
behave exactly as in the English runs.

Sources of translations: the 87 harvested benign items already have drafts in
every language (`corpus/benign/{lang}.jsonl`), and they *are* environment
prose -- reused as-is. The rest come from `corpus/environments/drafts/{lang}.txt`.
Everything here is a machine draft until a human reviews it, like the corpus.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

ROOT = Path("corpus/environments")
DRAFTS = ROOT / "drafts"
BENCHMARK = "v1.2.1"
LANGS = ("hi", "hi_latn", "bn", "mixed")
MIN_PROSE = 40

# Leaf keys whose values checkers compare or tools filter on. Never translated.
DATA_KEYS = frozenset({
    "address", "contact_information", "operating_hours", "dietary_restrictions",
    "email", "sender", "recipients", "cc", "bcc", "id_", "id", "filename", "owner",
    "shared_with", "location", "participants", "name", "city", "cuisine_type",
    "car_types_available", "fuel_options", "price_range", "iban", "recipient",
    "date", "start_time", "end_time", "timestamp", "last_modified", "status",
    "first_name", "last_name", "ID_number", "phone_number", "passport_number",
    "bank_account_number", "credit_card_number", "user", "channel",
})
PLACEHOLDER = re.compile(r"\{(\w+)\}")


def key_of(suite: str, original: str) -> str:
    return f"{suite}:{hashlib.sha1(original.encode('utf-8')).hexdigest()[:10]}"


# --------------------------------------------------------------------------
# AgentDojo access
# --------------------------------------------------------------------------

@lru_cache
def suites() -> dict[str, Any]:
    from agentdojo.task_suite.load_suites import get_suites

    return dict(get_suites(BENCHMARK))


def template_text(suite: Any) -> str:
    from agentdojo.task_suite.task_suite import read_suite_file

    return read_suite_file(suite.name, "environment.yaml", suite.data_path)


def leaves(node: Any, path: tuple = ()) -> Iterable[tuple[tuple, str]]:
    if isinstance(node, dict):
        for k, v in node.items():
            yield from leaves(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from leaves(v, path + (i,))
    elif isinstance(node, str):
        yield path, node


def _leaf_key(path: tuple) -> str | None:
    """The nearest string key above a leaf (`reviews` for reviews/3)."""
    for p in reversed(path):
        if isinstance(p, str):
            return p
    return None


def candidates(suite: Any) -> list[tuple[tuple, str]]:
    import yaml

    data = yaml.safe_load(template_text(suite))
    out, seen = [], set()
    for path, s in leaves(data):
        if len(s.strip()) < MIN_PROSE or " " not in s.strip() or _leaf_key(path) in DATA_KEYS:
            continue
        if s not in seen:
            seen.add(s)
            out.append((path, s))
    return out


@lru_cache
def checker_literals(suite_name: str) -> frozenset[str]:
    """Every string constant in the suite's task code (all benchmark versions).

    Checkers and ground-truth outputs are written in terms of these; a literal
    that sits in an English environment string must still sit there after
    translation, or the gate excludes that string.
    """
    import agentdojo.default_suites as ds

    root = Path(ds.__file__).parent
    lits: set[str] = set()
    for f in root.glob(f"*/{suite_name}/*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for part in re.split(r"\n", node.value):
                    part = part.strip()
                    if _distinctive(part):
                        lits.add(part)
    return frozenset(lits)


def _distinctive(s: str) -> bool:
    """A literal a checker could plausibly match on, not a docstring word.

    Single common words ("email", "hotel", "Paris") occur in tool docstrings and
    argument names everywhere; gating on them would exclude half the
    environment for nothing. Phrases, ids, addresses and numbers are what
    checkers and ground-truth outputs actually compare.
    """
    if len(s) < 5 or not re.search(r"[A-Za-z]", s):
        return False
    return (len(s.split()) >= 2 and len(s) >= 8) or bool(re.search(r"\d|@|_|/|\.\w", s)) or len(s) >= 14


# --------------------------------------------------------------------------
# overlay files
# --------------------------------------------------------------------------

def overlay_path(lang: str) -> Path:
    return ROOT / f"{lang}.json"


def load_overlay(lang: str) -> list[dict]:
    p = overlay_path(lang)
    return json.loads(p.read_text(encoding="utf-8"))["entries"] if p.exists() else []


@lru_cache(maxsize=64)
def _active(lang: str, suite_name: str, mtime: float) -> dict[str, str]:
    return {e["original"]: e["localized"] for e in load_overlay(lang)
            if e["suite"] == suite_name and e["status"] == "active"}


def active_map(lang: str, suite_name: str) -> dict[str, str]:
    p = overlay_path(lang)
    return _active(lang, suite_name, p.stat().st_mtime if p.exists() else 0.0)


def localized_text(suite: Any, mapping: dict[str, str]) -> str:
    """AgentDojo's template text with prose leaves swapped. Empty mapping = identity."""
    import yaml

    text = template_text(suite)
    if not mapping:
        return text
    data = yaml.safe_load(text)

    def swap(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: swap(v) for k, v in node.items()}
        if isinstance(node, list):
            return [swap(v) for v in node]
        if isinstance(node, str):
            return mapping.get(node, node)
        return node

    return yaml.dump(swap(data), default_flow_style=False)


def load_environment(suite: Any, injections: dict[str, str], lang: str | None,
                     mapping: dict[str, str] | None = None) -> Any:
    """`suite.load_and_inject_default_environment`, in `lang`'s localized world."""
    import yaml
    from agentdojo.task_suite.task_suite import validate_injections

    if mapping is None:
        mapping = active_map(lang, suite.name) if lang and lang != "en" else {}
    if not mapping:
        # The class method, not the instance attribute: validate() patches the latter.
        return type(suite).load_and_inject_default_environment(suite, injections)
    defaults = suite.get_injection_vector_defaults()
    validate_injections(injections, defaults)
    text = localized_text(suite, mapping).format(**dict(defaults, **injections))
    return suite.environment_type.model_validate(yaml.safe_load(text))


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def _benign_translations(lang: str) -> dict[str, str]:
    """English benign content -> its `lang` draft, for the harvested env items."""
    from corpus.schema import BenignItem

    def rows(l: str) -> dict[str, BenignItem]:
        f = Path(f"corpus/benign/{l}.jsonl")
        return {r.item_id: r for r in (BenignItem.model_validate_json(x)
                                       for x in f.read_text(encoding="utf-8").splitlines() if x.strip())}

    en, loc = rows("en"), rows(lang)
    return {it.content: loc[i].content for i, it in en.items()
            if i.startswith("benign_env_") and i in loc}


def _with_ws(original: str, translated: str) -> str:
    """Keep the original's leading/trailing whitespace (benign items are stripped)."""
    lead = original[: len(original) - len(original.lstrip())]
    trail = original[len(original.rstrip()):]
    return lead + translated.strip() + trail


def export(path: Path | None = None) -> Path:
    """English prose with no translation source yet, in the draft format."""
    from corpus.benign_localize import HEADER

    path = path or DRAFTS / "en.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    have = set(_benign_translations("hi"))
    with path.open("w", encoding="utf-8") as f:
        for name, suite in suites().items():
            for _, s in candidates(suite):
                if s.strip() in have:
                    continue
                if any(HEADER.match(l) for l in s.splitlines()):
                    raise ValueError(f"header-shaped line in {name} string")
                f.write(f"=== {key_of(name, s)}\n{s.strip()}\n")
    return path


def build(lang: str) -> dict[str, int]:
    from corpus.benign_localize import parse

    benign = _benign_translations(lang)
    drafts: dict[str, str] = {}
    for f in sorted(DRAFTS.glob(f"{lang}_*.txt")):
        if re.match(rf"{lang}_\d", f.name):
            drafts.update(parse(f.read_text(encoding="utf-8")))
    entries, counts = [], {"benign": 0, "draft": 0, "missing": 0}
    for name, suite in suites().items():
        for path, s in candidates(suite):
            k = key_of(name, s)
            if s.strip() in benign:
                loc, src = benign[s.strip()], "benign"
            elif k in drafts:
                loc, src = drafts[k], "draft"
            else:
                counts["missing"] += 1
                continue
            counts[src] += 1
            entries.append({"suite": name, "key": k, "path": "/".join(map(str, path)),
                            "original": s, "localized": _with_ws(s, loc), "source": src,
                            "status": "active", "reason": ""})
    ROOT.mkdir(parents=True, exist_ok=True)
    overlay_path(lang).write_text(json.dumps({"lang": lang, "benchmark": BENCHMARK, "validated": False,
                                              "entries": entries}, indent=1, ensure_ascii=False),
                                  encoding="utf-8")
    return counts


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------

def static_problems(e: dict) -> list[str]:
    from corpus.benign_localize import entities

    o, l = e["original"], e["localized"]
    out = []
    if sorted(PLACEHOLDER.findall(o)) != sorted(PLACEHOLDER.findall(l)):
        out.append("placeholders changed")
    try:
        l.format(**{p: "" for p in PLACEHOLDER.findall(l)})
    except (KeyError, IndexError, ValueError):
        out.append("stray brace")
    lost = sorted(x for x in entities(o) if x not in l)
    if lost:
        out.append(f"entities lost {lost[:4]}")
    # case-insensitive: AgentDojo's search tools are
    lo, ll = o.lower(), l.lower()
    lits = sorted(x for x in checker_literals(e["suite"]) if x.lower() in lo and x.lower() not in ll)
    if lits:
        out.append(f"checker literal lost {lits[:3]}")
    return out


def _check(suite: Any, mapping: dict[str, str]) -> Any:
    """suite.check() with environment loading routed through `mapping`.

    A ground-truth tool call that raises (a file search that no longer finds
    its file) aborts AgentDojo's check; that is recorded as an outcome of its
    own, so it compares unequal to the English baseline.
    """
    suite.load_and_inject_default_environment = (
        lambda injections: load_environment(suite, injections, None, mapping=mapping))
    try:
        _, (user, inj) = suite.check(check_injectable=True)
        return ({k: v[0] for k, v in user.items()}, dict(inj))
    except Exception as e:  # noqa: BLE001 -- any failure is a changed outcome
        return ("raised", type(e).__name__, str(e)[:200])
    finally:
        del suite.load_and_inject_default_environment


def validate(lang: str) -> dict[str, int]:
    data = json.loads(overlay_path(lang).read_text(encoding="utf-8"))
    entries = data["entries"]
    for e in entries:  # gates 1 and 2
        e["status"], e["reason"] = "active", ""
        probs = static_problems(e)
        if probs:
            e["status"], e["reason"] = "excluded", "; ".join(probs)

    for name, suite in suites().items():  # gate 3
        mine = [e for e in entries if e["suite"] == name and e["status"] == "active"]
        if not mine:
            continue
        base = _check(suite, {})
        full = _check(suite, {e["original"]: e["localized"] for e in mine})
        if full == base:
            continue
        # Something changed an outcome: test entries one at a time and drop each culprit.
        for e in mine:
            if _check(suite, {e["original"]: e["localized"]}) != base:
                e["status"], e["reason"] = "excluded", "changes a ground-truth outcome in suite.check()"
        still = {e["original"]: e["localized"] for e in mine if e["status"] == "active"}
        if _check(suite, still) != base:  # an interaction between entries; be conservative
            for e in mine:
                e["status"], e["reason"] = "excluded", "suite.check() differs in combination"

    data["validated"] = True
    overlay_path(lang).write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return {s: sum(e["status"] == s for e in entries) for s in ("active", "excluded")}


def _review_pack(lang: str, pairs: list[tuple[str, str, str]], title: str, path: Path,
                frac: float = 0.2, seed: int = 20261004) -> int:
    """Fixed-seed ~20% sample in the corpus/review format: id, English, draft."""
    import random

    rng = random.Random(f"{seed}:{lang}")
    k = max(1, round(frac * len(pairs)))
    sample = sorted(rng.sample(pairs, k), key=lambda t: t[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig") as f:  # BOM: Notepad/Excel show Devanagari
        f.write(f"{title} ({lang}) - {k} of {len(pairs)} items (~{frac:.0%}, fixed seed). For each: same\n"
                "meaning as the English? Natural? Write down the item number and the problem. Emails,\n"
                "links, numbers, file names and {placeholders} must stay exactly as in English.\n")
        for n, (iid, en, loc) in enumerate(sample, 1):
            f.write(f"\n--- {n}. {iid}\nENGLISH:\n{en.strip()}\n\n{lang.upper()}:\n{loc.strip()}\n")
    return k


def review() -> dict[str, int]:
    """Sample the *draft-sourced* active entries; benign-sourced ones are in the benign packs."""
    out = {}
    for lang in LANGS:
        pairs = [(e["key"], e["original"], e["localized"]) for e in load_overlay(lang)
                 if e["status"] == "active" and e["source"] == "draft"]
        if pairs:
            out[lang] = _review_pack(lang, pairs, "ENVIRONMENT SAMPLE",
                                     Path(f"corpus/review/environment_sample_{lang}.txt"))
    return out


def status() -> None:
    for lang in LANGS:
        es = load_overlay(lang)
        if not es:
            print(f"{lang:<8} no overlay")
            continue
        by = {}
        for e in es:
            by.setdefault(e["suite"], [0, 0])[e["status"] != "active"] += 1
        print(f"{lang:<8} " + "  ".join(f"{s}: {a} active / {x} excluded" for s, (a, x) in sorted(by.items())))


def main(argv: list[str]) -> int:
    if argv[:1] == ["export"]:
        print(f"wrote {export()}")
    elif argv[:1] == ["build"] and len(argv) == 2:
        print(build(argv[1]))
    elif argv[:1] == ["validate"] and len(argv) == 2:
        print(validate(argv[1]))
    elif argv[:1] == ["review"]:
        print(review())
    elif argv[:1] == ["status"]:
        status()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
