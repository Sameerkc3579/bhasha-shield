"""Grow the benign corpus past the 127 harvestable items, toward the plan's ~300.

    python -m corpus.benign_extend check            # validate every staged file, write nothing
    python -m corpus.benign_extend apply            # append staged items to corpus/benign/*.jsonl
    python -m corpus.benign_extend review           # ~20% fixed-seed review packs -> corpus/review/

The 87 harvested items are the whole pool of ordinary text in the four
AgentDojo environments, so growth has to be hand-written. Two kinds, both
English-first and staged in `corpus/benign_extra/`:

- **adversarial-benign** (`benign_adv_0040`+): more of the content that looks
  like an attack and is not -- now including the India-specific shapes a Hindi
  or Bengali user's tools actually return (society maintenance notices, bank
  KYC reminders, UPI receipts, school circulars), where an English-trained
  detector has the least to go on.
- **ordinary hand-written** (`benign_hw_0000`+): everyday tool output written
  in an Indian setting -- names, rupee amounts, cities, DD/MM/YYYY dates.
  This is the benign half of C3: content that looks like where these users
  live, not like AgentDojo's Emma Johnson.

Staged English uses a header carrying the metadata:

    === <item_id> | <tool_name> | adv|ord | <category>
    <content>

Localized drafts (`{lang}.txt`) use the plain `=== <item_id>` header of
`corpus/benign_localize.py`, and pass the same check: every email, URL, IBAN,
file name and 3+ digit number survives verbatim.

`apply` is idempotent: ids already present in a corpus file are skipped, and a
provenance row is written only for an item actually appended. Existing items,
their order and their provenance are never touched.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from corpus.benign_localize import BENIGN, check, parse
from corpus.provenance import now, record
from corpus.schema import BenignItem

STAGE = Path("corpus/benign_extra")
EN_HEADER = re.compile(r"^=== (\S+) \| (\S+) \| (adv|ord) \| (\S+)$")
LANG_METHOD = {"hi": "machine", "hi_latn": "transliterated", "bn": "machine", "mixed": "machine"}
LANG_SOURCE = {"hi": "en", "hi_latn": "hi", "bn": "en", "mixed": "en"}


def parse_en(text: str) -> list[BenignItem]:
    items: list[BenignItem] = []
    cur: dict | None = None
    buf: list[str] = []

    def close() -> None:
        if cur is not None:
            items.append(BenignItem(**cur, content="\n".join(buf).strip("\n")))

    for line in text.splitlines():
        if line.startswith("=== "):
            m = EN_HEADER.match(line)
            if not m:
                raise ValueError(f"bad header: {line!r}")
            close()
            iid, tool, kind, cat = m.groups()
            cur = {"item_id": iid, "lang": "en", "tool_name": tool, "adversarial_benign": kind == "adv",
                   "source": f"hand-written/{cat}"}
            buf = []
        elif cur is not None:
            buf.append(line)
    close()
    return items


def staged_en() -> list[BenignItem]:
    items: list[BenignItem] = []
    for f in sorted(STAGE.glob("en_*.txt")):
        items += parse_en(f.read_text(encoding="utf-8"))
    ids = [i.item_id for i in items]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"duplicate staged ids: {sorted(dup)}")
    return items


def staged_lang(lang: str) -> dict[str, str]:
    drafts: dict[str, str] = {}
    # `hi_*` would also match `hi_latn_*`: require a digit right after the language
    for f in sorted(p for p in STAGE.glob(f"{lang}_*.txt") if re.match(rf"{lang}_\d", p.name)):
        part = parse(f.read_text(encoding="utf-8"))
        if drafts.keys() & part.keys():
            raise ValueError(f"{sorted(drafts.keys() & part.keys())} staged twice for {lang}")
        drafts.update(part)
    return drafts


def _existing(lang: str) -> set[str]:
    f = BENIGN / f"{lang}.jsonl"
    if not f.exists():
        return set()
    return {BenignItem.model_validate_json(l).item_id
            for l in f.read_text(encoding="utf-8").splitlines() if l.strip()}


def validate() -> dict[str, list[str]]:
    from corpus.benign_build import MAX_CHARS

    en = {i.item_id: i for i in staged_en()}
    problems: dict[str, list[str]] = {"en": []}
    for i in en.values():
        if not (100 <= len(i.content) <= MAX_CHARS):
            problems["en"].append(f"{i.item_id}: length {len(i.content)}")
    taken = _existing("en") & en.keys()
    problems["en"] += [f"{k}: id already in en.jsonl with different content" for k in taken
                       if k in en and _content("en", k) != en[k].content]
    for lang in LANG_METHOD:
        drafts = staged_lang(lang)
        problems[lang] = check(en, drafts) if drafts else ["no drafts staged"]
        same = [k for k in drafts if k in en and drafts[k].strip() == en[k].content.strip()]
        problems[lang] += [f"{k}: identical to English" for k in same]
    return problems


def _content(lang: str, item_id: str) -> str | None:
    for l in (BENIGN / f"{lang}.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            it = BenignItem.model_validate_json(l)
            if it.item_id == item_id:
                return it.content
    return None


def apply() -> dict[str, int]:
    problems = validate()
    bad = {k: v for k, v in problems.items() if v}
    if bad:
        raise SystemExit("refusing to write:\n" + "\n".join(f"  [{k}] {p}" for k, v in bad.items() for p in v))
    en = staged_en()
    written: dict[str, int] = {}
    stamp = now()
    for lang in ("en", *LANG_METHOD):
        have = _existing(lang)
        drafts = None if lang == "en" else staged_lang(lang)
        new = [i for i in en if i.item_id not in have]
        with (BENIGN / f"{lang}.jsonl").open("a", encoding="utf-8") as f:
            for i in new:
                item = i if lang == "en" else i.model_copy(update={"lang": lang, "content": drafts[i.item_id]})
                f.write(item.model_dump_json() + "\n")
        if lang != "en":
            for i in new:
                record(f"benign/{i.item_id}", lang, LANG_METHOD[lang],
                       source_item_id=f"benign/{i.item_id}@{LANG_SOURCE[lang]}",
                       notes=f"hand-written extension, draft imported {stamp}; not human-verified")
        written[lang] = len(new)
    return written


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
    en = {i.item_id: i.content for i in staged_en()}
    out = {}
    for lang in LANG_METHOD:
        d = staged_lang(lang)
        out[lang] = _review_pack(lang, [(k, en[k], d[k]) for k in en if k in d],
                                 "BENIGN EXTENSION SAMPLE", Path(f"corpus/review/benign_extra_sample_{lang}.txt"))
    return out


def main(argv: list[str]) -> int:
    if argv[:1] == ["check"]:
        problems = validate()
        n = len(staged_en())
        for lang, ps in problems.items():
            print(f"[{lang}] {'ok' if not ps else f'{len(ps)} problem(s)'}")
            for p in ps[:30]:
                print("   ", p)
        print(f"{n} staged English items")
        return 1 if any(problems.values()) else 0
    if argv[:1] == ["review"]:
        print(review())
        return 0
    if argv[:1] == ["apply"]:
        for lang, n in apply().items():
            print(f"appended {n} items to corpus/benign/{lang}.jsonl")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
