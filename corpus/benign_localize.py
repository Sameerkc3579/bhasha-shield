"""Assemble a localized benign corpus from translation drafts, with checks.

    python -m corpus.benign_localize export en drafts/en.txt
    python -m corpus.benign_localize import hi --method machine drafts/hi_1.txt drafts/hi_2.txt ...

Drafts use a plain format a translator (human or model) can edit safely --
no JSON escaping:

    === benign_env_workspace_0000
    <content, any number of lines>
    === benign_env_workspace_0001
    ...

`import` refuses to write unless every English item is present exactly once
and every *entity* survived translation verbatim: email addresses, URLs,
IBAN-like account numbers, file names and numbers of 3+ digits. Those are what
tools and checkers act on, and a translation that mangles one changes what the
item *is* rather than what language it is in -- the same rule as the goals.

Item ids are kept identical to English, so FPR can be compared item by item
(paired) across languages. Metadata (tool_name, adversarial_benign, source) is
copied from the English item; only `content` and `lang` change. One provenance
row is appended per item.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from corpus.provenance import now, record
from corpus.schema import BenignItem

BENIGN = Path("corpus/benign")
HEADER = re.compile(r"^=== (\S+)$")

ENTITY = re.compile(
    r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"          # email
    r"|https?://\S+|www\.\S+"                 # URL
    r"|\b[A-Z]{2}\d{2}[A-Z0-9 ]{8,30}\d\b"    # IBAN-ish
    r"|[\w-]+\.(?:docx|xlsx|txt|yaml|py|pdf|csv)\b"  # file name
    r"|\d[\d,.]{2,}\d|\b\d{3,}\b"             # numbers with 3+ digits
)


def parse(text: str) -> dict[str, str]:
    items: dict[str, str] = {}
    cur: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        m = HEADER.match(line)
        if m:
            if cur is not None:
                items[cur] = "\n".join(buf).rstrip("\n")
            cur, buf = m.group(1), []
            if cur in items:
                raise ValueError(f"{cur} appears twice")
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        items[cur] = "\n".join(buf).rstrip("\n")
    return items


def entities(text: str) -> set[str]:
    return {e.rstrip(".,;:)'\"।!?") for e in ENTITY.findall(text)}


def check(en: dict[str, BenignItem], drafts: dict[str, str]) -> list[str]:
    problems = [f"missing {k}" for k in en if k not in drafts]
    problems += [f"unknown id {k}" for k in drafts if k not in en]
    for k, text in drafts.items():
        if k not in en:
            continue
        if not text.strip():
            problems.append(f"{k}: empty")
            continue
        lost = sorted(e for e in entities(en[k].content) if e not in text)
        if lost:
            problems.append(f"{k}: entities lost in translation {lost}")
    return problems


def load_en() -> dict[str, BenignItem]:
    rows = [BenignItem.model_validate_json(l)
            for l in (BENIGN / "en.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r.item_id: r for r in rows}


def export(path: Path) -> int:
    en = load_en()
    with path.open("w", encoding="utf-8") as f:
        for item in en.values():
            if any(HEADER.match(l) for l in item.content.splitlines()):
                raise ValueError(f"{item.item_id} contains a header-shaped line")
            f.write(f"=== {item.item_id}\n{item.content}\n")
    return len(en)


def import_drafts(lang: str, files: list[Path], *, method: str = "machine",
                  source_lang: str = "en", notes: str = "") -> int:
    en = load_en()
    drafts: dict[str, str] = {}
    for f in files:
        part = parse(f.read_text(encoding="utf-8"))
        dup = drafts.keys() & part.keys()
        if dup:
            raise ValueError(f"{sorted(dup)} in more than one draft file")
        drafts.update(part)
    problems = check(en, drafts)
    if problems:
        raise SystemExit("refusing to write:\n  " + "\n  ".join(problems))

    out = BENIGN / f"{lang}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for k, item in en.items():  # English order, so files diff cleanly
            f.write(item.model_copy(update={"lang": lang, "content": drafts[k]}).model_dump_json() + "\n")
    stamp = now()
    for k in en:
        record(f"benign/{k}", lang, method, source_item_id=f"benign/{k}@{source_lang}",
               notes=notes or f"draft imported {stamp}; not human-verified")
    return len(en)


def main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[0] == "export" and argv[1] == "en":
        print(f"wrote {export(Path(argv[2]))} items to {argv[2]}")
        return 0
    if len(argv) >= 3 and argv[0] == "import":
        lang, rest = argv[1], argv[2:]
        method, source, notes = "machine", "en", ""
        while rest and rest[0].startswith("--"):
            flag, val, rest = rest[0], rest[1], rest[2:]
            if flag == "--method":
                method = val
            elif flag == "--source-lang":
                source = val
            elif flag == "--notes":
                notes = val
        n = import_drafts(lang, [Path(p) for p in rest], method=method, source_lang=source, notes=notes)
        print(f"wrote {n} items to corpus/benign/{lang}.jsonl")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
