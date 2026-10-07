"""Append-only writer for corpus/provenance.jsonl.

Every non-English item gets a row when it is drafted, and another when a human
verifies or repairs it. Rows are never edited: the history of who touched what,
and when, is the audit trail reviewers ask for (and what makes the corpus
citable on its own).
"""

from __future__ import annotations

import datetime
from pathlib import Path

from corpus.schema import Lang, ProvenanceRecord, TranslationMethod

PATH = Path("corpus/provenance.jsonl")


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def record(item_id: str, lang: Lang, method: TranslationMethod, *, source_item_id: str | None = None,
           verified_by: str | None = None, verified_at: str | None = None, notes: str = "",
           path: Path = PATH) -> ProvenanceRecord:
    row = ProvenanceRecord(item_id=item_id, lang=lang, method=method, source_item_id=source_item_id,
                           verified_by=verified_by, verified_at=verified_at, notes=notes)
    with path.open("a", encoding="utf-8") as f:
        f.write(row.model_dump_json() + "\n")
    return row


def history(item_id: str, path: Path = PATH) -> list[ProvenanceRecord]:
    if not path.exists():
        return []
    return [r for r in (ProvenanceRecord.model_validate_json(l)
                        for l in path.read_text(encoding="utf-8").splitlines() if l.strip())
            if r.item_id == item_id]
